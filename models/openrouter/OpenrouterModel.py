import os
from dataclasses import dataclass

import requests

from models.model import ModelParams, Model, SubmissionError
from queries.test import DirectQuery

import requests
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type, TryAgain

# 1. Definiamo le eccezioni specifiche di requests
ERRORI_DI_RETE_REQUESTS = (
    requests.exceptions.ConnectionError,  # Caduta di linea, DNS non risolto, server irraggiungibile
    requests.exceptions.Timeout,          # Il server ci mette troppo a rispondere
    requests.exceptions.HTTPError         # Errori HTTP (es. 500, 502, 503) generati da raise_for_status()
)

@dataclass
class OpenrouterModel(Model[ModelParams]):
    reasoning: bool = False

    # 2. Applichiamo il decoratore
    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ERRORI_DI_RETE_REQUESTS),
        before_sleep=lambda retry_state: print(f"Errore di connessione. Ritento tra 10s... (Tentativo {retry_state.attempt_number})")
    )
    def _submit_prompt(self, prompt: str, schema: dict | None) -> tuple[str|None, int]:
        schema_dict = {
            "type": "json_schema",
            "json_schema": {
                "name": "schema",
                "strict": True,
                "schema": schema,
            },
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },

            json={
                "model": self.name_api,
                "provider": {"only": ["novita"], "require_parameters": True},
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.params.temperature if self.params.temperature != -1 else None,
                "reasoning": {"effort": "high" if self.reasoning else "none","enabled": self.reasoning},
                "response_format": schema_dict if schema is not None else "text",
                "plugins": [
                    {"id": "response-healing"}
                ] if schema is not None else None,
            },
        )

        response.raise_for_status()

        res = response.json()
        print(res)
        try:
            error = res["error"]
            raise TryAgain
        except KeyError:
            pass

        try:
            answer = res["choices"][0]["message"]["content"]
            if answer is None or answer == "":
                raise TryAgain
            tokens = res["usage"]["total_tokens"]
        except (KeyError, IndexError):
            raise TryAgain

        return answer, tokens

    def _submit_direct_query_inline(self, query: DirectQuery) -> None:
        query.response, query.tokens = self._submit_prompt(query.prompt, query.response_json_schema.model_json_schema())