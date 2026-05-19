import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import Client
from google.genai.types import BatchJob, ThinkingConfig, GenerationConfig, Content, Part, GenerateContentConfig

from models.model import _split_queries, SubmissionError, BatchableModel, BatchableModelParams
from models.together.TogetherModel import T_BatchID
from queries.test import Query, DirectQuery

type QIID = str

@dataclass
class GoogleModel(BatchableModel[QIID, Any, BatchableModelParams]):
    client: Client = None
    batch_max_tokens: int = 0
    supports_batched: bool = False
    reasoning: bool = False

    def __init_google_client(self):
        # Load API key from .env file
        self._load_env_file()
        api_key = os.getenv("GOOGLE_API_KEY")
        from google.genai import Client
        return Client(api_key=api_key)

    def _get_lotus_params(self) -> tuple[str,str|None]:
        return f"gemini/{self.name_api}", None

    def _count_tokens(self, prompt: str) -> int:
        if self.client is None:
            self.client = self.__init_google_client()
        response = self.client.models.count_tokens(contents=prompt, model=self.name_api)
        return response.total_tokens

    def _submit_direct_query_inline(self, query: DirectQuery) -> None:
        query.response, query.tokens = self._submit_prompt(query.prompt, query.response_json_schema.model_json_schema() if query.response_json_schema else None)

    def _submit_prompt(self, prompt: str, schema: dict | None) -> tuple[str | None, int]:
        MAX_RETRIES = 5
        RETRY_DELAY = 40
        QUOTA_WAIT_HOURS=8

        def handle_api_error(error_msg: str, retry_count: int) -> bool:
            """
            Gestisce gli errori dell'API e decide se fare retry.
            :param error_msg: Messaggio di errore
            :param retry_count: numero di retry già effettuati
            :returns: True se dovrebbe fare retry, False altrimenti
            """
            error_msg_lower = error_msg.lower()

            # Gestione quota exceeded
            if "429" in error_msg and "quota" in error_msg_lower and 'perday' in error_msg_lower:
                print(f"Limite quota raggiunto. Attendo {QUOTA_WAIT_HOURS} ore...")
                time.sleep(QUOTA_WAIT_HOURS * 60 * 60)
                raise SubmissionError("Superata la quota di Gemini")
                # return True # Non fare retry automatico, attendi

            # Gestione altri errori temporanei
            temporary_errors = [
                "timeout", "connection", "network", "service unavailable",
                "temporarily unavailable", "rate limit", "502", "503", "504",
                "GenerateContentPaidTierInputTokensPerModelPerMinute"
            ]

            is_temporary = any(temp_error in error_msg_lower for temp_error in temporary_errors)

            if is_temporary and retry_count < MAX_RETRIES:
                print(f"Errore temporaneo rilevato (retry {retry_count + 1}/{MAX_RETRIES})")
                print(f"Attendo {RETRY_DELAY} secondi prima del retry...")
                time.sleep(RETRY_DELAY)
                return True

            return False

        if self.client is None:
            self.client = self.__init_google_client()
        client: Client = self.client
        # con i modelli V3 si setta in un altro modo
        thinking_config = ThinkingConfig(include_thoughts=False)
        if self.reasoning:
            thinking_config.thinking_budget = -1 # auto hybrid thinking
        else:
            thinking_config.thinking_budget = 0
        gen_config = GenerateContentConfig(max_output_tokens=self.max_tokens, thinking_config=thinking_config,
                                      temperature=self.params.temperature if self.params.temperature != -1 else None,
                                      response_mime_type='application/json', response_json_schema=schema)

        retry_count = 0
        while retry_count <= MAX_RETRIES:
            try:
                response = client.models.generate_content(
                    model=self.name_api,
                    contents=prompt,
                    config=gen_config)
                answer: str|None = response.text.strip() if response.text is not None else None
                if response.usage_metadata is not None:
                    if response.usage_metadata.total_token_count is not None:
                        tokens: int = response.usage_metadata.total_token_count
                    else:
                        tokens: int = 0
                else:
                    tokens: int = 0
                return answer, tokens
            except Exception as e:
                error_msg = str(e)
                print(f"errore nella query a Gemini: {error_msg}")

                # Se non è l'ultimo tentativo, controlla se fare retry
                if retry_count < MAX_RETRIES and handle_api_error(error_msg, retry_count):
                    retry_count += 1
                    continue
                else:
                    raise SubmissionError("Gemini non risponde")

    def _process_batch_output(self, batch_queries: dict[str, DirectQuery], output_path: Path, batch_token_usage_path: Path):
        total_token_consumed = 0
        # 5. Parse the results
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                response = json.loads(line)
                q: Query = batch_queries[response['key']]
                q_id = getattr(q, 'id', response['key'])
                try:
                    if response['response']['candidates'][0]['finishReason'] != 'STOP':
                        print(f"Warning: Response for query {q_id} didn't finish.")
                    tokens_consumed = response['response']['usageMetadata']['totalTokenCount']
                    q.response = response['response']['candidates'][0]['content']['parts'][0]['text']
                    q.tokens = tokens_consumed
                except KeyError:
                    q.response = ''
                    print(f"Warning: Coudn't get response for query {q_id}.")
                total_token_consumed += tokens_consumed

        print(f"Total tokens consumed in batch: {total_token_consumed}")
        batch_token_usage_path.write_text(f"{total_token_consumed}")

        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")

    def _download_batch(self, batch_id: T_BatchID, output_path: Path):
        b = self.client.batches.get(name=batch_id)
        output_file_id = b.dest.file_name
        file_content = self.client.files.download(file=output_file_id)
        output_path.write_text(file_content.decode('utf-8'))

    def _check_batch(self, batch_id, error_path: Path) -> bool:
        b = self.client.batches.get(name=batch_id)
        status = b.state.name

        failing_states = {'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        successfully_completed_states = {'JOB_STATE_SUCCEEDED'}
        completed_states = successfully_completed_states | failing_states
        if status in completed_states:
            if status not in successfully_completed_states:
                print(f"Warning: Batch {batch_id} completed with state: {status}. error file ID {b.error}")
            return True
        else:
            return False

    def _send_batch(self, input_file_path: Path) -> Any:
        # Upload the file to the File API
        file_resp = self.client.files.upload(file=input_file_path, config={'mime_type': 'jsonl'})
        file_id = file_resp.name

        # Create the batch job
        batch: BatchJob = self.client.batches.create(model=self.name_api, src=file_id)
        return batch.name

    def _batch_input_to_qiids(self, batch_input_path: Path) -> list[QIID]:
        qiids_list: list[QIID] = []
        with batch_input_path.open("r", encoding="utf-8") as f:
            for line in f:
                batch_q = json.loads(line)
                qiids_list.append(batch_q['key'])

        return qiids_list

    def _build_batch_input(self, batch_queries: dict[str, DirectQuery]) -> list[dict]:
        requests: list[dict] = []
        # con i modelli V3 si setta in un altro modo
        thinking_config = ThinkingConfig(include_thoughts=False)
        if self.reasoning:
            thinking_config.thinking_budget = -1  # auto hybrid thinking
        else:
            thinking_config.thinking_budget = 0
        gen_config = GenerationConfig(max_output_tokens=self.max_tokens, thinking_config=thinking_config,
                                      temperature=self.params.temperature if self.params.temperature != -1 else None)
        for q_key, q in batch_queries.items():
            content = Content(parts=[Part(text=q.prompt)], role="user")
            req_gen_config = gen_config.model_copy()
            if q.response_json_schema is not None:
                req_gen_config.response_mime_type = "application/json"
                req_gen_config.response_json_schema = q.response_json_schema.model_json_schema()
            r = {
                "key": q_key,
                "request": {
                    "contents": [content.to_json_dict()],
                    "generation_config": req_gen_config.to_json_dict()
                }
            }
            requests.append(r)
        return requests

    def _init_model(self) -> None:
        self.client = self.__init_google_client()

    def _finish_model(self) -> None:
        self.client.close()

    def _split_batches(self, queries: list[DirectQuery]) -> list[list[DirectQuery]]:
        return _split_queries(self._count_tokens, self.batch_max_tokens, queries) if self.batch_max_tokens else None
