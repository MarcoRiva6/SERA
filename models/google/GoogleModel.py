import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import google.genai
from google.genai.types import BatchJob, ThinkingConfig, GenerationConfig, Content, Part

from models.model import Model, write_jsonl, _split_queries
from queries.test import Query

def convert_schema_to_gemini(schema):
    if isinstance(schema, dict):
        new_schema = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                new_schema[key] = value.upper()
            else:
                new_schema[key] = convert_schema_to_gemini(value)
        return new_schema

    elif isinstance(schema, list):
        return [convert_schema_to_gemini(item) for item in schema]

    else:
        return schema

@dataclass
class GoogleModel(Model):
    client: google.genai.Client = None
    batch_max_tokens: int = 0
    supports_batched: bool = False
    supports_reasoning: bool = False
    @dataclass
    class Params(Model.Params):
        batched: bool = True
        reasoning: bool = True
    params: Params = field(default_factory=Params)

    def __init_google_client(self):
        # Load API key from .env file
        self._load_env_file()
        api_key = os.getenv("GOOGLE_API_KEY")
        from google.genai import Client
        return Client(api_key=api_key)

    def _count_tokens(self, prompt: str) -> int:
        if self.client is None:
            self.client = self.__init_google_client()
        response = self.client.models.count_tokens(contents=prompt, model=self.name_api)
        return response.total_tokens

    def _submit_direct_inline(self, prompt: str) -> str | None:
        MAX_RETRIES = 3
        TIMEOUT = 600
        RETRY_DELAY = 40
        QUOTA_WAIT_HOURS=8
        REASONING=False

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
                return True  # Non fare retry automatico, attendi

            # Gestione altri errori temporanei
            temporary_errors = [
                "timeout", "connection", "network", "service unavailable",
                "temporarily unavailable", "rate limit", "502", "503", "504",
                "GenerateContentPaidTierInputTokensPerModelPerMinute"
            ]

            is_temporary = any(temp_error in error_msg_lower for temp_error in temporary_errors)

            if is_temporary and retry_count < MAX_RETRIES:
                print(f"Errore temporaneo rilevato (retry {retry_count + 1}/{MAX_RETRIES}): {error_msg}")
                print(f"Attendo {RETRY_DELAY} secondi prima del retry...")
                time.sleep(RETRY_DELAY)
                return True

            return False

        from google.genai import Client
        from google.genai import types
        if self.client is None:
            self.client = self.__init_google_client()
        client: Client = self.client

        retry_count = 0
        while retry_count <= MAX_RETRIES:
            try:
                response = client.models.generate_content(
                    model=self.name_api,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)) if REASONING else None,
                )
                return response.text.strip()
            except Exception as e:
                error_msg = str(e)
                print(f"errore nella query a Gemini: {error_msg}")

                # Se non è l'ultimo tentativo, controlla se fare retry
                if retry_count < MAX_RETRIES and handle_api_error(error_msg, retry_count):
                    retry_count += 1
                    continue
                else:
                    return None

    def _submit_direct_batched(self, folder: Path, queries: list[Query]) -> bool:
        poll_interval = 60 #seconds
        timeout = 86400  #seconds (24 hours)
        batch_id_path = folder / "batch_id.txt"
        input_path = folder / "batch_input.jsonl"
        output_path = folder / "batch_output.jsonl"
        error_path = folder / "batch_error.jsonl"
        batch_token_usage_path = folder / "batch_token_usage.txt"

        batch_queries: dict[str, Query] = {hashlib.md5(q.prompt.encode()).hexdigest(): q for q in queries}

        if os.path.exists(output_path):
            print(f"Using existing batch output file...")
        else:
            if os.path.exists(batch_id_path): # recover existing batch
                with open(batch_id_path, "r") as f:
                    batch_id = f.read().strip()
                print(f"Trying to recover already submitted batch {batch_id}...")
            else:
                requests: list[dict] = []

                thinking_config = ThinkingConfig(include_thoughts=False)
                if self.params.reasoning and self.supports_reasoning:
                    thinking_config.thinking_budget = -1 # auto hybrid thinking
                else:
                    thinking_config.thinking_budget = 0
                gen_config = GenerationConfig(max_output_tokens=self.max_tokens, thinking_config=thinking_config)
                for q_key, q in batch_queries.items():
                    content = Content(parts=[Part(text=q.prompt)], role="user")
                    req_gen_config = gen_config.model_copy()
                    if q.response_json_schema is not None and q.response_json_schema != '':
                        req_gen_config.response_mime_type = "application/json"
                        req_gen_config.response_schema = convert_schema_to_gemini(q.response_json_schema)
                    r = {
                        "key": q_key,
                        "request": {
                            "contents": [content.to_json_dict()],
                            "generation_config": req_gen_config.to_json_dict()
                        }
                    }
                    requests.append(r)

                folder.mkdir(parents=True, exist_ok=True)
                write_jsonl(input_path, requests)

                # Upload the file to the File API
                file_resp = self.client.files.upload(file=input_path, config={'mime_type': 'jsonl'})
                file_id = file_resp.name

                # Create the batch job
                batch: BatchJob = self.client.batches.create(model=self.name_api, src=file_id)
                batch_id = batch.name
                batch_id_path.write_text(batch_id)

            start_time = time.time()
            while True:
                b = self.client.batches.get(name=batch_id)
                status = b.state.name
                print(f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S')}] Batch {batch_id} status: {status}")

                failing_states = {'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
                successfully_completed_states = {'JOB_STATE_SUCCEEDED'}
                completed_states = successfully_completed_states | failing_states
                if status in completed_states:
                    if status not in successfully_completed_states:
                        print(f"Warning: Batch {batch_id} completed with state: {status}. error file ID {b.error}")
                    break
                if (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

                if self.params.no_waiting:
                    return False
                time.sleep(poll_interval)

            # download result file
            output_file_id = b.dest.file_name
            file_content = self.client.files.download(file=output_file_id)
            output_path.write_text(file_content.decode('utf-8'))

        total_token_consumed = 0
        # 5. Parse the results
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                response = json.loads(line)
                q: Query = batch_queries[response['key']]
                if response['response']['candidates'][0]['finishReason'] != 'STOP':
                    q_id = getattr(q, 'id', response['key'])
                    print(f"Warning: Response for query {q_id} didn't finish.")
                try:
                    q.response = response['response']['candidates'][0]['content']['parts'][0]['text']
                except KeyError:
                    q.response = None
                total_token_consumed += response['response']['usageMetadata']['totalTokenCount']

        print(f"Total tokens consumed in batch: {total_token_consumed}")
        batch_token_usage_path.write_text(f"{total_token_consumed}")

        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")
        return True

    def _init_model(self) -> None:
        self.client = self.__init_google_client()

    def _finish_model(self) -> None:
        self.client.close()

    def _submit_direct(self, queries: list[Query]):
        if not self.supports_batched and self.params.batched:
            print(f"Model {self.name} does not support batched submissions. Submitting inline.")
            self.params.batched = False
        if self.params.batched:
            batches = _split_queries(self._count_tokens, self.batch_max_tokens, queries) if self.batch_max_tokens else None
            for i, b in enumerate(batches):
                print(f"Submitting batch {i + 1}/{len(batches)} with {len(b)} queries...")
                if not self._submit_direct_batched(self.run_folder / f"batch_{i + 1}", b):
                    print("Not waiting for submission to complete...")
                    return
        else:
            for q in queries:
                q.response = self._submit_direct_inline(q.prompt)

