import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import together
from pydantic import BaseModel
from together.types import ChatCompletionResponse
from tqdm import tqdm

from models.model import Model, write_jsonl, SubmissionError
from queries.test import Query, DirectQuery


@dataclass
class TogetherModel(Model):
    client: together.Client = None
    supports_batched: bool = True
    supports_structured_output: bool = False
    @dataclass
    class Params(Model.Params):
        batched: bool = True
        temperature: float = -1
    params: Params = field(default_factory=Params)

    def _init_model(self) -> None:
        # Load API key from .env file
        self._load_env_file()
        api_key = os.getenv("TOGETHER_API_KEY")
        from together import Together
        self.client = Together(api_key=api_key)

    def _get_lotus_params(self) -> tuple[str,str|None]:
        return f"together_ai/{self.name_api}", None

    def _split_batches(self, queries: list[Query], total_prompt_length: int = 85000000) -> list[list[Query]]:
        batches = []
        current_batch = []
        current_length = 0

        for q in queries:
            prompt_length = len(q.prompt)
            if (current_length + prompt_length) > total_prompt_length and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_length = 0
            current_batch.append(q)
            current_length += prompt_length

        if current_batch:
            batches.append(current_batch)

        return batches

    def _submit_prompt(self, prompt: str, schema: BaseModel|None) -> str|None:
        from together import Together

        client = Together()

        answer = client.chat.completions.create(
            model=self.name_api,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=False,
            temperature=self.params.temperature if self.params.temperature != -1 else None,
            response_format=schema.model_json_schema() if schema is not None else None
        )
        if not isinstance(answer, ChatCompletionResponse):
            raise NotImplementedError("Received a streaming response, which is not supported")

        message = answer.choices[0].message
        if message is None or isinstance(message.content, list):
            return None
        return message.content

    def _submit_direct_query_inline(self, query: DirectQuery) -> None:
        query.response = self._submit_prompt(query.prompt, query.response_json_schema)

    def _submit_direct_queries_batched(self, folder: Path, queries: list[DirectQuery]) -> bool:
        poll_interval = 60 #seconds
        timeout = 86400  #seconds (24 hours)
        batch_id_path = folder / "batch_id.txt"
        input_path = folder / "batch_input.jsonl"
        output_path = folder / "batch_output.jsonl"
        error_path = folder / "batch_error.jsonl"
        batch_token_usage_path = folder / "batch_token_usage.txt"

        batch_queries: dict[str, DirectQuery] = {hashlib.md5(q.prompt.encode()).hexdigest(): q for q in queries}

        if os.path.exists(output_path):
            print(f"Using existing batch output file...")
        else:
            if os.path.exists(batch_id_path): # recover existing batch
                with open(batch_id_path, "r") as f:
                    batch_id = f.read().strip()
                print(f"Trying to recover already submitted batch {batch_id}...")
            else:
                requests = []
                for q_key, q in batch_queries.items():
                    r = {
                        "custom_id": q_key,
                        "body": {
                            "model": self.name_api,
                            "messages": [{"role": "user", "content": q.prompt}],
                        },
                        "max_tokens": self.max_tokens
                    }
                    if q.response_json_schema is not None and q.response_json_schema != '':
                        r['body']['response_format'] = {"type": "json_schema", "schema": q.response_json_schema}
                    if self.params.temperature != -1:
                        r['temperature'] = self.params.temperature
                    requests.append(r)

                # 1. Write requests to a .jsonl file
                folder.mkdir(parents=True, exist_ok=True)
                write_jsonl(input_path, requests)

                # 2. Upload the batch file
                file_resp = self.client.files.upload(file=input_path, purpose="batch-api")
                file_id = file_resp.id

                # 3. Create the batch job
                batch = self.client.batches.create_batch(file_id, endpoint="/v1/chat/completions")
                batch_id = batch.id
                batch_id_path.write_text(batch_id)

            start_time = time.time()
            while True:
                b = self.client.batches.get_batch(batch_id)
                status = b.status
                print(f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S')}] Batch {batch_id} status: {status}")

                if status == 'VALIDATING':
                    time.sleep(5)
                    continue
                if status == "COMPLETED":
                    if b.error_file_id is not None:
                        print(f"Warning: Batch {batch_id} completed with errors. error file ID {b.error_file_id}")
                        self.client.files.retrieve_content(id=b.error_file_id, output=str(error_path))
                    break
                if status in ("FAILED", "EXPIRED", "CANCELLED"):
                    with error_path.open("w", encoding="utf-8") as f:
                        f.write(b.model_dump_json())
                    raise RuntimeError(f"Batch {batch_id} failed with error {b.error}")

                if (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

                if self.params.no_waiting:
                    return False
                time.sleep(poll_interval)

            # 4. Download the result file
            output_file_id = b.output_file_id
            self.client.files.retrieve_content(id=output_file_id, output=str(output_path))

        total_token_consumed = 0
        # 5. Parse the results
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                response = json.loads(line)
                q: DirectQuery = batch_queries[response['custom_id']]
                if response['response']['body']['choices'][0]['finish_reason'] == 'length':
                    q_id = getattr(q, 'id', response['id'])
                    print(f"Warning: Response for query {q_id} was cut off due to length.")
                q.response = response['response']['body']['choices'][0]['message']['content']
                tokens_consumed = response['response']['body']['usage']['total_tokens']
                q.tokens = tokens_consumed
                total_token_consumed += tokens_consumed

        print(f"Total tokens consumed in batch: {total_token_consumed}")
        batch_token_usage_path.write_text(f"{total_token_consumed}")
        #TODO: handle errors per query (non capita mai che non venga data risposta, al massimo fallisce)
        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")
        return True

    def _submit_direct_queries(self, queries: list[DirectQuery]) -> None:
        if any(q.response_json_schema is not None for q in queries) and not self.supports_structured_output:
            raise SubmissionError(f"Model {self.name} does not support structured output required by some queries.")

        if not self.supports_batched and self.params.batched:
            print(f"Model {self.name} does not support batched submissions. Submitting direct.")
            self.params.batched = False

        if self.params.batched:
            batches = self._split_batches(queries)
            all_completed = True
            pbar = tqdm(batches, desc="Uploading batches", unit="batch")
            for i, b in enumerate(pbar):
                pbar.set_postfix(queries=len(b))
                completed = self._submit_direct_queries_batched(self.run_folder / f"batch_{i + 1}", b)
                all_completed = all_completed and completed
            if not all_completed:
                print("Not waiting for submission to complete...")
            return
        else:
            for q in queries:
                self._submit_direct_query_inline(q)
