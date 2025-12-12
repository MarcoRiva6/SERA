import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import together

from models.model import Model, write_jsonl
from queries.test import Query

@dataclass
class TogetherModel(Model):
    client: together.Client
    supports_batched: bool = True
    @dataclass
    class Params(Model.Params):
        batched: bool = True
    params: Params = field(default_factory=Params)

    def _init_model(self) -> None:
        # Load API key from .env file
        self._load_env_file()
        api_key = os.getenv("TOGETHER_API_KEY")
        from together import Together
        self.client = Together(api_key=api_key)

    def _submit_direct_inline(self, prompt: str) -> str:
        from together import Together

        client = Together()

        return client.chat.completions.create(
            model=self.name_api,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=False,
        ).response().choices[0].message.content

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
                requests = []
                for q_key, q in batch_queries.items():
                    r = {
                        "custom_id": q_key,
                        "body": {
                            "model": self.name_api,
                            "messages": [{"role": "user", "content": q.prompt}],
                            "reasoning": {"enable": False} # supportato solo da DS V3.1
                        },
                        "max_tokens": self.max_tokens
                    }
                    # notare che questo serve solamente a verificare che l'output rispetti lo schema, e non forza il modello a rispondere in quel modo
                    if q.response_json_schema is not None and q.response_json_schema != '':
                        r['body']['response_format'] = {"type": "json_schema", "schema": q.response_json_schema}
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
                q: Query = batch_queries[response['custom_id']]
                if response['response']['body']['choices'][0]['finish_reason'] == 'length':
                    q_id = getattr(q, 'id', response['id'])
                    print(f"Warning: Response for query {q_id} was cut off due to length.")
                q.response = response['response']['body']['choices'][0]['message']['content']
                total_token_consumed += response['response']['body']['usage']['total_tokens']

        print(f"Total tokens consumed in batch: {total_token_consumed}")
        batch_token_usage_path.write_text(f"{total_token_consumed}")

        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")
        return True

    def _submit_direct(self, queries: list[Query]) -> None:
        if not self.supports_batched and self.params.batched:
            print(f"Model {self.name} does not support batched submissions. Submitting direct.")
            self.params.batched = False
        if self.params.batched:
            if not self._submit_direct_batched(self.run_folder, queries):
                print("Not waiting for submission to complete...")
        else:
            for q in queries:
                q.response = self._submit_direct_inline(q.prompt)
