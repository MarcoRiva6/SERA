import json
import os
from dataclasses import dataclass
from pathlib import Path

import together
from pydantic import BaseModel
from together.types import ChatCompletionResponse

from models.model import BatchableModel, BatchableModelParams
from queries.test import Query, DirectQuery

type QIID = str
type T_BatchID = str

@dataclass
class TogetherModel(BatchableModel[QIID, T_BatchID, BatchableModelParams]):
    client: together.Client = None
    supports_structured_output: bool = False
    reasoning: bool = False

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

    def _submit_prompt(self, prompt: str, schema: BaseModel|None) -> tuple[str|None, int]:
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
            reasoning={"enabled": self.reasoning},
            max_tokens=self.max_tokens,
            temperature=self.params.temperature if self.params.temperature != -1 else None,
            response_format={"type": "json_schema", "json_schema": {"name": schema.__class__.__name__, "schema": schema.model_json_schema()}} if schema is not None else None
        )
        assert isinstance(answer, ChatCompletionResponse), "Received a streaming response, which is was never meant to"
        used_tokens = answer.usage.total_tokens if answer.usage is not None else 0

        if answer.choices is None or len(answer.choices) == 0:
            return None, used_tokens
        message = answer.choices[0].message
        finish_reason = answer.choices[0].finish_reason
        if finish_reason != 'stop':
            print("Warning: Response may have been cut off due to length. finish_reason:", finish_reason)
        if message is None or isinstance(message.content, list):
            return None, used_tokens
        return message.content, used_tokens

    def _submit_direct_query_inline(self, query: DirectQuery) -> None:
        query.response, query.tokens = self._submit_prompt(query.prompt, query.response_json_schema)

    def _batch_input_to_qiids(self, batch_input_path: Path) -> list[QIID]:
        qiids_list: list[QIID] = []
        with batch_input_path.open("r", encoding="utf-8") as f:
            for line in f:
                batch_q = json.loads(line)
                qiids_list.append(batch_q['custom_id'])

        return qiids_list

    def _process_batch_output(self, batch_queries: dict[str, DirectQuery], output_path: Path, batch_token_usage_path: Path):
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
        # TODO: handle errors per query (non capita mai che non venga data risposta, al massimo fallisce)
        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")

    def _download_batch(self, batch_id: T_BatchID, output_path: Path):
        b = self.client.batches.get_batch(batch_id)
        output_file_id = b.output_file_id
        self.client.files.retrieve_content(id=output_file_id, output=str(output_path))

    def _check_batch(self, batch_id: str, error_path) -> bool:
        b = self.client.batches.get_batch(batch_id)
        status = b.status

        match status:
            case "VALIDATING" | "IN_PROGRESS":
                return False
            case "COMPLETED":
                if b.error_file_id is not None:
                    print(f"Warning: Batch {batch_id} completed with errors. error file ID {b.error_file_id}")
                    self.client.files.retrieve_content(id=b.error_file_id, output=str(error_path))
                return True
            case "FAILED" | "EXPIRED" | "CANCELLED":
                with error_path.open("w", encoding="utf-8") as f:
                    f.write(b.model_dump_json())
                raise RuntimeError(f"Batch {batch_id} failed with error {b.error}")
            case _:
                raise RuntimeError(f"Batch {batch_id} has unrecognized status: {status}")


    def _send_batch(self, input_path: Path) -> str:
        # 2. Upload the batch file
        file_resp = self.client.files.upload(file=input_path, purpose="batch-api")
        file_id = file_resp.id

        # 3. Create the batch job
        batch = self.client.batches.create_batch(file_id, endpoint="/v1/chat/completions")
        return batch.id

    def _build_batch_input(self, mapping: dict[str, DirectQuery]) -> list:
        requests = []
        for q_key, q in mapping.items():
            r: dict = {
                "custom_id": q_key,
                "body": {
                    "model": self.name_api,
                    "messages": [{"role": "user", "content": q.prompt}]
                }
            }
            if self.max_tokens is not None:
                r["body"]["max_tokens"] = self.max_tokens
            if q.response_json_schema is not None:
                r['body']['response_format'] = {"type": "json_schema",
                                                "schema": q.response_json_schema.model_json_schema()}
            if self.params.temperature != -1:
                r['body']['temperature'] = self.params.temperature
            if self.reasoning:
                r['body']['reasoning'] = {"enabled": True}
            requests.append(r)
        return requests
