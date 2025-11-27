import hashlib
import json
import os
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from lmstudio import LMStudioError
from enum import StrEnum, auto
from experiments.run_type import RunType
from queries.test import Query, Test

class Backend(StrEnum):
    LM_STUDIO = auto()
    TOGETHER = auto()

@dataclass
class Submission:
    queries: list[Query]
    success: bool = False

class SubmissionError(RuntimeError):
    pass

@dataclass
class Model:
    name: str
    name_path: str
    name_api: str
    backend: Backend
    max_tokens: int
    run_type: RunType
    run_folder: Path
    seed: int
    batched: bool = False
    submissions: list[Submission] = None
    no_waiting: bool = False

    @classmethod
    def from_yaml_file(cls, **kwargs) -> 'Model':
        with open(kwargs.get('file')) as f:
            model_args = yaml.load(f, Loader=yaml.FullLoader) | {k: v for k, v in kwargs.items() if k != 'file'}
            data = cls(**model_args)
        if data.backend:
            data.backend = Backend(data.backend)
        return data

    def _submit_direct_lm_studio(self, prompt: str) -> str:
        import lmstudio as lms
        server_api_host = "localhost:1234"
        retries = 3

        while retries:
            try:
                with lms.Client(server_api_host) as client:
                    model = client.llm.model(self.name_api)

                    response = model.respond(
                        prompt,
                        on_prompt_processing_progress = (lambda progress: print(f"{progress*100}% complete")))

                return response.content

            except LMStudioError as e:
                print("LM Studio exception submission:", e)
                retries -= 1
                import time
                time.sleep(5)

        raise Exception("Failed to get response from LM Studio after retries.")

    def _submit_lotus_lm_studio(self, prompt, df) -> None:
        import lotus
        from lotus.cache import CacheFactory, CacheConfig, CacheType
        from lotus.models import LM

        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
        cache = CacheFactory.create_cache(cache_config)

        lm = LM(model="lm_studio/" + self.name_api,
                api_base="http://127.0.0.1:1234/v1",
                max_tokens=self.max_tokens if self.max_tokens else None,
                cache=cache,
                temperature=0.6)

        lotus.settings.configure(lm=lm, enable_cache=True)

        return prompt(df)

    def _submit_direct_together(self, prompt: str) -> str:
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

    def _submit_direct_together_batched(self, queries: list[Query]) -> bool:
        poll_interval = 60 #seconds
        timeout = 86400  #seconds (24 hours)
        batch_id_path = self.run_folder / "batch_id.txt"
        input_path = self.run_folder / "batch_input.jsonl"
        output_path = self.run_folder / "batch_output.jsonl"
        error_path = self.run_folder / "batch_error.json"
        batch_token_usage_path =self.run_folder / "batch_token_usage.txt"

        batch_queries: dict[str, Query] = {hashlib.md5(q.prompt.encode()).hexdigest(): q for q in queries}

        if os.path.exists(output_path):
            print(f"Using existing batch output file at {output_path}")
        else:
            # Load API key from .env file
            if not load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env"):
                raise FileNotFoundError()
            api_key = os.getenv("TOGETHER_API_KEY")
            from together import Together
            client = Together(api_key=api_key)

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
                    if q.response_json_schema is not None and q.response_json_schema != '':
                        r['body']['response_format'] = q.response_json_schema
                    requests.append(r)

                # 1. Write requests to a .jsonl file
                self.run_folder.mkdir(parents=True, exist_ok=True)
                with input_path.open("w", encoding="utf-8") as f:
                    for req in requests:
                        f.write(json.dumps(req) + "\n")

                # 2. Upload the batch file
                file_resp = client.files.upload(file=input_path, purpose="batch-api")
                file_id = file_resp.id

                # 3. Create the batch job
                batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")
                batch_id = batch.id
                batch_id_path.write_text(batch_id)

            start_time = time.time()
            while True:
                status = client.batches.get_batch(batch_id)
                print(f"Batch {batch_id} status: {status.status}")

                if status.status == 'VALIDATING':
                    time.sleep(5)
                    continue
                if status.status == "COMPLETED":
                    if status.error_file_id is not None:
                        print(f"Warning: Batch {batch_id} completed with errors. error file ID {status.error_file_id}")
                        client.files.retrieve_content(id=status.error_file_id, output=str(error_path))
                    break
                if status.status in ("FAILED", "EXPIRED", "CANCELLED"):
                    with error_path.open("w", encoding="utf-8") as f:
                        f.write(status.model_dump_json())
                    raise RuntimeError(f"Batch {batch_id} failed with error {status.error}")

                if (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

                if self.no_waiting:
                    return False
                time.sleep(poll_interval)

            # 4. Download the result file
            output_file_id = status.output_file_id
            client.files.retrieve_content(id=output_file_id, output=str(output_path))

        total_token_consumed = 0
        # 5. Parse the results
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                response = json.loads(line)
                if response['response']['body']['choices'][0]['finish_reason'] == 'length':
                    print(f"Warning: Response for query with prompt hash {response['id']} was cut off due to length.")
                batch_queries[response['custom_id']].response = response['response']['body']['choices'][0]['message']['content']
                total_token_consumed += response['response']['body']['usage']['total_tokens']

        print(f"Total tokens consumed in batch: {total_token_consumed}")
        batch_token_usage_path.write_text(f"{total_token_consumed}")

        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")
        return True

    def submit(self, test: Test, df=None) -> bool:
        #build submissions
        if self.batched:
            self.submissions = [Submission(queries=test.queries)]
        else:
            self.submissions = [Submission(queries=[q]) for q in test.queries]

        submit_name = '_submit_' + self.run_type.value + '_' + self.backend.value
        if self.batched:
            submit_name += '_batched'
        try:
            submit_method = getattr(self, submit_name)
        except AttributeError:
            raise NotImplementedError(f'Submit {submit_name} not implemented.')

        for i, submission in enumerate(self.submissions):
            if self.batched:
                print(f'Submitting {len(submission.queries)} queries...')
                try:
                    if not submit_method(submission.queries):
                        return False
                    submission.success = True
                except Exception as e:
                    raise SubmissionError(f'Error during batched submission: {e}')
            else:
                print(f'Submitting query {i + 1}/{len(self.submissions)}...')
                try:
                    if self.run_type == RunType.LOTUS:
                        response = submit_method(submission.queries[0].prompt, df)
                    else:
                        response = submit_method(submission.queries[0].prompt)
                    submission.queries[0].response = response
                    submission.success = True
                    print("Received response:", submission.queries[0].response)
                except Exception as e:
                    raise SubmissionError(f'Error during submission of query {i + 1}: {e}')
            if submission.success:
                for q in submission.queries:
                    try:
                        q.evaluations = test.evaluate_query(q)
                    except Exception as e:
                        raise SubmissionError(f'Error during evaluation of query: {e}')
        return True
