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
    batched: bool = False
    run_type: RunType = None
    submissions: list[Submission] = None
    run_folder: Path = None

    @classmethod
    def from_yaml_file(cls, file):
        with open(file) as f:
            data = cls(**yaml.load(f, Loader=yaml.FullLoader), name_path=file.stem)
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

    def _submit_lotus_lm_studio(self, prompt, df):
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

    def _submit_together(self, prompt: str) -> str:
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

    def _submit_direct_together_batched(self, queries: list[Query]):
        poll_interval = 60 #seconds
        timeout = 86400  #seconds (24 hours)
        max_tokens = 300
        input_path = self.run_folder / "batch_input.jsonl"
        output_path = self.run_folder / "batch_output.jsonl"

        batch_queries: dict[str, Query] = {hashlib.md5(q.prompt.encode()).hexdigest(): q for q in queries}

        if os.path.exists(output_path):
            print(f"Using existing batch output file at {output_path}")
        else:
            if os.path.exists(input_path): # retrieve existing input file
                print(f"Using existing batch input file at {input_path}")
            else:
                requests = [{
                        "custom_id": k,
                        "body": {
                            "model": self.name_api,
                            "messages": [{"role": "user", "content": v.prompt}]
                        },
                        "max_tokens": max_tokens
                    } for k, v in batch_queries.items()]

                # 1. Write requests to a .jsonl file
                with input_path.open("w", encoding="utf-8") as f:
                    for req in requests:
                        f.write(json.dumps(req) + "\n")

            # Load API key from .env file
            if not load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env"):
                raise FileNotFoundError()
            api_key = os.getenv("TOGETHER_API_KEY")
            from together import Together
            client = Together(api_key=api_key)
            # 2. Upload the batch file
            file_resp = client.files.upload(file=input_path, purpose="batch-api")
            file_id = file_resp.id

            # 3. Create the batch job
            batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")
            batch_id = batch.id

            start_time = time.time()
            while True:
                status = client.batches.get_batch(batch_id)
                print(f"Batch {batch_id} status: {status.status}")

                if status.status == "COMPLETED":
                    if status.error_file_id is not None:
                        print(f"Warning: Batch {batch_id} completed with errors. error file ID {status.error_file_id}")
                        error_path = self.run_folder / "batch_error.json"
                        client.files.retrieve_content(id=status.error_file_id, output=str(error_path))
                    break
                if status.status in ("FAILED", "EXPIRED", "CANCELLED"):
                    error_path = self.run_folder / "batch_error.json"
                    with error_path.open("w", encoding="utf-8") as f:
                        f.write(status.model_dump_json())
                    raise RuntimeError(f"Batch {batch_id} failed with error {status.error}")

                if (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

                time.sleep(poll_interval)

            # 4. Download the result file
            output_file_id = status.output_file_id
            client.files.retrieve_content(id=output_file_id, output=str(output_path))

        # 5. Parse the results
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                response = json.loads(line)
                if response['response']['body']['choices'][0]['finish_reason'] == 'length':
                    print(f"Warning: Response for query with prompt hash {response['id']} was cut off due to length.")
                batch_queries[response['custom_id']].response = response['response']['body']['choices'][0]['message']['content']

        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")

    def submit(self, test: Test, df=None):
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
                    submit_method(submission.queries)
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
