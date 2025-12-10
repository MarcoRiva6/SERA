import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any

import math
import yaml
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from google.genai.types import BatchJob
from lmstudio import LMStudioError
from enum import StrEnum, auto
from experiments.run_type import RunType
from queries.test import Query, Test

class Backend(StrEnum):
    LM_STUDIO = auto()
    TOGETHER = auto()
    GOOGLE = auto()

@dataclass
class Submission:
    queries: list[Query]
    success: bool = False

class SubmissionError(RuntimeError):
    pass


def write_jsonl(write_to: Path, lines: list[Any]):
    with write_to.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")


@dataclass
class Model:
    name: str
    name_path: str
    name_api: str
    backend: Backend
    max_tokens: int
    batch_max_tokens: int
    run_type: RunType
    run_folder: Path
    seed: int
    batched: bool = False
    submissions: list[Submission] = None
    no_waiting: bool = False
    initialized_client: Any = None

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
        error_path = self.run_folder / "batch_error.jsonl"
        batch_token_usage_path =self.run_folder / "batch_token_usage.txt"

        batch_queries: dict[str, Query] = {hashlib.md5(q.prompt.encode()).hexdigest(): q for q in queries}

        if os.path.exists(output_path):
            print(f"Using existing batch output file...")
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
                    # notare che questo serve solamente a verificare che l'output rispetti lo schema, e non forza il modello a rispondere in quel modo
                    if q.response_json_schema is not None and q.response_json_schema != '':
                        r['body']['response_format'] = {"type": "json_schema", "schema": q.response_json_schema}
                    requests.append(r)

                # 1. Write requests to a .jsonl file
                self.run_folder.mkdir(parents=True, exist_ok=True)
                write_jsonl(input_path, requests)

                # 2. Upload the batch file
                file_resp = client.files.upload(file=input_path, purpose="batch-api")
                file_id = file_resp.id

                # 3. Create the batch job
                batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")
                batch_id = batch.id
                batch_id_path.write_text(batch_id)

            start_time = time.time()
            while True:
                b = client.batches.get_batch(batch_id)
                status = b.status
                print(f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S')}] Batch {batch_id} status: {status}")

                if status == 'VALIDATING':
                    time.sleep(5)
                    continue
                if status == "COMPLETED":
                    if b.error_file_id is not None:
                        print(f"Warning: Batch {batch_id} completed with errors. error file ID {b.error_file_id}")
                        client.files.retrieve_content(id=b.error_file_id, output=str(error_path))
                    break
                if status in ("FAILED", "EXPIRED", "CANCELLED"):
                    with error_path.open("w", encoding="utf-8") as f:
                        f.write(b.model_dump_json())
                    raise RuntimeError(f"Batch {batch_id} failed with error {b.error}")

                if (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

                if self.no_waiting:
                    return False
                time.sleep(poll_interval)

            # 4. Download the result file
            output_file_id = b.output_file_id
            client.files.retrieve_content(id=output_file_id, output=str(output_path))

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

    def _count_tokens_google(self, prompt: str) -> int:
        if self.initialized_client is None:
            self.initialized_client = self.__init_google_client()
        client: Any = self.initialized_client
        response = client.models.count_tokens(contents=prompt, model=self.name_api)
        return response.total_tokens

    @staticmethod
    def __init_google_client():
        # Load API key from .env file
        if not load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env"):
            raise FileNotFoundError()
        api_key = os.getenv("GOOGLE_API_KEY")
        from google.genai import Client
        return Client(api_key=api_key)

    def _submit_direct_google(self, prompt: str) -> str:
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
        if self.initialized_client is None:
            self.initialized_client = self.__init_google_client()
        client: Client = self.initialized_client

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
                    raise SubmissionError(error_msg)


    def _submit_direct_google_batched(self, queries: list[Query]) -> bool:
        poll_interval = 60 #seconds
        timeout = 86400  #seconds (24 hours)
        batch_id_path = self.run_folder / "batch_id.txt"
        input_path = self.run_folder / "batch_input.jsonl"
        output_path = self.run_folder / "batch_output.jsonl"
        error_path = self.run_folder / "batch_error.jsonl"
        batch_token_usage_path =self.run_folder / "batch_token_usage.txt"

        batch_queries: dict[str, Query] = {hashlib.md5(q.prompt.encode()).hexdigest(): q for q in queries}

        if os.path.exists(output_path):
            print(f"Using existing batch output file...")
        else:
            client = self.__init_google_client()

            if os.path.exists(batch_id_path): # recover existing batch
                with open(batch_id_path, "r") as f:
                    batch_id = f.read().strip()
                print(f"Trying to recover already submitted batch {batch_id}...")
            else:
                requests: list[dict] = []

                for q_key, q in batch_queries.items():
                    r = {
                        "key": q_key,
                        "generationConfig": {
                            "maxOutputTokens": self.max_tokens,
                            "thinking_config": {
                                "include_thoughts": False,
                                "thinking_budget": 0
                            }
                            #"temperature": 0.7,
                            #"seed": self.seed
                        },
                        "request": {
                            "contents": [{"parts": [{"text": q.prompt}], "role": "user"}],
                        }
                    }
                    requests.append(r)

                write_jsonl(input_path, requests)

                # Upload the file to the File API
                file_resp = client.files.upload(file=input_path, config={'mime_type': 'jsonl'})
                file_id = file_resp.name

                # Create the batch job
                batch: BatchJob = client.batches.create(model=self.name_api, src=file_id)
                batch_id = batch.name
                batch_id_path.write_text(batch_id)

            start_time = time.time()
            while True:
                b = client.batches.get(name=batch_id)
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

                if self.no_waiting:
                    return False
                time.sleep(poll_interval)

            # download result file
            output_file_id = b.dest.file_name
            file_content = client.files.download(file=output_file_id)
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
                q.response = response['response']['candidates'][0]['content']['parts'][0]['text']
                total_token_consumed += response['response']['usageMetadata']['totalTokenCount']

        print(f"Total tokens consumed in batch: {total_token_consumed}")
        batch_token_usage_path.write_text(f"{total_token_consumed}")

        for bq in batch_queries.values():
            if not bq.response:
                print(f"Warning: No response for query with prompt hash {hash(bq.prompt)}")
        return True

    def _prepare_submissions(self, queries: list[Query]) -> None:
        """
        Populates self.submissions using the provided queries,
        """
        if self.batched:
            self.submissions = [Submission(queries=queries)]
            if self.batch_max_tokens != 0:
                counting_method = getattr(self, '_count_tokens_' + self.backend.value, None)
                if counting_method is not None:
                    total_batch_tokens = sum([counting_method(q.prompt) for q in queries])
                    if total_batch_tokens > self.batch_max_tokens:
                        margin = 0.1
                        n_batches = math.ceil(total_batch_tokens / ((1-margin) * self.batch_max_tokens))
                        queries_per_batch = math.ceil(len(queries) / n_batches)
                        self.submissions = [Submission(queries=queries[i*queries_per_batch : (i+1)*queries_per_batch])
                                            for i in range(n_batches)]
                else:
                    print(f"Warning: 'max_batch_tokens' specified, but counting method for backend {self.backend.value} not implemented. Will not split.")
        else:
            self.submissions = [Submission(queries=[q]) for q in queries]

    def submit(self, test: Test, df=None) -> bool:
        """
        Submit the test queries to the model.
        It automatically selects the appropriate submission method based on the model's run type and backend.
        :param test: the Test object containing the queries to be submitted
        :param df: used only for LOTUS run type, the dataframe to be passed to the prompt function
        :return: True when the submission is complete, False if the batched submission is still in progress
        """
        self._prepare_submissions(test.queries)

        submit_name = '_submit_' + self.run_type.value + '_' + self.backend.value
        if self.batched:
            submit_name += '_batched'
        try:
            submit_method = getattr(self, submit_name)
        except AttributeError:
            raise NotImplementedError(f'Submit {submit_name} not implemented.')

        print(f"Submitting {len(test.queries)} queries...")

        if self.batched:
            for b_number, submission in enumerate(self.submissions):
                print(f'Submitting batch {b_number + 1}/{len(self.submissions)} with {len(submission.queries)} queries...')
                self.run_folder = self.run_folder / f'batch_{b_number + 1}'
                self.run_folder.mkdir(parents=True, exist_ok=True)
                try:
                    if not submit_method(submission.queries):
                        return False
                    submission.success = True
                except Exception as e:
                    raise SubmissionError(f'Error during batched submission: {e}')
                finally:
                    self.run_folder = self.run_folder.parent
        else:
            for s_number, submission in enumerate(self.submissions):
                print(f'Submitting query {s_number + 1}/{len(self.submissions)}...')
                try:
                    if self.run_type == RunType.LOTUS:
                        response = submit_method(submission.queries[0].prompt, df)
                    else:
                        response = submit_method(submission.queries[0].prompt)
                    submission.queries[0].response = response
                    submission.success = True
                    print("Received response:", submission.queries[0].response)
                except Exception as e:
                    raise SubmissionError(f'Error during submission of query {s_number + 1}: {e}')

        for submission in self.submissions:
            if submission.success:
                for q in submission.queries:
                    try:
                        q.evaluations = test.evaluate_query(q)
                    except Exception as e:
                        print(f'Error during evaluation of query: {e}')

        return True
