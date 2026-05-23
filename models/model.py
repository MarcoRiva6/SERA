import ast
import hashlib
import json
import os
from abc import ABC
from time import sleep
from typing import Any

import math

import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv
from lotus.models import LM

from pandas import DataFrame
from tqdm import tqdm

from experiments.run_type import RunType
from queries.test import Query, PartitionedQuery, LotusQuery, DirectQuery, Test, QueryParameters


@dataclass
class Submission:
    queries: list[Query]
    success: bool = False

class SubmissionError(RuntimeError):
    pass

var_file: Path = Path(__file__).resolve().parent.parent / ".env"


def write_jsonl(write_to: Path, lines: list[Any]):
    with write_to.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

def _split_queries(counting_method, tokens_per_submission: int, queries: list[Query]) -> list[list[Query]]:
    total_batch_tokens = sum([counting_method(q.prompt) for q in queries])
    if total_batch_tokens > tokens_per_submission:
        margin = 0.1
        n_batches = math.ceil(total_batch_tokens / ((1-margin) * tokens_per_submission))
        queries_per_batch = math.ceil(len(queries) / n_batches)
        return [queries[i*queries_per_batch : (i+1)*queries_per_batch]
                            for i in range(n_batches)]
    else:
        return [queries]

@dataclass
class ModelParams:
    seed: int = 0
    temperature: float = -1

@dataclass
class BatchableModelParams(ModelParams):
    batched: bool = True

@dataclass
class Model[T_ModelParams: ModelParams](ABC):
    name: str
    name_path: str
    name_api: str
    family: str
    max_tokens: int
    run_type: RunType
    run_folder: Path
    params: T_ModelParams
    responses_file_name: str = 'responses.jsonl'
    params_file_name: str = 'model_params.json'

    def save_params(self) -> None:
        """
        Save the test configuration to a JSON file.
        """
        self.run_folder.mkdir(parents=True, exist_ok=True)
        with open(self.run_folder / self.params_file_name, 'w') as f:
            param_dict = asdict(self.params)
            param_dict.pop('no_waiting', None)
            json.dump(param_dict, f, indent=4)

    def same_params(self) -> bool:
        """
        Compare the current test configuration with a previously saved one.
        :return: True if the configurations match, False otherwise.
        """
        with open(self.run_folder / self.params_file_name, 'r') as f:
            saved_params = json.load(f)
        current_params = asdict(self.params)
        current_params.pop('no_waiting', None)
        return saved_params == current_params

    def _load_env_file(self) -> None:
        """
        Load environment variables from a .env file.
        """
        if not load_dotenv(var_file):
            raise FileNotFoundError()

    def _init_model(self) -> None:
        """
        Initialize the model, e.g., setting up API clients or loading local models. Called as first thing.
        """
        pass

    def _finish_model(self) -> None:
        """
        Finalize the model, e.g., closing connections or cleaning up resources. Called as last thing.
        """
        pass

    def _store_query_inline(self, qid: int, response: str | None, tokens) -> None:
        response_file_path: Path = self.run_folder / self.responses_file_name
        record = {
            "id": qid,
            "response": response,
            "tokens": tokens
        }
        with open(response_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _retrieve_queries_inline(self) -> dict[int, dict[str, str | None | int]]:
        response_file_path = self.run_folder / self.responses_file_name

        processed_queries = {}
        if response_file_path.exists():
            with open(response_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_queries[data['id']] = {"response": data['response'], "tokens": data['tokens']}
            if len(processed_queries) > 0:
                print(f"Retrieved {len(processed_queries)} already processed answer{"s" if len(processed_queries) > 1 else ""}.")

        return processed_queries

    def _submit_direct_query_inline(self, query: DirectQuery) -> None:
        """
        Submit a single direct query to the model using direct submission method.
        The method should populate the 'response' attribute of the Query object.
        """
        raise NotImplementedError(f"Direct submission not implemented for model {self.name}.")

    def _submit_direct_queries_inline(self, queries: list[DirectQuery]):
        processed_queries = self._retrieve_queries_inline()

        queries_to_process = []
        for query in queries:
            if query.id in processed_queries:
                query.response = processed_queries[query.id]["response"]
                query.tokens = processed_queries[query.id]["tokens"]
            else:
                queries_to_process.append(query)

        if not queries_to_process:
            return

        for q in tqdm(queries_to_process, desc=f"Quering {self.name}", unit="query"):
            self._submit_direct_query_inline(q)
            self._store_query_inline(q.id, q.response, q.tokens)

    def _submit_direct_queries(self, queries: list[DirectQuery]) -> None:
        self._submit_direct_queries_inline(queries)

    def _get_lotus_params(self) -> tuple[str,str|None]:
        raise NotImplementedError(f"Lotus submission not implemented for model {self.name}.")

    def _prepare_lotus(self):
        import lotus
        from lotus.cache import CacheConfig, CacheType, CacheFactory
        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
        cache = CacheFactory.create_cache(cache_config)
        model_string, api_string = self._get_lotus_params()
        temperature = self.params.temperature if self.params.temperature != -1 else 0.0 # lotus' default
        lm = LM(model=model_string, temperature=temperature, api_base=api_string, cache=cache)#, rate_limit=3)
        lotus.settings.configure(lm=lm)

    def _submit_lotus_queries(self, queries: list[LotusQuery]) -> None:
        """
        Submit queries to the model using Lotus submission method.
        The method should populate the 'response' attribute of each Query object in the queries list.
        """
        self._prepare_lotus()

        for i, query in enumerate(tqdm(queries, desc="Submitting queries with Lotus",unit="query")):
            file_path = os.path.join(self.run_folder, f"query_{i}.parquet")

            # la query è già stata eseguita
            if os.path.exists(file_path):
                query.response = pd.read_parquet(file_path)
                continue

            try:
                assert isinstance(query.prompt_df, DataFrame)
                #TODO: fix stats non salvate??
                result, stats = query.prompt_df.sem_topk(
                    query.prompt,
                    K=query.parameters.k,
                    method="quick", #quick #heap, naive,
                    return_stats=True
                )
                result.to_parquet(file_path, index=False)
                query.response = result
                query.tokens = stats["total_tokens"]
            except Exception as e:
                print(f"Errore durante la query {i}: {e}")
                break

    def _submit_partitioned_query(self, query: PartitionedQuery, test: Test) -> None:
        def write_prompt(prompt_beg, df: DataFrame, prompt_end) -> str:
            return f"{prompt_beg}\n{test.df_to_string_for_prompt(df, query.parameters.prompt_printing_mode)}\n\n{prompt_end}"

        max_attempts = 3
        curr_attempts = 0

        current_top_keys = []
        subq_id = query.id*10000+100000000
        while query.remaining_keys:
            # init round
            subquery_raw_df, updated_remaining_keys = test.select_next_partition(query.raw_df, query.parameters.partition_rate, query.remaining_keys, current_top_keys)
            query.remaining_keys = updated_remaining_keys
            gt, gt_scores = test.build_ground_truth(subquery_raw_df, None)
            subquery_prompt_df = test.build_prompt_df(subquery_raw_df)
            subquery = DirectQuery(
                id=subq_id,
                ds_id=query.ds_id,
                ground_truth=gt,
                ground_truth_scores=gt_scores,
                prompt_df=subquery_prompt_df,
                parameters=QueryParameters(k=query.parameters.k,prompt_level=query.parameters.prompt_level,names_level=query.parameters.names_level,
                                           n_elems=len(subquery_raw_df), prompt_printing_mode=query.parameters.prompt_printing_mode, completeness_level=query.parameters.completeness_level),
                prompt=write_prompt(query.prompt_beg, subquery_prompt_df, query.prompt_end),
                response_json_schema=query.response_json_schema,
                response=None,
                parsed_response=None,
                evaluations=None,
                parsing_failed=None,
                tokens=None
            )
            query.sub_queries.append(subquery)
            # execute
            self._submit_direct_query_inline(subquery)
            # process response
            test.parse_query(subquery)
            if subquery.parsed_response:
                # TODO: !!! Fare così è un problema perché in "salva" LLM in caso di allucinazione
                parsed_response = [entry for entry in subquery.parsed_response if entry in gt]
                if len(parsed_response) >= query.parameters.k:
                    current_top_keys = parsed_response[:query.parameters.k]
                    if subquery.tokens:
                        if not query.tokens:
                            query.tokens = 0
                        query.tokens += subquery.tokens
                    curr_attempts = 0
                else:
                    curr_attempts += 1
            else:
                curr_attempts += 1

            if curr_attempts == max_attempts:
                query.parsing_failed = True
                return
            subq_id += 1

        query.parsing_failed = False
        query.parsed_response = current_top_keys

    def _submit_partitioned_queries(self, queries: list[PartitionedQuery], test: Test) -> None:
        processed_queries = self._retrieve_queries_inline()

        queries_to_process = []
        for query in queries:
            if query.id in processed_queries:
                q_stored_response = processed_queries[query.id]["response"]
                if q_stored_response is None:
                    query.parsing_failed = True
                else:
                    query.parsed_response = ast.literal_eval(q_stored_response)
                    query.parsing_failed = False
                query.tokens = processed_queries[query.id]["tokens"]
            else:
                queries_to_process.append(query)

        if not queries_to_process:
            return

        for query in tqdm(queries_to_process, desc=f"Querying {self.name} via Ollama", unit="query", colour='yellow'):
            self._submit_partitioned_query(query, test)
            self._store_query_inline(query.id, str(query.parsed_response) if query.parsed_response is not None else None, query.tokens)

    def _query_fits_limit(self, query: Query) -> bool:
        """
        Check if the prompt of a single query fits within the model's token limit.
        """
        raise NotImplementedError(f"Token counting method not implemented for model {self.name}.")

    def queries_fits_limit(self, queries: list[Query]) -> bool:
        for q in queries:
            if not self._query_fits_limit(q):
                return False
        return True

    @classmethod
    def from_yaml_file(cls, **kwargs) -> 'Model':
        with open(kwargs.get('file')) as f:
            model_args = yaml.load(f, Loader=yaml.FullLoader) | {k: v for k, v in kwargs.items() if k != 'file'}
            return cls(**model_args)

    def run(self, queries: list[Query], test: Test):
        self._init_model()
        print(f"Model processing {len(queries)} queries...")
        try:
            match self.run_type:
                case RunType.DIRECT:
                    self._submit_direct_queries(queries)
                case RunType.LOTUS:
                    self._submit_lotus_queries(queries)
                case RunType.PARTITIONED:
                    self._submit_partitioned_queries(queries, test)
        except Exception as e:
            raise SubmissionError(f'Error during submission: {e}', 'error type:', type(e))
        finally:
            self._finish_model()

@dataclass
class BatchableModel[QIID: (str,int), T_BatchID: Any, T_ModelParam: BatchableModelParams](Model[T_ModelParam]):
    supports_batched: bool = True
    @dataclass
    class QueryBatchMapping:
        q_id: int
        batch_id: int
        qiid: int|str

    def _query_to_qiid(self, q: DirectQuery) -> QIID:
        return hashlib.md5(q.prompt.encode()).hexdigest()

    def _batch_input_to_qiids(self, batch_input_path: Path)  -> list[QIID]:
        raise NotImplementedError(f"Batch input parsing method not implemented for model {self.name}.")

    def _build_batch_input(self, mapping: dict[QIID, DirectQuery]) -> list:
        raise NotImplementedError(f"Batch input building not implemented for model {self.name}.")

    def _send_batch(self, input_file_path: Path) -> T_BatchID:
        raise NotImplementedError(f"Batch submission method not implemented for model {self.name}.")

    def _check_batch(self, batch_id: T_BatchID, error_path: Path) -> bool:
        raise NotImplementedError(f"Batch status checking method not implemented for model {self.name}.")

    def _download_batch(self, batch_id: T_BatchID, output_path: Path) -> None:
        raise NotImplementedError(f"Batch result downloading method not implemented for model {self.name}.")

    def _process_batch_output(self, batched_queries: dict[QIID, DirectQuery], output_path: Path, batch_token_usage_path: Path):
        raise NotImplementedError(f"Batch output processing method not implemented for model {self.name}.")

    def _submit_direct_queries_batch(self, batch_number: int, queries: list[DirectQuery]) -> None:
        batch_folder = self.run_folder / f"batch_{batch_number}"
        batch_id_path = batch_folder / "batch_id.txt"
        input_path = batch_folder / "batch_input.jsonl"
        output_path = batch_folder / "batch_output.jsonl"
        error_path = batch_folder / "batch_error.jsonl"
        batch_token_usage_path = batch_folder / "batch_token_usage.txt"

        batch_queries = {self._query_to_qiid(q): q for q in queries}

        if input_path.exists():
            input_file_qiids = self._batch_input_to_qiids(input_path)
            if set(input_file_qiids) != set(batch_queries.keys()):
                raise RuntimeError(f"Batch {batch_number} input file qiids don't match batch queries qiids. Interrupting.")

        if output_path.exists():
            print(f"Using existing batch output file...")
        else:
            if batch_id_path.exists(): # recover existing batch
                with open(batch_id_path, "r") as f:
                    batch_id: T_BatchID = f.read().strip()
                print(f"Trying to recover already submitted batch {batch_id}...")
            else:
                requests = self._build_batch_input(batch_queries)
                batch_folder.mkdir(parents=True, exist_ok=True)
                write_jsonl(input_path, requests)

                batch_id: T_BatchID = self._send_batch(input_path)
                batch_id_path.write_text(batch_id)
                sleep(2)

            downloadable = self._check_batch(batch_id, error_path)
            if downloadable:
                print(f"downloading batch {batch_number}...")
                self._download_batch(batch_id, output_path)
            else:
                print(f"batch {batch_number} hasn't finished yet")
                return

        self._process_batch_output(batch_queries, output_path, batch_token_usage_path)

    def _split_batches(self, queries: list[DirectQuery]) -> list[list[DirectQuery]]:
        raise NotImplementedError(f"Batch splitting method not implemented for model {self.name}.")

    def _map_queries_to_batches(self, queries: list[DirectQuery]) -> dict[int, list[DirectQuery]]:
        batches_queries = {self._query_to_qiid(q): q for q in queries}
        batches_map: dict[int, list[DirectQuery]] = {}
        for elem in self.run_folder.iterdir():
            if elem.is_dir() and elem.name.startswith("batch_"):
                maybe_batch_number = elem.name.replace("batch_", "")
                if maybe_batch_number.isdigit():
                    batch_number = int(maybe_batch_number)
                    batch_folder = self.run_folder / f"batch_{batch_number}"
                    batch_input_path = batch_folder / "batch_input.jsonl"
                    if batch_input_path.exists():
                        batch_quiids = self._batch_input_to_qiids(batch_folder / "batch_input.jsonl")
                        batches_map[batch_number] = [batches_queries.pop(qiid) for qiid in batch_quiids]
        batches_map[-1] = list(batches_queries.values())

        return batches_map

    def _submit_direct_queries(self, queries: list[DirectQuery]) -> None:
        if not self.supports_batched and self.params.batched:
            print(f"Model {self.name} does not support batched submissions. Submitting direct.")
            self.params.batched = False

        if self.params.batched:
            if self.family == "google" and "smartphones" in queries[0].prompt:
                self.batch_max_tokens = int(self.batch_max_tokens / 2)
            # parse existing batches
            batches_map = self._map_queries_to_batches(queries)
            # new batches
            to_be_split: list[DirectQuery] = batches_map.pop(-1)
            if to_be_split:
                new_batches_lists = self._split_batches(to_be_split)
                old_batches_ids = batches_map.keys()
                new_batches_start_id = max(old_batches_ids) + 1 if old_batches_ids else 1
                for new_batch_id, qs in enumerate(new_batches_lists, start=new_batches_start_id):
                    batches_map[new_batch_id] = qs
                print(f"New {len(to_be_split)} queries to submit in {len(new_batches_lists)} batches.")
            # process batches
            batches_count = len(batches_map.keys())
            print(f"Processing {batches_count} batches...")
            for progress_counter, (batch_number, batch_queries) in enumerate(batches_map.items(), start=1):
                print(f"Processing batch {batch_number} ({progress_counter}/{batches_count})...")
                self._submit_direct_queries_batch(batch_number, batch_queries)
        else:
            self._submit_direct_queries_inline(queries)
