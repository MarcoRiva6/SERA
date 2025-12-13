import json
from abc import ABC
from typing import Any

import math
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

from pandas import DataFrame

from experiments.run_type import RunType
from queries.test import Query

@dataclass
class Submission:
    queries: list[Query]
    success: bool = False

class SubmissionError(RuntimeError):
    pass

var_file: Path = Path(__file__).parent.parent / ".env"


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
class Model(ABC):
    name: str
    name_path: str
    name_api: str
    family: str
    max_tokens: int
    run_type: RunType
    run_folder: Path
    @dataclass
    class Params(ABC):
        seed: int = 0
        no_waiting: bool = False
    params: Params

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

    def _submit_direct(self, queries: list[Query]) -> None:
        """
        Submit queries to the model using direct submission method.
        The method should populate the 'response' attribute of each Query object in the queries list.
        """
        raise NotImplementedError(f"Direct submission not implemented for model {self.name}.")

    def _submit_lotus(self, queries: list[Query], df: DataFrame) -> None:
        """
        Submit queries to the model using Lotus submission method.
        The method should populate the 'response' attribute of each Query object in the queries list.
        """
        raise NotImplementedError(f"Lotus submission not implemented for model {self.name}.")

    @classmethod
    def from_yaml_file(cls, **kwargs) -> 'Model':
        with open(kwargs.get('file')) as f:
            model_args = yaml.load(f, Loader=yaml.FullLoader) | {k: v for k, v in kwargs.items() if k != 'file'}
            return cls(**model_args)

    def run(self, queries: list[Query], df=None):
        self._init_model()
        print(f"Submitting {len(queries)} queries...")
        try:
            if self.run_type == RunType.DIRECT:
                self._submit_direct(queries)
            elif self.run_type == RunType.LOTUS:
                self._submit_lotus(queries, df)
        except Exception as e:
            raise SubmissionError(f'Error during submission: {e}')
        finally:
            self._finish_model()
