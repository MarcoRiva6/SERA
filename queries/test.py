from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict, field
from enum import Enum, StrEnum, auto
from numbers import Number
from pathlib import Path
from typing import TypedDict, Dict, TypeAlias, Any

import pandas as pd

import yaml
from pandas import DataFrame

from experiments.run_type import RunType

data_folder = Path(__file__).parent.parent / 'data'

class Metric(StrEnum):
    pass

Evaluations: TypeAlias = dict[Metric, Number]

@dataclass
class Query(ABC):
    prompt: Any
    response: str
    evaluations: Evaluations

    def to_dict(self) -> dict:
        if isinstance(self.prompt, DataFrame):
            prompt_str = self.prompt['prompt']
        else:
            prompt_str = str(self.prompt)
        base = {"prompt": prompt_str, "response": self.response}
        evals = {
            (k.value if hasattr(k, "value") else str(k)): v
            for k, v in (self.evaluations or {}).items()
        }
        return {**base, **evals}

@dataclass
class Test(ABC):
    family: str
    name_path: str
    run_type: RunType = None
    run_folder: Path = None
    debug: bool = True
    queries: list[Query] = None
    evaluations: Evaluations = None

    @classmethod
    def from_yaml_file(cls, file):
        with open(file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader), name_path=file.stem)

    @abstractmethod
    def prepare_direct(self) -> None:
        pass

    @abstractmethod
    def prepare_lotus(self) -> None:
        pass

    def prepare(self) -> None:
        """
        After this method is called, queries should be populated.
        """
        method_path = 'prepare_' + self.run_type.value
        method = getattr(self, method_path)
        if method is not None:
            method()
        else:
            raise NotImplementedError(f'Run mode {self.run_type} not implemented for this test.')

    @abstractmethod
    def evaluate_query(self, submission: Query) -> Evaluations:
        pass

    def evaluate(self):
        # accumulator for sums
        totals: Evaluations = {}

        # sum all metrics across queries
        for q in self.queries:
            for metric, value in q.evaluations.items():
                totals.update({metric: totals.get(metric, 0) + value})

        # compute mean values
        n = len(self.queries)
        aggregated: Evaluations = {metric: totals[metric] / n for metric in totals}

        self.evaluations = aggregated
        return aggregated

    def evaluations_to_csv(self, file_name: str = 'test_evaluations.csv'):
        df = pd.DataFrame([self.evaluations], index=[0])
        df.to_csv(self.run_folder / file_name, index=False)

    def queries_to_csv(self, file_name: str = 'queries.csv'):
        df = pd.DataFrame([q.to_dict() for q in self.queries])
        df.to_csv(self.run_folder / file_name, index=False)