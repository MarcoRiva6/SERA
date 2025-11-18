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

data_folder = Path('data')

class Metric(StrEnum):
    pass

Evaluations: TypeAlias = dict[Metric, Number]

@dataclass
class Query(ABC):
    prompt: Any
    response: str
    evaluations: Evaluations

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
    def prepare_direct(self):
        pass

    @abstractmethod
    def prepare_lotus(self):
        pass

    def prepare(self):
        method_path = 'prepare_' + self.run_type.value
        method = getattr(self, method_path)
        if method is not None:
            method()
        else:
            raise NotImplementedError(f'Run mode {self.run_type} not implemented.')

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

    def evaluations_to_csv(self, file_path: str = None):
        if file_path is None:
            file_path = self.run_folder / 'evaluations.csv'

        df = pd.DataFrame([self.evaluations], index=[0])
        df.to_csv(file_path, index=False)

    def queries_to_csv(self, file_path: str = None):
        if file_path is None:
            file_path = self.run_folder / 'queries.csv'
        rows = []

        for q in self.queries:
            base = asdict(q)                # convert dataclass to dictionary
            eval_dict = base.pop("evaluations")  # remove evaluation dict
            flat = {**base, **eval_dict}      # flatten into top-level
            rows.append(flat)

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)