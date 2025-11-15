from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict, field
from enum import Enum, StrEnum, auto
from numbers import Number
from pathlib import Path
from typing import TypedDict, Dict, TypeAlias

import pandas as pd

import yaml
from pandas import DataFrame

data_folder = Path('data')

class Metric(StrEnum):
    pass

Evaluations: TypeAlias = dict[Metric, Number]

@dataclass
class Submission(ABC):
    prompt: str
    response: str
    evaluations: Evaluations

@dataclass
class Query(ABC):
    family: str
    name_path: str
    run_folder: Path = None
    debug: bool = True
    pre_submissions_df: DataFrame = None
    submissions: list[Submission] = None
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

    @abstractmethod
    def evaluate_submission(self, submission: Submission) -> Evaluations:
        pass

    def evaluate(self):
        # accumulator for sums
        totals: Evaluations = {}

        # sum all metrics across submissions
        for sub in self.submissions:
            for metric, value in sub.evaluations.items():
                totals.update({metric: totals.get(metric, 0) + value})

        # compute mean values
        n = len(self.submissions)
        aggregated: Evaluations = {metric: totals[metric] / n for metric in totals}

        self.evaluations = aggregated
        return aggregated

    def evaluations_to_csv(self, file_path: str = None):
        if file_path is None:
            file_path = self.run_folder / 'evaluations.csv'

        df = pd.DataFrame([self.evaluations], index=[0])
        df.to_csv(file_path, index=False)

    def submissions_to_csv(self, file_path: str = None):
        if file_path is None:
            file_path = self.run_folder / 'submissions.csv'
        rows = []

        for sub in self.submissions:
            base = asdict(sub)                # convert dataclass to dictionary
            eval_dict = base.pop("evaluations")  # remove evaluation dict
            flat = {**base, **eval_dict}      # flatten into top-level
            rows.append(flat)

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)