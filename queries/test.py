from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import StrEnum
from numbers import Number
from pathlib import Path
from typing import TypeAlias, Any
import pandas as pd
import yaml
from pandas import DataFrame

from experiments.run_type import RunType

data_folder: Path = Path(__file__).parent.parent / 'data' # path to the project's data folder

class Metric(StrEnum):
    """
    Enumeration of possible metrics for a test.
    """
    pass

Evaluations: TypeAlias = dict[Metric, Number] # A dictionary mapping metrics to their numeric evaluation values.

@dataclass
class Query(ABC):
    """
    A single query consisting of a prompt, a response, and its evaluations.
    Prompt can be of any type, including a DataFrame (useful for lotus).
    """
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
    """
    Represents a single test, used to build queries.
    """
    family: str # the family this test belongs to (e.g. "movies", "molecules")
    name_path: str # the path-safe name of the test
    run_type: RunType = None
    run_folder: Path = None
    debug: bool = True
    queries: list[Query] = None # The list of queries generated for this test.
    evaluations: Evaluations = None # Aggregated evaluations across all queries for this test.

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
    def evaluate_query(self, query: Query) -> Evaluations:
        """
        Evaluate a single query after submission.
        :param query: the query to evaluate
        :return: A dictionary (an Evaluations object: dict[Metric, Number]) with the evaluation metrics for the query.
        """
        pass

    def evaluate(self) -> Evaluations:
        """
        Aggregate evaluations across all queries for this test by computing the mean for each metric.
        :return: A dictionary (an Evaluations object: dict[Metric, Number]) with the aggregated evaluation metrics.
        """
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

    def evaluations_to_csv(self, file_name: str = 'test_evaluations.csv') -> None:
        """
        Save the aggregated evaluations to a CSV file.
        :param file_name: the name of the CSV file to save the evaluations to.
        """
        df = pd.DataFrame([self.evaluations], index=[0])
        df.to_csv(self.run_folder / file_name, index=False)

    def queries_to_csv(self, file_name: str = 'queries.csv') -> None:
        """
        Save the variable queries to a CSV file.
        :param file_name: the name of the CSV file to save the queries to.
        """
        df = pd.DataFrame([q.to_dict() for q in self.queries])
        df.to_csv(self.run_folder / file_name, index=False)