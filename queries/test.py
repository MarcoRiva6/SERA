import ast
import json
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import StrEnum
from numbers import Number
from pathlib import Path
from typing import TypeAlias, Any, get_type_hints

import numpy as np
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

def extract_json(raw_text: str) -> dict:
    """
    Extract the JSON part from the LLM output safely.
    """
    # find JSON block
    match = re.search(r"{(.|\n)*}", raw_text)
    if not match:
        raise ValueError("No valid JSON found in LLM output.")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except Exception as e:
        print("Failed to parse JSON. Raw JSON:")
        print(json_str)
        raise e


def ndcg_at_k(ground_truth, llm_ranking, k) -> float:
    # Truncate both lists to k
    gt_k = ground_truth[:k]
    llm_k = llm_ranking[:k]

    # Relevance: 1 if LLM's item at rank i is inside true top-k, else 0
    rel = np.array([1 if cust in gt_k else 0 for cust in llm_k])

    # DCG
    def dcg(scores):
        return np.sum([
            score / np.log2(idx + 2)
            for idx, score in enumerate(scores)
        ])

    dcg_llm = dcg(rel)

    # Ideal DCG = DCG of perfectly sorted relevance list: [1, 1, 1, ... 1] up to same number of relevant items
    ideal_rel = sorted(rel, reverse=True)
    dcg_ideal = dcg(ideal_rel)

    if dcg_ideal == 0:
        return 0.0

    return dcg_llm / dcg_ideal

@dataclass
class Query(ABC):
    """
    A single query consisting of a prompt, a response, and its evaluations.
    Prompt can be of any type, including a DataFrame (useful for lotus).
    """
    prompt: Any
    response: str
    evaluations: Evaluations
    response_json_schema: dict

    class Query(ABC):
        """
        A single query consisting of a prompt, a response, and its evaluations.
        Prompt can be of any type, including a DataFrame (useful for lotus).
        """
    prompt: Any
    response: str
    evaluations: Evaluations
    response_json_schema: str

    def to_dict(self) -> dict:
        base: dict = {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_")  # skip private/internal stuff, optional
        }
        # Normalize the prompt
        if isinstance(self.prompt, DataFrame):
            # You might want .to_dict() /.to_json() here depending on your use case
            base["prompt"] = self.prompt["prompt"]
        else:
            base["prompt"] = str(self.prompt)
        # Extract and flatten evaluations, if present
        evaluations = base.pop("evaluations", None)
        evals = {
            (k.value if hasattr(k, "value") else str(k)): v
            for k, v in (evaluations or {}).items()
        }
        # Merge: base now includes subclass attributes as well
        return {**base, **evals}

@dataclass
class Test(ABC):
    """
    Represents a single test, used to build queries.
    """
    family: str # the family this test belongs to (e.g. "movies", "molecules")
    name_path: str # the path-safe name of the test
    run_type: RunType
    run_folder: Path
    seed: int # random seed for reproducibility
    debug: bool = True
    queries: list[Query] = None # The list of queries generated for this test.
    evaluations: Evaluations = None # Aggregated evaluations across all queries for this test.

    def prepare_queries(self) -> None:
        """
        After this method is called, variable 'queries' should be populated.
        It automatically calls the appropriate 'prepare' method (based on the current run_type) as 'prepare_queries_for_<run_type>()'.
        """
        method_name = 'prepare_queries_for_' + self.run_type.value
        method = getattr(self, method_name)
        if method is not None:
            method()
        else:
            raise NotImplementedError(f'Run mode {self.run_type} not implemented for this test (method {method_name} does not exists).')

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
        :param file_name: the name of the CSV file to save the queries to (with extension).
        """
        df = pd.DataFrame([q.to_dict() for q in self.queries])
        df.to_csv(self.run_folder / file_name, index=False)

    def csv_to_queries(self, query_cls: type, file_name: str = 'queries.csv') -> None:
        """
        Load a CSV and convert each row into an instance of `cls`.
        Handles columns that are lists stored as strings.
        Used to restore an already computed list of queries previously sqaved to a CSV.
        :param query_cls: the class type to instantiate for each query (subclass of Query)
        """
        # Load CSV into dataframe
        df = pd.read_csv(self.run_folder / file_name)

        # Get type hints from the class (e.g., {"id": int, "scores": List[float]})
        hints = get_type_hints(query_cls)

        self.queries = []

        for _, row in df.iterrows():
            kwargs = {}

            for field_name, field_type in hints.items():
                try:
                    value = row[field_name]
                except KeyError:
                    kwargs[field_name] = None
                    continue  # Skip if the field is not in the CSV
                # Case 1: list fields (e.g. List[int], List[float])
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    # Convert the string to a real list
                    try:
                        parsed = ast.literal_eval(value) if isinstance(value, str) else value
                    except Exception:
                        parsed = []
                    value = parsed
                # Case 2: boolean strings like "True"/"False"
                if field_type is bool and isinstance(value, str):
                    value = value.lower() == "true"
                # Case 3: automatic type casting for int, float
                try:
                    value = field_type(value)
                except Exception:
                    pass  # fallback if casting isn't necessary or valid

                kwargs[field_name] = value

            self.queries.append(query_cls(**kwargs))
