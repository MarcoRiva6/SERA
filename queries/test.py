import ast
import json
import random
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
from scipy.stats import spearmanr

from experiments.run_type import RunType

data_folder: Path = Path(__file__).parent.parent / 'data' # path to the project's data folder

class Metric(StrEnum):
    """
    Enumeration of possible metrics for a test.
    """
    pass

Evaluations: TypeAlias = dict[Metric, Number] # A dictionary mapping metrics to their numeric evaluation values.
def extract_json(text: str) -> str:
    """
    Extract the first JSON block from the given text.
    :param text: the text containing the JSON block.
    :return: the extracted JSON as a dictionary, or an empty string if extraction/parsing fails.
    """
    # Find first {...} block (non-greedy, handles nested braces)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        print("No JSON block found.")
        return ""
    if len(match.groups()) > 1:
        print("Multiple JSON blocks found, using the last one.")

    json_str = match.lastgroup

    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError:
        print("Failed to parse JSON.")
        return ""

def extract_list(text: str) -> list:
    """
    Extract the first list block from the given text.
    :param text: the text containing the list block.
    :return: the extracted list as a Python list, or an empty list if extraction/parsing fails.
    """
    # Find first [...] block (non-greedy, handles nested brackets)
    matches = re.findall(r"\[.*\]", text, flags=re.DOTALL)
    if not matches:
        print("No list block found.")
        return []
    if len(matches) > 1:
        print("Multiple list blocks found, using the last one.")

    list_str = matches[-1]

    try:
        return json.loads(list_str)
    except json.decoder.JSONDecodeError:
        print("Failed to parse list.")
        return []

#TODO: È COMPLETANEMTE ROTTO
#TODO: sistemare questo, che così è inutile e specifico per custumer_segmentation...
def ndcg_at_k_scores(ground_truth_scores, llm_ranking, k):
    """
    More powerful NDCG@k using graded relevance:
    ground_truth_scores is a dict {customer_id: similarity_score}
    """
    llm_k = llm_ranking[:k]

    def dcg(scores):
        return np.sum([
            score / np.log2(i + 2)
            for i, score in enumerate(scores)
        ])

    dcg_llm = dcg(llm_k)

    # Ideal DCG uses sorted true scores
    ideal_scores = sorted(ground_truth_scores, reverse=True)[:k]
    dcg_ideal = dcg(ideal_scores)

    if dcg_ideal == 0:
        return 0.0

    return dcg_llm / dcg_ideal


def precision_at_k(ground_truth, llm_ranking, k):
    """
    Precision@k = (# correctly predicted in top-k) / (# predicted in top-k)
    ground_truth: list of true top customers (sorted by similarity)
    llm_ranking: list of predicted top customers (sorted by similarity)
    """
    gt_k = set(ground_truth[:k])
    llm_k = llm_ranking[:k]

    if len(llm_k) == 0:
        return 0.0

    tp = sum(1 for c in llm_k if c in gt_k)
    return tp / len(llm_k)


def spearman_rank_correlation(ground_truth, llm_ranking):
    """
    Spearman rank correlation between two rankings.

    Both inputs are lists of customer IDs ordered from most to least similar.
    We only consider the items that appear in both lists.
    """
    common = list(set(ground_truth).intersection(llm_ranking))

    # Need at least 2 points to define correlation
    if len(common) < 2:
        return 0.0

    gt_pos = {cust: i for i, cust in enumerate(ground_truth)}
    llm_pos = {cust: i for i, cust in enumerate(llm_ranking)}

    gt_order = [gt_pos[c] for c in common]
    llm_order = [llm_pos[c] for c in common]

    rho, _ = spearmanr(gt_order, llm_order)
    if np.isnan(rho):
        return 0.0

    return float(rho)


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
        try:
            method = getattr(self, method_name)
            method()
        except AttributeError:
            raise NotImplementedError(
                f'Run mode {self.run_type} not implemented for this test (method {method_name} does not exists).')

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

    def evaluations_to_csv(self, file_name: str = 'test_evaluations.csv', dest: Path = None) -> None:
        """
        Save the aggregated evaluations to a CSV file.
        :param dest: the destination folder to save the CSV file to.
        :param file_name: the name of the CSV file to save the test evaluations to.
        """
        if dest is None:
            print('Warning: No destination folder provided for evaluations CSV. Using run_folder.')
            dest = self.run_folder
        df = pd.DataFrame([self.evaluations], index=[0])
        df.to_csv(dest / file_name, index=False)

    def queries_to_csv(self, file_name: str, dest: Path = None) -> None:
        """
        Save the variable queries to a CSV file.
        :param dest: the destination folder to save the CSV file to.
        :param file_name: the name of the CSV file to save the queries to (with extension).
        """
        if dest is None:
            dest = self.run_folder
        df = pd.DataFrame([q.to_dict() for q in self.queries])
        df.to_csv(dest / file_name, index=False)

    def csv_to_queries(self, query_cls: type, file_name: str, folder: Path = None) -> None:
        """
        Load a CSV and convert each row into an instance of `cls`.
        Handles columns that are lists stored as strings.
        Used to restore an already computed list of queries previously saved to a CSV.
        :param query_cls: the class type to instantiate for each query (subclass of Query)
        :param folder: The directory where the CSV file is located. If None, uses the current run folder.
        :param file_name: the name of the CSV file to load (with extension).
        """
        if folder is None:
            folder = self.run_folder
        # Load CSV into dataframe
        df = pd.read_csv(folder / file_name)

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
