import ast
import csv
import json
import os
import random
import re
import shutil
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import StrEnum
from numbers import Number
from pathlib import Path
from typing import TypeAlias, Any, get_type_hints, get_origin

import kagglehub
import math
import numpy as np
import pandas as pd
import requests
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
def extract_json(text: str, q_id=None) -> dict:
    """
    Extract the first JSON block from the given text.
    :param text: the text containing the JSON block.
    :param q_id: used in logging to identify the query.
    :return: the extracted JSON as a dictionary. Raises TypeError if failed.
    """
    # Find first {...} block (non-greedy, handles nested braces)
    matches = re.findall(r"{.*?}", text, flags=re.DOTALL)
    if not matches:
        raise TypeError("No JSON block found.")
    if len(matches) > 1:
        print(f"Query {q_id}: Multiple JSON blocks found, using the last one.")

    json_str = matches[-1]

    return json.loads(json_str)

def extract_list(text: str, q_id=None) -> list:
    """
    Extract the last list block from the given text.
    :param text: the text containing the list block.
    :param q_id: used in logging to identify the query.
    :return: the extracted list as a Python list, or an empty list if extraction/parsing fails.
    """
    # Find first [...] block (non-greedy, handles nested brackets)
    matches = re.findall(r"\[.*?]", text, flags=re.DOTALL)
    if not matches:
        raise TypeError("No list block found.")
    if len(matches) > 1:
        print(f"Query {q_id}: Multiple list blocks found, using the last one.")

    list_str = matches[-1]

    return json.loads(list_str)

def extract_pipe_sequence(input_text: str) -> list[list[str]]:
    """
    Extract sequences of words separated by pipes from the given text.
    :param input_text: the input text
    :return: a list of lists, where each inner list contains words from a line separated by pipes
    """
    lines = input_text.splitlines()
    pattern = re.compile(r"\b[^|\n]+(?:\s*\|\s*[^|\n]+)+\b")
    matched_lines = [line for line in lines if pattern.search(line)]
    return [[n.strip() for n in line.split("|") if n.strip()] for line in matched_lines]

def ndcg_at_k_scores(ground_truth_scores: list[float], llm_ranking_scores: list[float], k: int):
    """
    NDCG@k using graded relevance.
    Scores must be non-negative.
    """
    def dcg(scores):
        return sum([
            score / math.log2(i + 1)
            for i, score in enumerate(scores, start=1)
        ])

    llm_k = llm_ranking_scores[:k]
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

def miss_rate_at_k(ground_truth_full: list[Any], llm_ranking: list[Any], k: int) -> float:
    """
    Miss Rate @ k = (# of LLM's top-k items NOT in ground truth) / (total # of ground truth items)

    :param ground_truth_full: list of true top customers (sorted by similarity)
    :param llm_ranking: list of predicted top customers (sorted by similarity)
    :param k: evaluate only the LLM's top-k predictions
    :return: Miss Rate @ k
    """
    gt_set = set(ground_truth_full)
    llm_k_set = set(llm_ranking[:k])

    misses = sum(1 for c in llm_k_set if c not in gt_set)

    return misses / k if k > 0 else 0.0

def mare_at_k(ground_truth_full: list[Any], llm_ranking: list[Any], k: int) -> float:
    """
    Mean Absolute Rank Error @ k. It only considers items that appear in both rankings.

    :param ground_truth_full: *full* ranking list sorted by true similarity
    :param llm_ranking: predicted ranking list from LLM
    :param k: evaluate only the LLM's top-k predictions
    :return: MARE@k. k if no common items.
    """
    gt = ground_truth_full
    llm = llm_ranking[:k]
    # Position maps
    gt_pos = {c: i for i, c in enumerate(gt)}
    llm_pos = {c: i for i, c in enumerate(llm)}
    # Items that appear in both rankings
    common = set(gt).intersection(llm)

    if len(common) == 0:
        return k   # or return k, depending on your preference

    errors = [
        abs(gt_pos[c] - llm_pos[c])
        for c in common
    ]

    return sum(errors) / len(errors)


def spearman_rho_at_k(ground_truth_full: list[Any], llm_ranking: list[Any], k: int) -> float:
    """
    Spearman rank correlation (ρ) @ k based on global ground truth. It only considers items that appear in both rankings.

    - ground_truth_full: full ranking list sorted by true similarity
    - llm_ranking: predicted ranking list from the LLM (length >= k ideally)
    - k: evaluate only the LLM's top-k predictions

    For each of the LLM's top-k customers, we compare:
      - pred_rank: its position in the LLM ranking (0..k-1)
      - true_rank: its position in the full ground-truth ranking (0..N-1)
    """

    gt = ground_truth_full#[:k]
    llm = llm_ranking[:k]

    # Items that both rankings cover
    common = list(set(gt).intersection(llm))
    if len(common) < 2:
        return 0.0  # cannot compute correlation with <2 points

    # Position lookup
    gt_pos = {c: i for i, c in enumerate(gt)}
    llm_pos = {c: i for i, c in enumerate(llm)}

    # Build rank position vectors
    gt_order = [gt_pos[c] for c in common]
    llm_order = [llm_pos[c] for c in common]

    rho, _ = spearmanr(gt_order, llm_order)
    if np.isnan(rho):
        return 0.0

    return float(rho)

def ensure_kaggle_ds(ds_name: str, file_path: Path) -> None:
    """
    Ensure that the specified file from a Kaggle dataset is available locally.
    If not, download the dataset and move the file to the specified path.
    :param ds_name: the name of the Kaggle dataset
    :param file_path: the full local path where the file should be located.
    """
    if not os.path.exists(file_path):
        ds_folder = kagglehub.dataset_download(ds_name, force_download=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(Path(ds_folder) / file_path.name, file_path)

def download_csv(url: str, dest_path: Path) -> None:
    response = requests.get(url)

    if response.status_code == 200:
        content = response.text
        csv_reader = csv.reader(content.splitlines())
        header = next(csv_reader)
        rows = list(csv_reader)
        # Write the CSV data to a local file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
            csv_writer.writerows(rows)
    else:
        raise Exception(response.text)

@dataclass
class Query(ABC):
    """
    A single query consisting of a prompt, a response, and its evaluations.
    Prompt can be of any type, including a DataFrame (useful for lotus).
    """
    id: int
    prompt: Any
    response: str
    evaluations: Evaluations
    response_json_schema: dict
    parsing_failed: bool

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

    def evaluate(self) -> dict:
        """
        Aggregate evaluations across all queries for this test by computing the mean for each metric.
        The method should also populate the 'evaluations' variable.
        :return: A dictionary with the aggregated evaluation metrics.
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
        df.to_csv(dest / file_name, index=False, decimal=',', sep=';')

    def queries_to_df(self) -> pd.DataFrame:
        """
        Convert the variable queries to a Pandas DataFrame.
        :return: DataFrame containing all queries.
        """
        return pd.DataFrame([q.to_dict() for q in self.queries])

    def queries_to_csv(self, file_name: str, dest: Path = None) -> None:
        """
        Save the variable queries to a CSV file.
        :param dest: the destination folder to save the CSV file to.
        :param file_name: the name of the CSV file to save the queries to (with extension).
        """
        if dest is None:
            dest = self.run_folder
        df = self.queries_to_df()
        df.to_csv(dest / file_name, index=False, decimal=',', sep=';')

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
        df = pd.read_csv(folder / file_name, decimal=',', sep=';')
        # Get type hints from the class (e.g., {"id": int, "scores": List[float]})
        hints = get_type_hints(query_cls)

        self.queries = []

        for _, row in df.iterrows():
            kwargs = {}

            for field_name, field_type in hints.items():
                # 1. If the column doesn't exist → keep it as None
                try:
                    value = row[field_name]
                except KeyError:
                    kwargs[field_name] = None
                    continue

                # 2. Normalize pandas nulls (NaN, NA) and empty strings to None
                #    pd.isna handles np.nan, pd.NA, None
                if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                    kwargs[field_name] = None
                    continue

                origin = get_origin(field_type)
                # 3. List fields (e.g. list[int], list[float], list[str])
                if origin is list:
                    # If it's a string, try to parse it as a Python literal (e.g. "[1, 2, 3]")
                    if isinstance(value, str):
                        try:
                            value = ast.literal_eval(value)
                        except Exception:
                            # Could not parse; choose your default (None or [])
                            value = []
                    # optional: you could also enforce element types using get_args(field_type)

                # 4. Booleans from strings like "True"/"False"
                if field_type is bool and isinstance(value, str):
                    value = value.strip().lower() == "true"

                # 5. Light type casting for simple types (int, float, str, etc.)
                #    Skip if it's already of the right type.
                try:
                    if not isinstance(value, field_type):
                        value = field_type(value)
                except Exception:
                    # If casting fails, just leave the original value
                    pass

                kwargs[field_name] = value

            self.queries.append(query_cls(**kwargs))
