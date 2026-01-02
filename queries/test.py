import ast
import csv
import json
import os
import re
import shutil
from abc import abstractmethod, ABC
from dataclasses import dataclass, field, asdict
from enum import StrEnum
from numbers import Number
from pathlib import Path
from typing import TypeAlias, Any, get_type_hints, get_origin, Counter, TypeVar, Generic

import kagglehub
import math
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from scipy.stats import spearmanr

from experiments.run_type import RunType

data_folder: Path = Path(__file__).resolve().parent.parent / 'data' # path to the project's data folder

@dataclass
class TestParameters(ABC):
    """
    Base class for test parameters.
    All configuration parameters that could be changed via the yaml file should be declared here.
    """
    seed: int = 0 # if set to 0, a random seed is used

T_TestParameters = TypeVar('T_TestParameters', bound=TestParameters)

@dataclass
class QueryParameters(ABC):
    """
    Base class for query parameters.
    """
    pass

T_QueryParameters = TypeVar('T_QueryParameters', bound=QueryParameters)

@dataclass
class Evaluations(ABC):
    pass

T_Evaluations = TypeVar('T_Evaluations', bound=Evaluations)

def mark_duplicates(llm_response: list[Any], ground_truth: list[Any]) -> list[Any]:
    """
    Mark duplicated values in llm_response that are not in the correct position according to ground_truth.
    Duplicated values are replaced with the string "<duplicated>".
    """
    # Count occurrences in llm_response
    counts = Counter(llm_response)

    # Map ground_truth value -> its index
    gt_index = {value: i for i, value in enumerate(ground_truth)}

    result = llm_response.copy()

    for i, value in enumerate(llm_response):
        # Only care about duplicated values
        if counts[value] > 1 and value in gt_index:
            # If this is NOT the correct position, mark as duplicated
            if i != gt_index[value]:
                result[i] = "<duplicated>"

    return result

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
    pattern = re.compile(r"[^|\n]+(?:\s*\|\s*[^|\n]+)+")
    matched_lines = [line for line in lines if pattern.search(line)]
    return [[n.strip() for n in line.split("|") if n.strip()] for line in matched_lines]

def extract_separator_sequence(input_text: str, separator: str, top_k: int) -> list[list[str]]:
    """
    Extract sequences of strings separated by a custom separator from the given text.
    :param input_text: the input text
    :param separator: the custom separator string
    :param top_k: expected number of elements in each sequence.
    :return: a list of lists, each of length between top_k and top_k+2
    """
    lines = input_text.splitlines()
    matched_lines = [line for line in lines if top_k - 1 <= line.count(separator) <= top_k + 1] # at least top_k-1 separators and at most top_k+1
    result = []
    for line in matched_lines:
        parts = [n.strip(" .") for n in line.split(separator) if n.strip(" .")] # remove leading/trailing spaces and dots
        new_parts = [p for p in parts if p != ''] # remove empty strings
        # keep only lists with length between top_k and top_k+2 (to account for cases with 'separator' at the beginning and/or end)
        if top_k <= len(new_parts) <= top_k + 2:
            result.append(new_parts)
    return result

def _dcg(scores):
    return sum([
        score / math.log2(i + 1)
        for i, score in enumerate(scores, start=1)
    ])

def ndcg_k(relevance_scores: list[Number], ground_truth_scores: list[Number]=None, k: int=None):
    """
    NDCG@k. It computes the Normalized Discounted Cumulative Gain for *relevance_scores* and *ground_truth_scores*,
    then normalizes the result.
    All scores must be non-negative.
    :param relevance_scores: the relevance scores, used as they are.
    :param ground_truth_scores: scores for the ideal ranking. Defaults to [k, k-1, ..., 1] if not specified.
    :param k: consider only the top-k elements. If not specified, uses the full length of *relevance_scores*.
    :return: NDCG value.
    """
    if k is None:
        k = len(relevance_scores)

    llm = relevance_scores[:k]
    gt = ground_truth_scores[:k] if ground_truth_scores is not None else [k - i for i in range(k)]
    dcg_llm = _dcg(llm)
    dcg_ideal = _dcg(gt)

    return dcg_llm / dcg_ideal

def precision_at_k(llm_ranking, ground_truth, k):
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

def hallucination_rate(llm_ranking: list[Any], ground_truth: list[Any]) -> float:
    """
    Miss Rate = (# of LLM's top-k items NOT in ground truth) / (total # of ground truth items)

    :param llm_ranking: list of predicted top customers (sorted by similarity)
    :param ground_truth: list of true top customers (sorted by similarity)
    :return: Miss Rate
    """
    length = len(llm_ranking)
    gt_set = set(ground_truth)
    llm_k_set = set(llm_ranking)

    misses = sum(1 for c in llm_k_set if c not in gt_set)

    return misses / length if length > 0 else 0.0

def mare_k(llm_ranking: list[Any], ground_truth: list[Any], k: int=None) -> float:
    """
    Mean Absolute Rank Error @ k. Computes the distance between the predicted and true ranks and computes its mean.
    If an element from *llm_ranking* is missing in *ground_truth*, it is penalized with *k* or *len(ground_truth)* (see below).

    :param llm_ranking: predicted ranking list from LLM
    :param ground_truth: ranking list sorted by true similarity
    :param k: consider only the top-k elements for both *llm_ranking* and *ground_truth*.
        If not specified, uses the full length for both; in this case, if an element from *llm_ranking* is still missing
         in *ground_truth*, it is penalized with *len(ground_truth)*.
    :return: MARE@k.
    """
    if k is not None:
        llm = llm_ranking[:k]
        gt = ground_truth[:k]
    else:
        llm = llm_ranking
        gt = ground_truth
    penality = len(ground_truth)
    # Position maps
    llm_pos_position_map = {c: i for i, c in enumerate(llm)}
    gt_position_map = {c: i for i, c in enumerate(gt)}

    errors = [
        abs(gt_position_map[c] - llm_pos_position_map[c]) if c in gt_position_map else penality
        for c in llm
    ]

    return sum(errors) / len(errors)

def spearman_rho_k(llm_ranking: list[Any], ground_truth: list[Any], k: int = None) -> float:
    """
    Spearman rank correlation (ρ) @ k.
    It computes the spearman ranking correlation between *llm_ranking* and *ground_truth* rankings. It penalizes missing
    elements in *llm_ranking* with *k* or *len(ground_truth)* (see below).

    :param llm_ranking: predicted ranking list from the LLM
    :param ground_truth: ranking list sorted by true similarity
    :param k: consider only the top-k elements for both *llm_ranking* and *ground_truth*.
        If not specified, uses the full length of both; in this case, if an element from *llm_ranking* is still missing
            in *ground_truth*, it is penalized with *len(ground_truth)*.
    """
    if k is None:
        llm = llm_ranking
        gt = ground_truth
    else:
        llm = llm_ranking[:k]
        gt = ground_truth[:k]
    penality = len(ground_truth)

    if len(llm_ranking) < 2:
        return 0.0  # cannot compute correlation with <2 points
    # Position lookup
    gt_position_map = {c: i for i, c in enumerate(gt)}
    # Build rank position vectors
    llm_order = [gt_position_map[c] if c in gt else penality
                 for c in llm]
    gt_order = range(len(llm_order))

    # check for constant list
    if all(x == llm_order[0] for x in llm_order):
        return 0.0

    rho, _ = spearmanr(gt_order, llm_order)
    if np.isnan(rho): # constant list, but should not happen due to previous check
        return 0.0
    return float(rho)

def kendall_tau_k(llm_ranking: list[Any], ground_truth: list[Any], k: int = None) -> float:
    """
    Kendall's Tau @ k.
    It computes the kendall tau ranking correlation between *llm_ranking* and *ground_truth* rankings. It penalizes missing
    elements in *llm_ranking* with *k* or *len(ground_truth)* (see below).

    :param llm_ranking: predicted ranking list from the LLM
    :param ground_truth: ranking list sorted by true similarity
    :param k: consider only the top-k elements for both *llm_ranking* and *ground_truth*.
        If not specified, uses the full length of both; in this case, if an element from *llm_ranking* is still missing
            in *ground_truth*, it is penalized with *len(ground_truth)*.
    """
    if k is None:
        llm = llm_ranking
        gt = ground_truth
    else:
        llm = llm_ranking[:k]
        gt = ground_truth[:k]
    penality = len(ground_truth)

    if len(llm_ranking) < 2:
        return 0.0  # cannot compute correlation with <2 points
    # Position lookup
    gt_position_map = {c: i for i, c in enumerate(gt)}
    # Build rank position vectors
    llm_order = [gt_position_map[c] if c in gt else penality
                 for c in llm]
    gt_order = range(len(llm_order))

    from scipy.stats import kendalltau
    tau, _ = kendalltau(gt_order, llm_order)
    if np.isnan(tau):
        return 0.0
    return float(tau)

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
        shutil.move(Path(ds_folder) / file_path.name, file_path) # move the DS

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
class Query(ABC, Generic[T_QueryParameters, T_Evaluations]):
    """
    A single query consisting of a prompt, a response, and its evaluations.
    Prompt can be of any type, including a DataFrame (useful for lotus).
    """
    id: int # must be unique within a test
    prompt: Any
    response: str
    parameters: T_QueryParameters
    evaluations: T_Evaluations
    response_json_schema: dict
    parsing_failed: bool

    def to_dict(self) -> dict:
        base: dict = asdict(self)
        # Normalize the prompt
        if isinstance(self.prompt, DataFrame):
            base["prompt"] = self.prompt["prompt"]
        else:
            base["prompt"] = str(self.prompt)
        # Extract and flatten parameters, if present
        try:
            parameters = base["parameters"]
            if not parameters:
                raise KeyError
            params = {
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in parameters.items()
            }
        except KeyError:
            params = {}
        # Extract and flatten evaluations, if present
        try:
            evaluations = base["evaluations"]
            if not evaluations:
                raise KeyError
            evals = {
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in evaluations.items()
            }
        except KeyError:
            evals = {}

        return base | params | evals

T_Query = TypeVar("T_Query", bound=Query)

@dataclass
class Test(ABC, Generic[T_Query, T_TestParameters, T_Evaluations]):
    """
    Represents a single test, used to build queries.
    """
    family: str # the family this test belongs to (e.g. "movies", "molecules")
    name_path: str # the path-safe name of the test
    run_type: RunType
    run_folder: Path # /data
    debug: bool = True
    queries: list[T_Query] = None # The list of queries generated for this test.
    prepared_queries_file_name: str = "prepared_queries.pkl" # Path to the file where prepared queries are stored.
    parameters: T_TestParameters = None
    evaluations: T_Evaluations = None # Aggregated evaluations across all queries for this test.
    params_file_name: str = 'test_params.json'

    def save_params(self) -> None:
        """
        Save the test configuration to a JSON file.
        """
        path: Path = self.run_folder / self.params_file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self.parameters), f, indent=4)

    def same_params(self) -> bool:
        """
        Compare the current test configuration with a previously saved one.
        :return: True if the configurations match, False otherwise.
        """
        with open(self.run_folder / self.params_file_name, 'r') as f:
            saved_params = json.load(f)
        current_params = asdict(self.parameters)
        return saved_params == current_params

    def generate_queries(self) -> None:
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
    def evaluate_query(self, query: Query) -> T_Evaluations:
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
        totals = {}

        # sum all metrics across queries
        for q in self.queries:
            for metric, value in asdict(q.evaluations).items():
                totals.update({metric: totals.get(metric, 0) + value})

        # compute mean values
        n = len(self.queries)
        aggregated = {metric: totals[metric] / n for metric in totals}

        self.evaluations = self.queries[0].evaluations.__class__(**aggregated)
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
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / file_name, index=False, decimal=',', sep=';')

    def queries_to_df(self) -> pd.DataFrame:
        """
        Convert the variable queries to a Pandas DataFrame.
        :return: DataFrame containing all queries.
        """
        return queries_to_df(self.queries)

    def store_queries(self) -> None:
        """
        Store the variable queries as a CSV file in the run folder.
        The file name is 'queries_<test_name_path>.csv'.
        """
        import pickle
        self.run_folder.mkdir(parents=True, exist_ok=True)
        with (self.run_folder / self.prepared_queries_file_name).open("wb") as f:
            pickle.dump(self.queries, f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_queries(self) -> None:
        import pickle
        with (self.run_folder / self.prepared_queries_file_name).open("rb") as f:
            self.queries = pickle.load(f)

    def queries_to_csv(self, file_name: str, dest: Path = None) -> None:
        """
        Save the variable queries to a CSV file.
        :param dest: the destination folder to save the CSV file to.
        :param file_name: the name of the CSV file to save the queries to (with extension).
        """
        if dest is None:
            dest = self.run_folder
        df = self.queries_to_df()
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / file_name, index=False, decimal=',', sep=';')

    def csv_to_queries(self, query_cls: type, file_name: str, folder: Path = None) -> None:
        """
        Used to restore an already computed list of queries previously saved to a CSV.
        Automatically sets the variable `queries`.
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
                    # optional: could also enforce element types using get_args(field_type)
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

def queries_to_df(queries: list[Query]) -> pd.DataFrame:
    """
    Convert the variable queries to a Pandas DataFrame.
    :return: DataFrame containing all queries.
    """
    df = pd.DataFrame([q.to_dict() for q in queries])
    put_first = ['id', 'prompt', 'response_json_schema']
    return df[put_first + [col for col in df.columns if col not in put_first]]