import os
import random
import sys
from itertools import product

from tqdm import tqdm

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)

import ast
import csv
import json
import re
import shutil
from abc import ABC
from dataclasses import dataclass, asdict, field
from enum import StrEnum
from pathlib import Path
from typing import get_type_hints, get_origin, TypeVar, Generic

import kagglehub
import pandas as pd
import requests

from pandas import DataFrame
from pydantic import BaseModel, ValidationError

from experiments.run_type import RunType
from metrics import *

data_folder: Path = Path(__file__).resolve().parent.parent / 'data' # path to the project's data folder

class PromptLevel(StrEnum):
    generic = 'generic'
    instruct = 'instruct'
    formula = 'formula'

class NamesLevel(StrEnum):
    fake = 'fake'
    real = 'real'

@dataclass
class TestParameters(ABC):
    """
    Base class for test parameters.
    All configuration parameters that could be changed via the yaml file should be declared here.
    """
    seed: int = 0 # if set to 0, a random seed is used
    n_queries: int = 10
    kp: list[float] = field(default_factory=lambda: [0.05, 0.15, 0.25])
    elems_per_query: list[int] = field(default_factory=lambda: [30, 70])
    prompt_levels: tuple[PromptLevel, ...] = tuple(PromptLevel)
    names_levels: tuple[NamesLevel, ...] = tuple(NamesLevel)
    enforce_json_schema: bool = True

T_TestParameters = TypeVar('T_TestParameters', bound=TestParameters)

@dataclass
class QueryParameters:
    prompt_level: PromptLevel
    names_level: NamesLevel
    k: int
    n_elems: int

@dataclass
class Evaluations:
    ndcg_scores: float
    ndcg_k: float
    mare: float
    mare_k: float
    spearman: float
    spearman_k: float
    kendall: float
    kendall_k: float
    hallucination_rate: float
    tokens: int

def mark_duplicates(llm_response: list[Any]) -> list[Any]:
    """
    Mark duplicated values in llm_response that are not in the correct position according to ground_truth.
    Duplicated values are replaced with the string "<duplicated>".
    """
    cleaned_ranking = []
    seen = set()

    for item in llm_response:
        if item not in seen:
            # È la prima occorrenza: la teniamo
            cleaned_ranking.append(item)
            seen.add(item)
        else:
            # È un duplicato
            cleaned_ranking.append("<duplicated>")

    return cleaned_ranking

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
class Query:
    """
    A single query consisting of a prompt, a response, and its evaluations.
    Prompt can be of any type, including a DataFrame (useful for lotus).
    """
    id: int # must be unique within a test
    ds_id: int
    prompt: str
    ground_truth: list[str]
    ground_truth_scores: list[float]
    parameters: QueryParameters
    response_json_schema: dict | None
    prompt_df: DataFrame | None = None
    response: str | DataFrame | None = None
    parsed_response: list[str] | None = None
    evaluations: Evaluations | None = None
    parsing_failed: bool | None = None
    tokens: int | None = None

    def to_dict(self) -> dict:
        base: dict = asdict(self)
        # Normalize the prompt
        if self.prompt_df is not None:
            base["prompt_df"] = self.prompt_df.to_string(index=False)
            if isinstance(self.response, DataFrame):
                base["response"] = self.response.to_string(index=False)
        # Extract and flatten parameters, if present
        try:
            parameters = base["parameters"]
            if not parameters:
                raise KeyError
            params = {
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in parameters.items()
            }
            del base["parameters"]
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
            del base["evaluations"]
        except KeyError:
            evals = {}

        return base | params | evals

@dataclass
class Test(ABC, Generic[T_TestParameters]):
    """
    Represents a single test, used to build queries.
    """
    family: str # the family this test belongs to (e.g. "movies", "molecules")
    name_path: str # the path-safe name of the test
    run_type: RunType
    run_folder: Path # /data/{direct | lotus}
    #json_schema: BaseModel
    queries: list[Query] = None # The list of queries generated for this test.
    prepared_queries_file_name: str = "prepared_queries.pkl" # Path to the file where prepared queries are stored.
    parameters: T_TestParameters = None
    evaluations: Evaluations = None # Aggregated evaluations across all queries for this test.
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
        current_params = json.loads(json.dumps(asdict(self.parameters)))
        return saved_params == current_params

    def generate_queries(self) -> None:
        """
        After this method is called, variable 'queries' should be populated.
        It automatically calls the appropriate 'prepare' method (based on the current run_type) as 'prepare_queries_for_<run_type>()'.
        """
        method_name = 'prepare_queries_for_' + "direct"
        try:
            method = getattr(self, method_name)
            method()
        except AttributeError:
            raise NotImplementedError(
                f'Run mode {self.run_type} not implemented for this test (method {method_name} does not exists).')

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        """first item is the query general DF, second is the current seed
        """
        raise NotImplementedError()

    def _anonymize_query_df(self, df: DataFrame) -> DataFrame:
        """returns the query DataFrame with real or fake names according to name_mode. If not overridden, returns the original DF."""
        return df

    def _select_query_target(self, current_seed, real_df: DataFrame, anon_df) -> tuple[str | int | None, str | int | None]:
        return None, None

    def _build_ground_truth(self, df: DataFrame, target: str|int|None) -> tuple[list[str | int], list[float]]:
        """returns a tuple with the list of ground truth ids and the list of their relevance scores
        :param target:
        """
        raise NotImplementedError()

    def _build_prompt_df(self, df: DataFrame) -> DataFrame:
        """Returns the cleaned version of the DF, ready for create_prompt. It's called right before creating the Query. By default, it returns the input."""
        return df

    def _create_prompt(self, df: DataFrame, target: str|int|None, k: int, prompt_level: PromptLevel) -> str:
        """returns the prompt string built according to the given parameters"""
        raise NotImplementedError()

    def _build_query(self, q_id: int, ds_id: int, df: DataFrame, target: str|int|None, k: int, gt_ids: list[str], gt_vals: list[float], prompt_level: PromptLevel, names_level: NamesLevel, n_elems: int) -> Query:
        return Query(
            id=q_id,
            ds_id=ds_id,
            prompt=self._create_prompt(df, target, k, prompt_level),
            prompt_df=df if self.run_type==RunType.LOTUS else None,
            ground_truth=gt_ids,
            ground_truth_scores=gt_vals,
            parameters=QueryParameters(k=k, prompt_level=prompt_level, names_level=names_level, n_elems=n_elems),
            response_json_schema=self.json_schema.model_json_schema() if self.parameters.enforce_json_schema else None,
        )

    def _init_query(self, seed: int, df: DataFrame, elem_per_query: int) -> tuple[tuple[DataFrame, str | int | None, list[int | str], list[float]],tuple[DataFrame, str | int | None, list[int | str], list[float]]]:
        sampled_df = self._sample_for_query(seed, df, elem_per_query)
        anon_sampled_df = self._anonymize_query_df(sampled_df)
        target, anon_target = self._select_query_target(seed, sampled_df, anon_sampled_df)
        gt_ids, gt_scores = self._build_ground_truth(sampled_df, target)
        anon_gt_ids, anon_gt_scores = self._build_ground_truth(anon_sampled_df, anon_target)
        prompt_df = self._build_prompt_df(sampled_df)
        anon_prompt_df = self._build_prompt_df(anon_sampled_df)
        return (prompt_df, target, gt_ids, gt_scores), (anon_prompt_df, anon_target, anon_gt_ids, anon_gt_scores)

    def _prepare_queries(self, clean_df) -> list[Query]:
        queries = []
        current_seed = self.parameters.seed
        counter = 0
        ds_id = 0

        total_iters = (self.parameters.n_queries
                       * len(self.parameters.elems_per_query)
                       * len(self.parameters.kp)
                       * len(self.parameters.prompt_levels)
                       * len(self.parameters.names_levels))
        with tqdm(total=total_iters, desc="Generating queries", unit='query', colour='green') as pbar:

            for _ in range(self.parameters.n_queries):
                for elem_per_query in self.parameters.elems_per_query:
                    if self.parameters.seed != 0:
                        random.seed(current_seed)

                    (real_prompt_df, real_target, real_gt_ids, real_gt_scores), (anon_prompt_df, anon_target, anon_gt_ids, anon_gt_scores) = self._init_query(
                        current_seed, clean_df, elem_per_query)

                    for kp, name_mode, prompt_level in product(self.parameters.kp, self.parameters.names_levels, self.parameters.prompt_levels):
                        k = max(1, math.ceil(kp * elem_per_query))
                        match name_mode:
                            case NamesLevel.real:
                                prompt_df = real_prompt_df
                                target = real_target
                                gt_ids = real_gt_ids
                                gt_scores = real_gt_scores
                            case NamesLevel.fake:
                                prompt_df = anon_prompt_df
                                target = anon_target
                                gt_ids = anon_gt_ids
                                gt_scores = anon_gt_scores

                        query = self._build_query(q_id=counter, ds_id=ds_id, df=prompt_df, target=target, k=k,
                                                  gt_ids=gt_ids, gt_vals=gt_scores, prompt_level=prompt_level,
                                                  names_level=name_mode, n_elems=elem_per_query)
                        queries.append(query)

                        counter += 1
                        pbar.update(1)

                    current_seed += 1
                    ds_id += 1

        return queries

    def _load_ds(self) -> DataFrame:
        """loads the dataset file as a DataFrame"""
        raise NotImplementedError()

    def _prepare_df(self, df: DataFrame) -> DataFrame:
        """Here the loaded dataset's DataFrame" can be processed, if necessary. By default, it passes the input."""
        return df

    def _ensure_enough_combinations(self, df: DataFrame) -> bool:
        for elem_per_query in self.parameters.elems_per_query:
            if math.comb(len(df), elem_per_query) < self.parameters.n_queries:
                return False
        return True

    def prepare_queries_for_direct(self) -> None:
        print('loading dataset...')
        full_df = self._load_ds()
        print('preparing DF...')
        clean_df = self._prepare_df(full_df)
        print('initializing queries...')
        if not self._ensure_enough_combinations(clean_df):
            raise ValueError(
                f"Not enough unique combinations of elems to generate the requested number of queries.")
        self.queries = self._prepare_queries(clean_df)

    def _parse_query_manual(self, query: Query) -> bool:
        raise NotImplementedError("Parsing of not structured output is not implemented.")

    def parse_query(self, query: Query) -> None:
        match query.response:
            case DataFrame():
                try:
                    # take the first column as the response list
                    query.parsed_response = query.response[self.named_index_col].tolist()
                    parsing_failed = False
                except IndexError:
                    parsing_failed = True
            case str():
                if query.response_json_schema is not None:
                    try:
                        parsed_response = self.json_schema.model_validate_json(query.response)
                        query.parsed_response = getattr(parsed_response, list(self.json_schema.model_fields.keys())[0])
                        parsing_failed = False
                    except ValidationError:
                        parsing_failed = True
                else:
                    parsing_failed = self._parse_query_manual(query)
            case _:
                parsing_failed = True

        if parsing_failed:
            print(f"Couldn't evaluate query {query.id}. Response was: {query.response}")
        query.parsing_failed = parsing_failed

    def evaluate_query(self, query: Query) -> Evaluations:
        """
        Evaluate a single query after submission.
        :param query: the query to evaluate
        :return: A dictionary (an Evaluations object: dict[Metric, Number]) with the evaluation metrics for the query.
        """
        failing_scores = Evaluations(kendall=0.0, kendall_k=0.0, ndcg_scores=0.0, ndcg_k=0.0, mare=query.parameters.k, mare_k=query.parameters.k, spearman=-1.0, spearman_k=-1.0, hallucination_rate=1, tokens=query.tokens if query.tokens is not None else 0)

        if query.parsing_failed or len(query.parsed_response) < query.parameters.k:
            return failing_scores
        assert query.parsed_response is not None

        llm_answer_with_duplicates = query.parsed_response[:query.parameters.k]
        llm_answer = mark_duplicates(llm_answer_with_duplicates)
        llm_scores: list[float] = [query.ground_truth_scores[query.ground_truth.index(elem)]
                                   if elem in query.ground_truth
                                   else 0.0
                                   for elem in llm_answer]

        return Evaluations(ndcg_scores=ndcg_k(llm_scores, query.ground_truth_scores, query.parameters.k),
                           ndcg_k=ndcg_k(
                               relevance_scores=standard_ndcg_scoring(llm_answer, query.ground_truth,
                                                                      query.parameters.k),
                               k=query.parameters.k),
                           mare=mare_k(llm_answer, query.ground_truth),
                           mare_k=mare_k(llm_answer, query.ground_truth, query.parameters.k),
                           kendall=kendall_tau_k(llm_answer, query.ground_truth),
                           kendall_k=kendall_tau_k(llm_answer, query.ground_truth, query.parameters.k),
                           spearman=spearman_rho_k(llm_answer, query.ground_truth),
                           spearman_k=spearman_rho_k(llm_answer, query.ground_truth, query.parameters.k),
                           hallucination_rate=hallucination_rate(llm_answer_with_duplicates, query.ground_truth),
                           tokens=query.tokens if query.tokens is not None else 0)

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

    def queries_to_pickle(self, path: Path) -> None:
        import pickle
        with path.open("wb") as f:
            pickle.dump(self.queries, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_to_queries(self, path: Path) -> None:
        import pickle
        with path.open("rb") as f:
            self.queries = pickle.load(f)

    def store_queries(self) -> None:
        """
        Store the variable queries as a CSV file in the run folder.
        """
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.queries_to_pickle(self.run_folder / self.prepared_queries_file_name)

    def restore_queries(self) -> None:
        self.pickle_to_queries(self.run_folder / self.prepared_queries_file_name)

    def queries_to_csv(self, file_name: str, dest: Path = None) -> None:
        """
        Save the variable queries to a CSV file.
        :param dest: the destination folder to save the CSV file to.
        :param file_name: the name of the CSV file to save the queries to (with extension).
        """
        exclude_columns = ['response_json_schema']
        if dest is None:
            dest = self.run_folder

        df = self.queries_to_df()
        df = df.drop(columns=exclude_columns)

        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / file_name, index=False, decimal=',', sep=';')

    def csv_to_queries(self, query_cls: type, params_cls: type, evals_cls: type, file_name: str, folder: Path = None) -> None:
        """
        Used to restore an already computed list of queries previously saved to a CSV.
        Automatically sets the variable `queries`.
        :param params_cls:
        :param evals_cls:
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

        def parse(row, f_name, f_type) -> Any:
            value = None
            # 1. If the column doesn't exist → keep it as None
            try:
                value = row[f_name]
            except KeyError:
                return value
            # 2. Normalize pandas nulls (NaN, NA) and empty strings to None
            #    pd.isna handles np.nan, pd.NA, None
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                return None
            origin = get_origin(f_type)
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
            if f_type is bool and isinstance(value, str):
                value = value.strip().lower() == "true"
            # 5. Light type casting for simple types (int, float, str, etc.)
            #    Skip if it's already of the right type.
            try:
                if not isinstance(value, f_type):
                    value = f_type(value)
            except Exception:
                # If casting fails, just leave the original value
                pass

            return value

        for _, row in df.iterrows():
            kwargs = {}

            for field_name, field_type in hints.items():
                if field_name == 'parameters':
                    param_kwargs = {}
                    for field_name, field_type in get_type_hints(params_cls).items():
                        param_kwargs[field_name] = parse(row, field_name, field_type)
                    kwargs['parameters'] = params_cls(**param_kwargs)
                elif field_name == 'evaluations':
                    eval_kwargs = {}
                    for field_name, field_type in get_type_hints(evals_cls).items():
                        eval_kwargs[field_name] = parse(row, field_name, field_type)
                    kwargs['evaluations'] = evals_cls(**eval_kwargs)
                kwargs[field_name] = parse(row, field_name, field_type)

            self.queries.append(query_cls(**kwargs))

def queries_to_df(queries: list[Query]) -> pd.DataFrame:
    """
    Convert the variable queries to a Pandas DataFrame.
    :return: DataFrame containing all queries.
    """
    df = pd.DataFrame([q.to_dict() for q in queries])
    put_first = ['id', 'prompt', 'response_json_schema']
    return df[put_first + [col for col in df.columns if col not in put_first]]