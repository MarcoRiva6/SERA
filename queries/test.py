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
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from enum import StrEnum
from pathlib import Path
from typing import get_type_hints, get_origin, Literal

import kagglehub
import pandas as pd
import requests

from pandas import DataFrame
from pydantic import BaseModel, ValidationError

from experiments.run_type import RunType
from metrics import *

data_folder: Path = Path(__file__).resolve().parent.parent / 'data' # path to the project's data folder
type GroundTruthScoreElem = float
type GroundTruthScoreList = list[float]
type PrintingMode = Literal['plain', 'markdown']

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
    setwise_partition_rate: list[int] = field(default_factory=lambda: [10])
    prompt_printing_modes: list[PrintingMode] = field(default_factory=lambda: ['plain'])

@dataclass
class QueryParameters:
    prompt_level: PromptLevel
    names_level: NamesLevel
    k: int
    n_elems: int
    prompt_printing_mode: PrintingMode

@dataclass
class PartitionedQueryParameters(QueryParameters):
    partition_rate: int

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

def mark_duplicates[T: (str,int)](llm_response: list[T]) -> list[T]:
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
            match item:
                case str():
                    to_append = "<duplicated>"
                case int():
                    to_append = -1
                case _:
                    raise ValueError("type in llm_response not supported")
            cleaned_ranking.append(to_append)

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

def sample_ds_interesting(df: pd.DataFrame, length: int, min_unique: int, key: str) -> pd.DataFrame:
    """
    Estrae esattamente 'length' righe, garantendo almeno 'min_unique' id univoci.
    Massimizza il numero di righe per id.
    """
    indici_per_id = df.groupby(key).groups
    # Ordiniamo gli ID in base a quante righe hanno (dal più grande al più piccolo).
    # Questo garantisce che sceglieremo "serbatoi" grandi che possono assorbire molte righe.
    tutti_id = list(indici_per_id.keys())
    # 1. Ordiniamo rigorosamente dal più grande al più piccolo
    tutti_id.sort(key=lambda x: len(indici_per_id[x]), reverse=True)
    # 2. CAOS CONTROLLATO: Definiamo quanto deve essere grande il "bacino" di pesca
    # Moltiplicare k per 3 o 4 dà ottima varietà garantendo comunque household molto densi
    ampiezza_bacino = min(len(tutti_id), min_unique * 4)
    # 3. Estraiamo l'élite, la mischiamo, e la rimettiamo in cima
    elite_id = tutti_id[:ampiezza_bacino]
    random.shuffle(elite_id)
    tutti_id[:ampiezza_bacino] = elite_id

    id_potenziali_multi = [uid for uid in tutti_id if len(indici_per_id[uid]) >= 2]

    m_min = (min_unique + 1) // 2
    s_max = min_unique - m_min

    if len(id_potenziali_multi) < m_min:
        raise ValueError(f"Servono almeno {m_min} id con >=2 righe.")

    # Selezioniamo i 'min_unique' elementi iniziali e assegniamo il minimo
    scelti_multi = id_potenziali_multi[:m_min]
    rimanenti_disponibili = [uid for uid in tutti_id if uid not in scelti_multi]
    scelti_resto = rimanenti_disponibili[:s_max]

    quote = {}
    for uid in scelti_multi: quote[uid] = 2
    for uid in scelti_resto: quote[uid] = 1

    righe_assegnate = sum(quote.values())
    if length < righe_assegnate:
        raise ValueError(f"l={length} è troppo piccolo. Ne servono almeno {righe_assegnate}.")

    da_assegnare = length - righe_assegnate

    # Distribuzione densa (Il cambiamento principale)
    # espandibili = elementi attualmente selezionati che hanno ancora righe a disposizione
    espandibili = {uid for uid in quote if quote[uid] < len(indici_per_id[uid])}
    non_ancora_scelti = set(rimanenti_disponibili[s_max:])

    while da_assegnare > 0:
        if espandibili:
            # PRIMA SCELTA ASSOLUTA: Aggiungiamo righe a chi è già nel pool!
            # Non introduciamo nuovi elementi finché c'è spazio in questi.
            uid = random.choice(list(espandibili))
            quote[uid] += 1
            da_assegnare -= 1

            # Se ha esaurito tutte le sue righe originali, non è più espandibile
            if quote[uid] == len(indici_per_id[uid]):
                espandibili.remove(uid)
        else:
            # Tutti i 'k' elementi selezionati sono stati "spremuti" al massimo
            # e ci mancano ancora righe per arrivare a 'l'.
            # Solo in questo caso introduciamo un NUOVO elementi.
            if not non_ancora_scelti:
                raise ValueError("Dataset esaurito! Hai richiesto più righe di quelle totali disponibili.")

            # Prendiamo il prossimo elementi disponibile
            # Visto che non usiamo random.pop() sui set per mantenere l'ordinamento:
            uid = next(uid for uid in tutti_id if uid in non_ancora_scelti)
            non_ancora_scelti.remove(uid)

            # Gli assegnamo la sua prima riga
            quote[uid] = 1
            da_assegnare -= 1
            if len(indici_per_id[uid]) > 1:
                espandibili.add(uid)

    # Estrazione vera e propria dei dati
    indici_finali = []
    for uid, num_righe in quote.items():
        indici_scelti = random.sample(list(indici_per_id[uid]), num_righe)
        indici_finali.extend(indici_scelti)

    df_finale = df.loc[indici_finali].sample(frac=1).reset_index(drop=True)

    return df_finale

@dataclass
class Query[GTT: (str,int), QPT: QueryParameters](ABC):
    id: int # must be unique within a test
    ds_id: int
    ground_truth: list[GTT]
    ground_truth_scores: GroundTruthScoreList
    parameters: QPT
    parsed_response: list[GTT] | None
    evaluations: Evaluations | None
    parsing_failed: bool | None
    tokens: int | None

    # necessario per poter salvare l'oggetto con pickle, causa GTT
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('__orig_class__', None)
        return state

    def to_dict(self) -> dict:
        base: dict = asdict(self)
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

    @abstractmethod
    def is_answered(self) -> bool:
        pass

@dataclass
class DirectQuery[GTT: (str,int)](Query[GTT, QueryParameters]):
    prompt: str
    prompt_df: DataFrame
    response: str|None
    response_json_schema: BaseModel|None

    def is_answered(self) -> bool:
        return self.response is not None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["prompt"] = self.prompt
        base["response"] = self.response
        return base


@dataclass
class LotusQuery[GTT: (str,int)](Query[GTT, QueryParameters]):
    prompt: str
    prompt_df: DataFrame
    response: DataFrame|None

    def is_answered(self) -> bool:
        return self.response is not None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["prompt"] = self.prompt
        base["response"] = self.response
        return base


@dataclass
class PartitionedQuery[GTT: (str,int)](Query[GTT, PartitionedQueryParameters]):
    prompt_beg: str
    prompt_end: str
    raw_df: DataFrame
    sub_queries: list[DirectQuery]
    remaining_keys: list[GTT]
    response_json_schema: BaseModel|None

    def is_answered(self) -> bool:
        if self.sub_queries:
            return all(sq.is_answered() for sq in self.sub_queries)
        else:
            return False

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["prompt"] = self.prompt_beg + "\n{DATAFRAME}\n" + self.prompt_end
        base["raw_df"] = self.raw_df.to_string(index=False)
        if self.remaining_keys:
            base["remaining_keys"] = self.remaining_keys
        else:
            del base["remaining_keys"]
        del base["sub_queries"]
        base["sub_queries_parsed_response"] = [subq.parsed_response for subq in self.sub_queries]
        return base

@dataclass
class Test[GTT: (str,int), T_TestParameters: TestParameters](ABC):
    """
    Represents a single test, used to build queries.
    """
    family: str # the family this test belongs to (e.g. "movies", "molecules")
    name_path: str # the path-safe name of the test
    run_type: RunType
    run_folder: Path # /data/{direct | lotus}
    json_schema: BaseModel = None
    named_index_col: str = None
    queries: list[Query[GTT, QueryParameters]] = None # The list of queries generated for this test.
    prepared_queries_file_name: str = "prepared_queries.pkl" # Path to the file where prepared queries are stored.
    parameters: T_TestParameters = None
    evaluations: Evaluations = None # Aggregated evaluations across all queries for this test.
    params_file_name: str = 'test_params.json'

    def __post_init__(self):
        if self.json_schema is None:
            self.json_schema = self.__class__.json_schema
        if self.named_index_col is None:
            self.named_index_col = self.__class__.named_index_col

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
        self.prepare_queries_for_direct()
        return
        method_name = 'prepare_queries_for_' + "direct"
        try:
            method = getattr(self, method_name)
            method()
        except AttributeError:
            raise NotImplementedError(
                f'Run mode {self.run_type} not implemented for this test (method {method_name} does not exists).')

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        raise NotImplementedError()

    def _anonymize_query_df(self, df: DataFrame) -> DataFrame:
        """returns the query DataFrame with fake names. If not overridden, returns the original DF."""
        return df

    def _select_query_target(self, current_seed, real_df: DataFrame, anon_df) -> tuple[GTT | None, GTT | None]:
        """return (real_target, anon_target)"""
        return None, None

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], GroundTruthScoreList]:
        """
        returns a tuple with the list of ground truth ids and the list of their relevance scores
        """
        raise NotImplementedError()

    def build_prompt_df(self, df: DataFrame) -> DataFrame:
        """
        Returns the cleaned version of the DF, ready for create_prompt. It's called right before creating the Query.
        By default, it returns the input.
        """
        return df

    def _create_prompt_direct(self, df: DataFrame, target: GTT | None, q_params: QueryParameters) -> str:
        """Should return the final prompt string. By default, it uses the strings produce by _create_prompt_partitioned, putting the dataframe in the middle.
        """
        if not self.parameters.enforce_json_schema:
            raise NotImplementedError("il caso senza json_schema non è più supportato")

        beginning, end = self._create_prompt_partitioned(df, target, q_params)
        return f"{beginning}\n{self.df_to_string_for_prompt(df, q_params.prompt_printing_mode)}\n\n{end}"

    def _create_prompt_lotus(self, df: DataFrame, target: GTT | None, q_params: QueryParameters) -> str:
        raise NotImplementedError("Lotus not implemented for this test")

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        """
        returns the beginning and the end of the prompt, with the dataframe ideally in the middle.
        The DataFrame should NOT be included in these strings.
        :param q_params:
        """
        raise NotImplementedError()

    def df_to_string_for_prompt(self, df: DataFrame, printing_mode: PrintingMode) -> str:
        match printing_mode:
            case 'plain':
                return df.to_string(index=False)
            case 'markdown':
                return df.to_markdown(index=False)

    def _build_query(self, q_id: int, ds_id: int, prompt_df: DataFrame, full_df: DataFrame, target: GTT | None, k: int,
                     gt_ids: list[GTT], gt_vals: GroundTruthScoreList, prompt_level: PromptLevel,
                     names_level: NamesLevel, partition_rate: int, n_elems: int, printing_mode: PrintingMode) -> Query:
        q_standard_params = QueryParameters(k=k, prompt_level=prompt_level, names_level=names_level, n_elems=n_elems, prompt_printing_mode=printing_mode)
        base_args = {'id':q_id,'ds_id':ds_id,'ground_truth':gt_ids,'ground_truth_scores':gt_vals}
        none_args = {'parsed_response': None,'evaluations': None,'parsing_failed': None,'tokens': None}
        match self.run_type:
            case RunType.DIRECT:
                return DirectQuery[GTT](
                    **base_args,
                    parameters=q_standard_params,
                    prompt=self._create_prompt_direct(prompt_df, target, q_standard_params),
                    prompt_df=prompt_df,
                    response_json_schema=self.json_schema if self.parameters.enforce_json_schema else None,
                    **none_args,
                    response=None,
                )
            case RunType.LOTUS:
                return LotusQuery[GTT](
                    **base_args,
                    parameters=q_standard_params,
                    prompt=self._create_prompt_lotus(prompt_df, target, q_standard_params),
                    prompt_df=prompt_df,
                    **none_args,
                    response=None,
                )
            case RunType.PARTITIONED:
                q_partitioned_params = PartitionedQueryParameters(k=k, prompt_level=prompt_level, names_level=names_level,
                                                        n_elems=n_elems, partition_rate=partition_rate,
                                                        prompt_printing_mode=printing_mode)
                prompt_beg, prompt_end = self._create_prompt_partitioned(prompt_df, target, q_partitioned_params)
                return PartitionedQuery[GTT](
                    **base_args,
                    parameters=q_partitioned_params,
                    prompt_beg=prompt_beg,
                    prompt_end=prompt_end,
                    raw_df=full_df,
                    sub_queries=[],
                    remaining_keys=prompt_df[self.named_index_col].tolist(),
                    response_json_schema=self.json_schema if self.parameters.enforce_json_schema else None,
                    **none_args,
                )

    def _init_query(self, seed: int, df: DataFrame, elem_per_query: int) -> tuple[tuple[DataFrame, DataFrame, GTT|None, list[GTT], GroundTruthScoreList],tuple[DataFrame, DataFrame, GTT|None, list[GTT], GroundTruthScoreList]]:
        sampled_df = self._sample_for_query(seed, df, elem_per_query)
        anon_sampled_df = self._anonymize_query_df(sampled_df)
        target, anon_target = self._select_query_target(seed, sampled_df, anon_sampled_df)
        gt_ids, gt_scores = self.build_ground_truth(sampled_df, target)
        anon_gt_ids, anon_gt_scores = self.build_ground_truth(anon_sampled_df, anon_target)
        prompt_df = self.build_prompt_df(sampled_df)
        anon_prompt_df = self.build_prompt_df(anon_sampled_df)
        return (sampled_df, prompt_df, target, gt_ids, gt_scores), (anon_sampled_df, anon_prompt_df, anon_target, anon_gt_ids, anon_gt_scores)

    def _prepare_queries(self, clean_df, t_parameters: TestParameters) -> list[Query[GTT, QueryParameters]]:
        queries = []
        current_seed = t_parameters.seed
        counter = 0
        ds_id = 0

        total_iters = (t_parameters.n_queries
                       * len(t_parameters.elems_per_query)
                       * len(t_parameters.kp)
                       * len(t_parameters.prompt_levels)
                       * len(t_parameters.names_levels)
                       * len(t_parameters.prompt_printing_modes)
                       * (len(t_parameters.setwise_partition_rate) if self.run_type == RunType.PARTITIONED else 1))
        with tqdm(total=total_iters, desc="Generating queries", unit='query', colour='green') as pbar:

            for _ in range(t_parameters.n_queries):
                for elem_per_query in t_parameters.elems_per_query:
                    if t_parameters.seed != 0:
                        random.seed(current_seed)

                    (real_full_df, real_prompt_df, real_target, real_gt_ids, real_gt_scores), (anon_full_df, anon_prompt_df, anon_target, anon_gt_ids, anon_gt_scores) = self._init_query(
                        current_seed, clean_df, elem_per_query)

                    for kp, name_mode, prompt_level, printing_mode in product(t_parameters.kp, t_parameters.names_levels, t_parameters.prompt_levels, t_parameters.prompt_printing_modes):
                        if kp >= 1:
                            k = int(kp)
                        else:
                            k = max(1, math.ceil(kp * elem_per_query))

                        match name_mode:
                            case NamesLevel.real:
                                full_df = real_full_df
                                prompt_df = real_prompt_df
                                target = real_target
                                gt_ids = real_gt_ids
                                gt_scores = real_gt_scores
                            case NamesLevel.fake:
                                full_df = anon_full_df
                                prompt_df = anon_prompt_df
                                target = anon_target
                                gt_ids = anon_gt_ids
                                gt_scores = anon_gt_scores

                        match self.run_type:
                            case RunType.PARTITIONED:
                                if target is not None:
                                    raise NotImplementedError("I test con un target non sono supportati in modalità partitioned")
                                for pr in t_parameters.setwise_partition_rate:
                                    query = self._build_query(q_id=counter, ds_id=ds_id, prompt_df=prompt_df,
                                                              full_df=full_df, target=target, k=k, gt_ids=gt_ids,
                                                              gt_vals=gt_scores, prompt_level=prompt_level,
                                                              names_level=name_mode, partition_rate=pr,
                                                              n_elems=elem_per_query, printing_mode=printing_mode)
                                    queries.append(query)
                                    counter += 1
                                    pbar.update(1)
                            case _:
                                query = self._build_query(q_id=counter, ds_id=ds_id, prompt_df=prompt_df,
                                                          full_df=full_df, target=target, k=k, gt_ids=gt_ids,
                                                          gt_vals=gt_scores, prompt_level=prompt_level,
                                                          names_level=name_mode, partition_rate=0,
                                                          n_elems=elem_per_query, printing_mode=printing_mode)
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
        self.queries = self._prepare_queries(clean_df, self.parameters)

    def _parse_direct_query_schema(self, query: DirectQuery[GTT]) -> None:
        if query.response is None:
            query.parsing_failed = True
            return
        if query.response_json_schema is None:
            raise ValueError("Si è cercato di fare parsing con schema su una query senza schema.")
        try:
            parsed_response = query.response_json_schema.model_validate_json(query.response)
            query.parsed_response = getattr(parsed_response, list(
                query.response_json_schema.model_fields.keys())[0])
            parsing_failed = False
        except ValidationError:
            parsing_failed = True
        query.parsing_failed = parsing_failed

    def _parse_direct_query_manual(self, query: DirectQuery[GTT]) -> None:
        raise NotImplementedError("Parsing of not structured output is not implemented.")

    def _parse_direct_query(self, query: DirectQuery[GTT]) -> None:
        if query.response is None:
            query.parsing_failed = True
            return
        if query.response_json_schema is not None:
            self._parse_direct_query_schema(query)
        else:
            self._parse_direct_query_manual(query)

    def _parse_lotus_query(self, query: LotusQuery[GTT]) -> None:
        if query.response is None:
            query.parsing_failed = True
            return
        try:
            # take the first column as the response list
            query.parsed_response = query.response[self.named_index_col].tolist()
            parsing_failed = False
        except IndexError:
            parsing_failed = True
        query.parsing_failed = parsing_failed

    def _parse_partitioned_query(self, query: PartitionedQuery[GTT]) -> None:
        return

    def parse_query(self, query: Query) -> None:
        match query:
            case DirectQuery():
                self._parse_direct_query(query)
            case LotusQuery():
                self._parse_lotus_query(query)
            case PartitionedQuery():
                self._parse_partitioned_query(query)
            case _:
                raise ValueError("Query not recognized")

    def select_next_partition(self, full_df: DataFrame, partition_rate: int, remaining_keys: list[GTT], current_top_keys: list[GTT]) -> tuple[DataFrame, list[GTT]]:
        def keys_to_rows(df: DataFrame, keys: list[GTT]) -> DataFrame:
            return df[df[self.named_index_col].isin(keys)]

        free_slots: int = partition_rate - len(current_top_keys)
        new_entries_keys: list[GTT] = remaining_keys[:free_slots]
        remaining_keys = remaining_keys[free_slots:]
        current_top_df = keys_to_rows(full_df, current_top_keys)
        new_entries_df = keys_to_rows(full_df, new_entries_keys)
        arena: DataFrame = pd.concat([current_top_df, new_entries_df], ignore_index=True)

        return arena, remaining_keys

    def evaluate_query(self, query: Query[GTT, QueryParameters]) -> Evaluations:
        """
        Evaluate a single query after submission.
        :param query: the query to evaluate
        :return: A dictionary (an Evaluations object: dict[Metric, Number]) with the evaluation metrics for the query.
        """
        failing_scores = Evaluations(kendall=0.0, kendall_k=0.0, ndcg_scores=0.0, ndcg_k=0.0, mare=query.parameters.k, mare_k=query.parameters.k, spearman=-1.0, spearman_k=-1.0, hallucination_rate=1, tokens=query.tokens if query.tokens is not None else 0)

        if query.parsing_failed or query.parsed_response is not None and len(query.parsed_response) < query.parameters.k:
            return failing_scores
        assert query.parsed_response is not None

        llm_answer_with_duplicates = query.parsed_response[:query.parameters.k]
        llm_answer = mark_duplicates(llm_answer_with_duplicates)
        gt_scores_normalized = normalize(query.ground_truth_scores)
        llm_scores_normalized: list[float] = [ gt_scores_normalized[query.ground_truth.index(elem)]
                                   if elem in query.ground_truth
                                   else 0.0
                                   for elem in llm_answer]

        return Evaluations(ndcg_scores=ndcg_k(llm_scores_normalized, gt_scores_normalized, query.parameters.k),
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
                           tokens=query.tokens or 0)

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