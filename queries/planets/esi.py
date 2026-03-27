import re
from dataclasses import dataclass, field
from json import JSONDecodeError

import math
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from queries.test import Test, data_folder, Query, Evaluations, download_csv, extract_json, extract_list, \
    extract_pipe_sequence, mark_duplicates, QueryParameters, TestParameters, PromptLevel, NamesLevel
from queries.metrics import ndcg_k, hallucination_rate, mare_k, spearman_rho_k, kendall_tau_k, standard_ndcg_scoring


def compute_ground_truth(df: DataFrame, top_k: int = None) -> DataFrame:
    sorted_df = df.sort_values(by=['ESI'], ascending=False, inplace=False)
    return sorted_df if top_k is None else sorted_df.head(top_k)


class MostSimilarPlanets(BaseModel):
    top_k: list[str] = Field(description="The ordered list of the top k most similar planets.")

class PlanetScore(BaseModel):
    planet_name: str = Field(description="The name of the planet.")
    esi: float = Field(description="The Earth Similarity Index score of the planet.")

class MostSimilarPlanetsScore(BaseModel):
    top_k: list[PlanetScore] = Field(description="The ordered list of top k most similar planets.")

def create_prompt(df: DataFrame, prompt_level: PromptLevel, top_k: int, json_schema: bool) -> str:
    if not json_schema:
        raise NotImplementedError("il caso senza json_schema non è più supportato.")

    job = f"You are given a dataset of planets, with various attributes:\n{df.to_string(index=False)}"
    output = "Your output must contain only the required list of planets."

    match prompt_level:
        case PromptLevel.generic:
            prompt = \
f"""{job}

Return the top {top_k} planets that are most similar to Earth.\n{output}"""
        case PromptLevel.instruct:
            prompt = \
f"""{job}

Return the top {top_k} planets that are most similar to Earth, using the Earth Similarity Index (ESI) as the only criterion for similarity.\n{output}"""
        case PromptLevel.formula:
            prompt = \
f"""{job}

Provide a ranked list of the top {top_k} planets that are most similar to Earth, using only the ESI (Earth Similarity Index) score.
The ESI formula is explained below:

The formula takes as input a planet's radius (R) and solar flux (S).
it is computed as follows:
1. compute the solar flux ratio (SR): SR = ( (S - 1) / (S + 1) )^2
2. compute the radius ratio (RR): RR = ( (R - 1) / (R + 1) )^2
3. compute the final score: score = 1 - sqrt( 0.5 * (SR + RR) )

{output}"""
    return prompt

@dataclass
class PlanetEvaluations(Evaluations):
    ndcg_scores: float
    ndcg_k: float
    mare: float
    mare_k: float
    spearman: float
    spearman_k: float
    kendall: float
    kendall_k: float
    hallucination_rate: float

@dataclass
class PlanetQueryParameters(QueryParameters):
    k: int
    prompt_level: PromptLevel
    names_level: NamesLevel
    n_elems: int

@dataclass
class PlanetQuery(Query[PlanetQueryParameters, PlanetEvaluations]):
    parsed_response: list[str]
    ground_truth: list[dict]

@dataclass
class PlanetTestParameters(TestParameters):
    kp: list[float] = field(default_factory=lambda: [0.05, 0.1])
    prompt_levels: tuple[PromptLevel, ...] = tuple(PromptLevel) # all types
    names_levels: tuple[NamesLevel, ...] = tuple(NamesLevel) # all types
    n_queries: int = 10
    planets_per_query: list[int] = field(default_factory=lambda: [50]) #70 tot
    enforce_json_schema: bool = True

@dataclass
class esi(Test[PlanetQuery, PlanetTestParameters, PlanetEvaluations]):
    name: str = "ESI"
    simplified_df: DataFrame = None
    clean_df: DataFrame = None

    def load_csvs(self) -> None:
        folder = data_folder / self.family
        simplified_ds_path = folder / 'hwc_simplified.csv'
        full_ds_path = folder / 'hwc.csv'
        if not simplified_ds_path.exists():
            download_csv('https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc_table_all.csv', simplified_ds_path) # simplified DS
        if not full_ds_path.exists():
            download_csv("https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc.csv", full_ds_path) # full DS
        self.simplified_df = pd.read_csv(simplified_ds_path)

    def _parse_query(self, query: PlanetQuery) -> bool:
        if query.response is None or query.response == '':
            return False
        if query.response_json_schema is not None:
            try:
                parsed_response = MostSimilarPlanets.model_validate_json(query.response)
                query.parsed_response = parsed_response.top_k
                return True
            except ValidationError:
                pass

        return False

    def evaluate_query(self, query: PlanetQuery) -> PlanetEvaluations:
        failing_scores = PlanetEvaluations(kendall=0.0, kendall_k=0.0, ndcg_scores=0.0, ndcg_k=0.0, mare=query.parameters.k, mare_k=query.parameters.k, spearman=-1.0, spearman_k=-1.0, hallucination_rate=1)

        if query.response_json_schema:
            query.parsing_failed = not self._parse_query(query)
            if query.parsing_failed or len(query.parsed_response) < query.parameters.k:
                return failing_scores
        else:
            matched_lists = extract_pipe_sequence(query.response)
            if len(matched_lists) == 0:
                print(f"Cannot evaluate query {query.id}: no pipe-separated list found in the response.")
                query.parsing_failed = True
                return failing_scores
            elif len(matched_lists) > 1:
                matched_lists = [lst for lst in matched_lists if len(lst) == query.parameters.k]
                if len(matched_lists) == 0:
                    print(f"Cannot evaluate query {query.id}: no valid pipe-separated list found in the response.")
                    query.parsing_failed = True
                    return failing_scores
                elif len(matched_lists) > 1:
                    print(f"Warning for query {query.id}: multiple pipe-separated lists found in the response. Using the last one.")
            top_k_list = matched_lists[-1] # use the last matched list
            top_k_list = [s.replace("*", "") for s in top_k_list] # remove possible asterisks

            query.parsing_failed = False
            query.parsed_response = top_k_list

        # removes possible prepended numbers
        llm_names: list[str] = query.parsed_response[:query.parameters.k]
        for i, llm_name in enumerate(query.parsed_response[:query.parameters.k]):
            for gt_name in query.ground_truth:
                if gt_name['planet_name'].lower() == re.sub(r"^\s*\d+\.\s*", "", llm_name.lower()):
                    llm_names[i] = gt_name['planet_name']
                    break

        ground_truth_names: list[str] = [item['planet_name'] for item in query.ground_truth]
        ground_truth_scores: list[float] = [item['esi'] for item in query.ground_truth]
        llm_names_with_duplicates = llm_names
        llm_names = mark_duplicates(llm_names, ground_truth_names[:query.parameters.k])
        llm_scores: list[float] = []
        for i, pred_planet in enumerate(llm_names):
            score = 0.0
            for gt_name in query.ground_truth:
                if gt_name['planet_name'] == pred_planet:
                    score = gt_name['esi']
                    break
            llm_scores.append(score)

        return PlanetEvaluations(ndcg_scores=ndcg_k(llm_scores, ground_truth_scores, query.parameters.k),
                           ndcg_k=ndcg_k(
            relevance_scores=standard_ndcg_scoring(llm_names, ground_truth_names, query.parameters.k),
            k=query.parameters.k),
                           mare=mare_k(llm_names, ground_truth_names),
                           mare_k=mare_k(llm_names, ground_truth_names, query.parameters.k),
                           kendall = kendall_tau_k(llm_names, ground_truth_names),
                           kendall_k = kendall_tau_k(llm_names, ground_truth_names, query.parameters.k),
                           spearman=spearman_rho_k(llm_names, ground_truth_names),
                           spearman_k=spearman_rho_k(llm_names, ground_truth_names, query.parameters.k),
                           hallucination_rate=hallucination_rate(llm_names_with_duplicates, ground_truth_names))

    def prepare_df(self) -> None:
        def clean_html_tags(text: str) -> str:
            # Replace <i>...</i> with its content
            text = re.sub(r"<i>(.*?)</i>", lambda m: m.group(1), text)
            # Replace <sub>...</sub> with its lowercase content
            text = re.sub(r"<sub>(.*?)</sub>", lambda m: m.group(1).lower(), text)
            # Replace <br> or <br/> with a space
            text = re.sub(r"<br\s*/?>", " ", text)

            return text

        df = self.simplified_df.rename(columns=clean_html_tags, copy=True)

        def find_contained_strings(strings):
            contained_pairs = []

            for i, s1 in enumerate(strings):
                for j, s2 in enumerate(strings):
                    if i == j:
                        continue
                    if s1 in s2:
                        contained_pairs.append((s1, s2))

            return contained_pairs
        if len(find_contained_strings(df['Name'].tolist())) > 0:
            print("Warning: some planet names contain other planet names. This can affect the output parsing.")

        self.clean_df = df

    def init_queries(self) -> None:
        for p_per_query in self.parameters.planets_per_query:
            if math.comb(len(self.clean_df), p_per_query) < self.parameters.n_queries:
                raise ValueError(f"Not enough unique combinations of planets to generate the requested number of queries (planets per query: {p_per_query}).")

        self.queries = []
        current_seed = self.parameters.seed
        counter = 0
        ds_id = 0
        pbar = tqdm(total=self.parameters.n_queries*len(self.parameters.planets_per_query)*len(self.parameters.kp)*len(self.parameters.prompt_levels)*len(self.parameters.names_levels),
                    desc="Generating queries",
                    unit="query",
                    colour='green')

        for _ in range(self.parameters.n_queries):
            for p_per_query in self.parameters.planets_per_query:
                selected_planets = self.clean_df.sample(n=p_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)

                for kp in self.parameters.kp:
                    k = max(1, math.ceil(kp * p_per_query))

                    for planet_name_mod in self.parameters.names_levels:
                        match planet_name_mod:
                            case NamesLevel.real:
                                q_df = selected_planets
                            case NamesLevel.fake:
                                fake_selected_planets = selected_planets.copy()
                                fake_selected_planets['Name'] = "Planet " + fake_selected_planets.index.astype(str)
                                q_df = fake_selected_planets
                        ground_truth_df = compute_ground_truth(q_df)
                        prompt_df = q_df.drop(columns='ESI', inplace=False)
                        ground_truth = [{'planet_name': row['Name'],
                                         'esi': row['ESI']} for _, row
                                        in ground_truth_df.iterrows()]

                        for prompt_level in self.parameters.prompt_levels:
                            match prompt_level:
                                case PromptLevel.generic:
                                    response_schema = MostSimilarPlanets.model_json_schema()
                                case PromptLevel.instruct | PromptLevel.formula:
                                    response_schema = MostSimilarPlanetsScore.model_json_schema()
                            query = PlanetQuery(
                                id=counter,
                                ds_id=ds_id,
                                prompt=create_prompt(prompt_df, prompt_level, k, self.parameters.enforce_json_schema),
                                parameters=PlanetQueryParameters(k=k, prompt_level=prompt_level, names_level=planet_name_mod, n_elems=p_per_query),
                                ground_truth=ground_truth,
                                response=None,
                                evaluations=None,
                                response_json_schema=MostSimilarPlanets.model_json_schema() if self.parameters.enforce_json_schema else None,
                                parsing_failed=None,
                                parsed_response=None
                            )
                            self.queries.append(query)
                            counter += 1
                            pbar.update(1)

                current_seed = current_seed + 1
                ds_id += 1

        pbar.close()

    def prepare_queries_for_direct(self) -> None:
        print('loading dataset...')
        self.load_csvs()
        print('preparing DF...')
        self.prepare_df()
        print('initializing queries...')
        self.init_queries()
