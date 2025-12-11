import json
import os
import re
from dataclasses import dataclass, field
from enum import auto
from json import JSONDecodeError

import math
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import Test, data_folder, Query, Evaluations, download_csv, Metric, extract_json, extract_list, \
    hallucination_rate, extract_pipe_sequence, mare_k, spearman_rho_k, ndcg_k


def compute_ground_truth(df: DataFrame, top_k: int = None) -> DataFrame:
    sorted_df = df.sort_values(by=['ESI'], ascending=False, inplace=False)
    return sorted_df if top_k is None else sorted_df.head(top_k)


class MostSimilarPlanets(BaseModel):
    top_k: list[str] = Field(description="The ordered list of top k most similar planets.")

class PlanetScore(BaseModel):
    planet_name: str = Field(description="The name of the planet.")
    esi: float = Field(description="The Earth Similarity Index score of the planet.")

class MostSimilarPlanetsScore(BaseModel):
    top_k: list[PlanetScore] = Field(description="The ordered list of top k most similar planets.")

def create_prompt(df: DataFrame, prompt_level: str, top_k: int, json_schema: bool) -> str:
    if json_schema:
        if prompt_level == 'generic':
            output_string = f"""Your output MUST contain only a sorted list of the most similar planets (represented by their 'Name')
from most to least similar, as per the following JSON schema:
{json.dumps(MostSimilarPlanets.model_json_schema())}"""
        else:
            output_string = f"""Your output MUST contain only a sorted list of the most similar planets (represented by their 'Name' and
 the computed similarity score) from most to least similar, as per the following JSON schema:
{json.dumps(MostSimilarPlanetsScore.model_json_schema())}"""
    else:
        output_string = f"""Your output MUST contain only a sorted list of the most similar planets (represented by their 'Name'),
from most to least similar, separated by the character '|'."""

    if prompt_level == 'generic':
        prompt = \
f"""
Your task is to identify the most similar planets to Earth based on the provided dataset:
{df.to_string(index=False)}

The dataset includes various planetary characteristics. Please analyze the data and provide a ranked list of
the top {top_k} planets that are most similar to Earth.

{output_string}
"""
    elif prompt_level == 'esi_instruct':
        prompt = \
f"""
Your task is to identify the most similar planets to Earth based on the provided dataset:
{df.to_string(index=False)}

Please provide a ranked list of the top {top_k} planets that are most similar to Earth,
using the Earth Similarity Index (ESI) as THE ONLY criterion for similarity.

{output_string}
"""
    elif prompt_level == 'esi_formula':
        prompt = \
f"""
Your task is to identify the most similar planets to Earth based on the provided dataset:
{df.to_string(index=False)}

Please provide a ranked list of the top {top_k} planets that are most similar to Earth,
using the following formula as THE ONLY criterion for similarity:

The formula takes as input a planet's radius (R) and solar flux (S).
it is computed as follows:
1. compute the solar flux ratio (SR): SF = ( (S - 1) / (S + 1) )^2
2. compute the radius ratio (RR): RR = ( (R - 1) / (R + 1) )^2
3. compute the final score: score = 1 - sqrt( 0.5 * (SR + RR) )

{output_string}
"""
    else:
        raise ValueError(f"Unknown prompt_level: {prompt_level}")
    return prompt

class PlanetMetrics(Metric):
    NDCG_SCORES = auto() # NDCG considerando i punteggi reali, indipendentemente da k
    NDCG_K = auto() # NDCG considerando k-i se l'elemento i-esimo dilla lista predetta è presente nella k-ground truth (indipendentemente dalla sua posizione)
    MARE = auto() # MARE considerando tutti gli elementi dalla ground truth
    MARE_K = auto() # MARE considerando solo i primi k elementi della ground truth
    SPEARMAN = auto()
    SPEARMAN_K = auto()
    HALLUCINATION_RATE = auto()

@dataclass
class PlanetQuery(Query):
    parsed_response: str
    prompt_level: str
    plant_names_mod: str
    ground_truth: list[dict]

@dataclass
class esi(Test):

    name: str = "ESI"
    simplified_df: DataFrame = None
    clean_df: DataFrame = None
    @dataclass
    class Params(Test.Params):
        prompt_levels: list[str] = field(default_factory=lambda: ['generic']) # 'generic', 'esi_instruct', 'esi_formula'
        plant_names_mods: list[str] = field(default_factory=lambda: ['real']) # 'real', 'fake'
        n_queries: int = 10
        planets_per_query: int = 50
        top_k: int = 10
        enforce_json_schema: bool = False
    params: Params = field(default_factory=Params)
        

    def load_csvs(self) -> None:
        folder = data_folder / self.family
        simplified_ds_path = folder / 'hwc_simplified.csv'
        full_ds_path = folder / 'hwc.csv'
        if not simplified_ds_path.exists():
            download_csv('https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc_table_all.csv', simplified_ds_path) # simplified DS
        if not full_ds_path.exists():
            download_csv("https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc.csv", full_ds_path) # full DS
        self.simplified_df = pd.read_csv(simplified_ds_path)

    def evaluate_query(self, query: PlanetQuery) -> Evaluations:
        failing_scores = Evaluations(ndcg_scores=0.0, ndcg_k=0.0, mare=self.params.top_k, mare_k=self.params.top_k, spearman=-1.0, spearman_k=-1.0, hallucination_rate=self.params.top_k)

        if query.response_json_schema:
            try:
                json_response = extract_json(query.response)
                top_k_list = json_response['top_k']
            except (JSONDecodeError, KeyError, TypeError):
                try:
                    json_response = {'top_k': extract_list(query.response)}
                    top_k_list = json_response['top_k']
                except (JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Cannot evaluate query: {e}")
                    query.parsing_failed = True
                    return failing_scores
        else:
            matched_lists = extract_pipe_sequence(query.response)
            if len(matched_lists) == 0:
                print(f"Cannot evaluate query {query.id}: no pipe-separated list found in the response.")
                query.parsing_failed = True
                return failing_scores
            elif len(matched_lists) > 1:
                matched_lists = [lst for lst in matched_lists if len(lst) == self.params.top_k]
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
        llm_names: list[str] = top_k_list
        for i, llm_name in enumerate(top_k_list):
            for gt_name in query.ground_truth:
                if gt_name['planet_name'].lower() == re.sub(r"^\s*\d+\.\s*", "", llm_name.lower()):
                    llm_names[i] = gt_name['planet_name']
                    break

        ground_truth_names: list[str] = [item['planet_name'] for item in query.ground_truth]
        ground_truth_scores: list[float] = [item['esi'] for item in query.ground_truth]
        llm_scores: list[float] = []
        for i, pred_planet in enumerate(llm_names):
            score = 0.0
            for gt_name in query.ground_truth:
                if gt_name['planet_name'] == pred_planet:
                    score = gt_name['esi']
                    break
            llm_scores.append(score)

        return Evaluations(ndcg_scores=ndcg_k(llm_scores, ground_truth_scores, self.params.top_k),
                           ndcg_k=ndcg_k(
            [self.params.top_k - i if p in ground_truth_names[:self.params.top_k] else 0 for i, p in enumerate(llm_names)],
            k=self.params.top_k),
                           mare=mare_k(llm_names, ground_truth_names),
                           mare_k=mare_k(llm_names, ground_truth_names, self.params.top_k),
                           spearman=spearman_rho_k(llm_names, ground_truth_names),
                           spearman_k=spearman_rho_k(llm_names, ground_truth_names, self.params.top_k),
                           hallucination_rate=hallucination_rate(llm_names, ground_truth_names))

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
        if math.comb(len(self.clean_df), self.params.planets_per_query) < self.params.n_queries:
            raise ValueError("Not enough unique combinations of planets to generate the requested number of queries.")

        self.queries = []
        current_seed = self.params.seed
        counter = 0

        for _ in range(self.params.n_queries):
            selected_planets = self.clean_df.sample(n=self.params.planets_per_query, replace=False, random_state=current_seed if self.params.seed != 0 else None)

            for planet_name_mod in self.params.plant_names_mods:
                if planet_name_mod == 'real':
                    q_df = selected_planets
                elif planet_name_mod == 'fake':
                    fake_selected_planets = selected_planets.copy()
                    fake_selected_planets['Name'] = "Planet " + fake_selected_planets.index.astype(str)
                    q_df = fake_selected_planets
                else:
                    raise ValueError(f"Unknown planet_name_mod: {planet_name_mod}")
                ground_truth_df = compute_ground_truth(q_df)
                prompt_df = q_df.drop(columns='ESI', inplace=False)
                ground_truth = [{'planet_name': row['Name'],
                                 'esi': row['ESI']} for _, row
                                in ground_truth_df.iterrows()]
                for prompt_level in self.params.prompt_levels:
                    if prompt_level == 'generic':
                        response_schema = MostSimilarPlanets.model_json_schema()
                    elif prompt_level in ['esi_instruct', 'esi_formula']:
                        response_schema = MostSimilarPlanetsScore.model_json_schema()
                    else:
                        raise ValueError(f"Unknown prompt_level: {prompt_level}")
                    query = PlanetQuery(
                        id=counter,
                        prompt=create_prompt(prompt_df, prompt_level, self.params.top_k, self.params.enforce_json_schema),
                        prompt_level=prompt_level,
                        plant_names_mod=planet_name_mod,
                        ground_truth=ground_truth,
                        response=None,
                        evaluations=Evaluations(),
                        response_json_schema=response_schema if self.params.enforce_json_schema else None,
                        parsing_failed=None,
                        parsed_response=None
                    )
                    self.queries.append(query)
                    counter += 1

            current_seed = current_seed + 1

    def prepare_queries_for_direct(self) -> None:
        print('loading dataset...')
        self.load_csvs()
        print('preparing DF...')
        self.prepare_df()
        print('initializing queries...')
        self.init_queries()
