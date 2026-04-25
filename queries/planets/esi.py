import re
from dataclasses import dataclass, field

import math
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from experiments.run_type import RunType
from queries.test import Test, data_folder, Query, Evaluations, download_csv, \
    extract_pipe_sequence, mark_duplicates, QueryParameters, TestParameters, PromptLevel, NamesLevel
from queries.metrics import ndcg_k, hallucination_rate, mare_k, spearman_rho_k, kendall_tau_k, standard_ndcg_scoring

index_name = "Name"

def compute_ground_truth(df: DataFrame, top_k: int = None) -> DataFrame:
    sorted_df = df.sort_values(by=['ESI'], ascending=False, inplace=False)
    return sorted_df if top_k is None else sorted_df.head(top_k)


class MostSimilarPlanets(BaseModel):
    top_k: list[str] = Field(description="The ordered list of the top k most similar planets.")

def create_prompt(df: DataFrame, prompt_level: PromptLevel, top_k: int, json_schema: bool, run_type: RunType) -> str:
    # shadowing voluto
    index_name = "Planet"
    if not json_schema:
        raise NotImplementedError("il caso senza json_schema non è più supportato.")

    job = f"You are given a dataset of planets, with various attributes:\n{df.to_string(index=False)}"
    output = "Your output must contain only the required list of planets."
    lotus_attributes_without_index = {", ".join([f"{{{c}}}" for c in df.columns if c != index_name])}


    match run_type:
        case RunType.LOTUS:
            match prompt_level:
                case prompt_level.generic:
                    prompt = f"""Return the {{{index_name}}} most similar to Earth, considering only the provided attributes ({lotus_attributes_without_index})."""
                case prompt_level.instruct:
                    prompt = f"""Return the {{{index_name}}} that is most similar to Earth, using the Earth Similarity Index (ESI) as the only criterion for similarity, considering only the provided attributes ({lotus_attributes_without_index})."""
                case prompt_level.formula:
                    prompt = \
                        f"""Provide the {{{index_name}}} that is most similar to Earth, using only the ESI (Earth Similarity Index) score, considering only the provided attributes ({lotus_attributes_without_index})
The ESI formula is explained below:

The formula takes as input a planet's radius (R) and solar flux (S).
it is computed as follows:
1. compute the solar flux ratio (SR): SR = ( (S - 1) / (S + 1) )^2
2. compute the radius ratio (RR): RR = ( (R - 1) / (R + 1) )^2
3. compute the final score: score = 1 - sqrt( 0.5 * (SR + RR) )"""
        case RunType.DIRECT:
            match prompt_level:
                case PromptLevel.generic:
                    instruction = \
        f"""Return the top {top_k} planets that are most similar to Earth."""
                case PromptLevel.instruct:
                    instruction = \
        f"""Return the top {top_k} planets that are most similar to Earth, using the Earth Similarity Index (ESI) as the only criterion for similarity."""
                case PromptLevel.formula:
                    instruction = \
        f"""Provide a ranked list of the top {top_k} planets that are most similar to Earth, using only the ESI (Earth Similarity Index) score.
The ESI formula is explained below:

The formula takes as input a planet's radius (R) and solar flux (S).
it is computed as follows:
1. compute the solar flux ratio (SR): SR = ( (S - 1) / (S + 1) )^2
2. compute the radius ratio (RR): RR = ( (R - 1) / (R + 1) )^2
3. compute the final score: score = 1 - sqrt( 0.5 * (SR + RR) )"""
            prompt = f"{job}\n\n{instruction}\n\n{output}"

    return prompt

@dataclass
class PlanetTestParameters(TestParameters):
    planets_per_query: list[int] = field(default_factory=lambda: [50]) #70 tot

@dataclass
class esi(Test[PlanetTestParameters]):
    name: str = "ESI"
    name_short = "ESI"
    json_schema = MostSimilarPlanets
    simplified_df: DataFrame = None
    clean_df: DataFrame = None
    # DA USARE SOLO DA CREATE_PROMPT IN POI
    named_index_col = "Planet"

    def load_csvs(self) -> None:
        folder = data_folder / self.family
        simplified_ds_path = folder / 'hwc_simplified.csv'
        full_ds_path = folder / 'hwc.csv'
        if not simplified_ds_path.exists():
            download_csv('https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc_table_all.csv', simplified_ds_path) # simplified DS
        if not full_ds_path.exists():
            download_csv("https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc.csv", full_ds_path) # full DS
        self.simplified_df = pd.read_csv(simplified_ds_path)

    def _parse_query_manual(self, query: Query) -> bool:
        matched_lists = extract_pipe_sequence(query.response)
        if len(matched_lists) == 0:
            print(f"Cannot evaluate query {query.id}: no pipe-separated list found in the response.")
            return True
        elif len(matched_lists) > 1:
            matched_lists = [lst for lst in matched_lists if len(lst) == query.parameters.k]
            if len(matched_lists) == 0:
                print(f"Cannot evaluate query {query.id}: no valid pipe-separated list found in the response.")
                return True
            elif len(matched_lists) > 1:
                print(f"Warning for query {query.id}: multiple pipe-separated lists found in the response. Using the last one.")
        top_k_list = matched_lists[-1] # use the last matched list
        top_k_list = [s.replace("*", "") for s in top_k_list] # remove possible asterisks

        query.parsed_response = top_k_list
        return False

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
                        prompt_df = q_df.drop(columns='ESI', inplace=False).rename(columns={index_name: 'Planet'})

                        for prompt_level in self.parameters.prompt_levels:
                            query = Query(
                                id=counter,
                                ds_id=ds_id,
                                prompt=create_prompt(prompt_df, prompt_level, k, self.parameters.enforce_json_schema,
                                                     self.run_type),
                                prompt_df=prompt_df if self.run_type==RunType.LOTUS else None,
                                parameters=QueryParameters(k=k, prompt_level=prompt_level, names_level=planet_name_mod,
                                                           n_elems=p_per_query),
                                ground_truth=ground_truth_df['Name'].tolist(),
                                ground_truth_scores=ground_truth_df['ESI'].tolist(),
                                response_json_schema=self.json_schema.model_json_schema() if self.parameters.enforce_json_schema else None,
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
