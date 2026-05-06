import re
from dataclasses import dataclass, field

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import Test, data_folder, download_csv, extract_pipe_sequence, TestParameters, PromptLevel, \
    DirectQuery, GroundTruthScoreList, QueryParameters, PartitionedQueryParameters

type GTT = str
index_name = "Name"

class MostSimilarPlanets(BaseModel):
    top_k: list[GTT] = Field(description="The ordered list of the top k most similar planets.")

class PlanetTestParameters(TestParameters):
    elems_per_query: list[int] = field(default_factory=lambda: [30,69])

def add_line(df: DataFrame) -> DataFrame:
    medie = df.mean(numeric_only=True)

    nuova_riga = medie.to_dict()
    nuova_riga['Type'] = 'M Warm Superterran'
    nuova_riga['Detection Method'] = 'Transit'

    nuova_riga[index_name] = 'VL 120 e'

    df.loc[len(df)] = nuova_riga
    return df

@dataclass
class esi(Test[GTT, PlanetTestParameters]):
    name: str = "ESI"
    name_short = "ESI"
    json_schema = MostSimilarPlanets
    named_index_col = "Planet"

    def _load_ds(self) -> DataFrame:
        folder = data_folder / self.family
        simplified_ds_path = folder / 'hwc_simplified.csv'
        full_ds_path = folder / 'hwc.csv'
        if not simplified_ds_path.exists():
            download_csv('https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc_table_all.csv', simplified_ds_path) # simplified DS
        if not full_ds_path.exists():
            download_csv("https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc.csv", full_ds_path) # full DS
        return pd.read_csv(simplified_ds_path)

    def _parse_query_manual(self, query: DirectQuery) -> bool:
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

    def _prepare_df(self, df: DataFrame) -> DataFrame:
        def clean_html_tags(text: str) -> str:
            # Replace <i>...</i> with its content
            text = re.sub(r"<i>(.*?)</i>", lambda m: m.group(1), text)
            # Replace <sub>...</sub> with its lowercase content
            text = re.sub(r"<sub>(.*?)</sub>", lambda m: m.group(1).lower(), text)
            # Replace <br> or <br/> with a space
            text = re.sub(r"<br\s*/?>", " ", text)

            return text

        df = add_line(df)

        df = df.rename(columns=clean_html_tags, copy=True)
        df = df.rename(columns={index_name: self.named_index_col})

        def find_contained_strings(strings):
            contained_pairs = []

            for i, s1 in enumerate(strings):
                for j, s2 in enumerate(strings):
                    if i == j:
                        continue
                    if s1 in s2:
                        contained_pairs.append((s1, s2))

            return contained_pairs
        if len(find_contained_strings(df[self.named_index_col].tolist())) > 0:
            print("Warning: some planet names contain other planet names. This can affect the output parsing.")

        return df

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        return df.sample(n=elem_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)

    def _anonymize_query_df(self, df: DataFrame) -> DataFrame:
        fake_selected_planets = df.copy()
        fake_selected_planets[self.named_index_col] = "Planet " + fake_selected_planets.index.astype(str)
        return fake_selected_planets

    def build_prompt_df(self, df: DataFrame) -> DataFrame:
        return df.drop(columns='ESI', inplace=False)

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], GroundTruthScoreList]:
        ground_truth_df = df.sort_values(by=['ESI'], ascending=False, inplace=False)
        return ground_truth_df[self.named_index_col].tolist(), ground_truth_df['ESI'].tolist()

    def _create_prompt_lotus(self, df: DataFrame, target: GTT | None, q_params: QueryParameters) -> str:
        # shadowing voluto
        index_name = "Planet"

        lotus_attributes_without_index = {", ".join([f"{{{c}}}" for c in df.columns if c != index_name])}
        match q_params.prompt_level:
            case PromptLevel.generic:
                prompt = f"""Return the {{{index_name}}} most similar to Earth, considering only the provided attributes ({lotus_attributes_without_index})."""
            case PromptLevel.instruct:
                prompt = f"""Return the {{{index_name}}} that is most similar to Earth, using the Earth Similarity Index (ESI) as the only criterion for similarity, considering only the provided attributes ({lotus_attributes_without_index})."""
            case PromptLevel.formula:
                prompt = \
    f"""Provide the {{{index_name}}} that is most similar to Earth, using only the ESI (Earth Similarity Index) score, considering only the provided attributes ({lotus_attributes_without_index})
The ESI formula is explained below:

The formula takes as input a planet's radius (R) and solar flux (S).
it is computed as follows:
1. compute the solar flux ratio (SR): SR = ( (S - 1) / (S + 1) )^2
2. compute the radius ratio (RR): RR = ( (R - 1) / (R + 1) )^2
3. compute the final score: score = 1 - sqrt( 0.5 * (SR + RR) )"""

        return prompt

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given a dataset of planets, with various attributes:"
        output = "Your output must contain only the required list of planets."

        match q_params.prompt_level:
            case PromptLevel.generic:
                instruction = \
                    f"""Return the top {q_params.k} planets that are most similar to Earth."""
            case PromptLevel.instruct:
                instruction = \
                    f"""Return the top {q_params.k} planets that are most similar to Earth, using the Earth Similarity Index (ESI) as the only criterion for similarity."""
            case PromptLevel.formula:
                instruction = \
                    f"""Provide the top {q_params.k} planets that are most similar to Earth, using only the ESI (Earth Similarity Index) score.
The ESI score formula is explained below:

The formula takes as input a planet's radius (R) and solar flux (S). It is computed as follows:
    1. compute the solar flux ratio (SR): SR = ( (S - 1) / (S + 1) )^2
    2. compute the radius ratio (RR): RR = ( (R - 1) / (R + 1) )^2
    3. compute the final score: score = 1 - sqrt( 0.5 * (SR + RR) )"""

        return job, f"{instruction}\n\n{output}"
