import random
from dataclasses import dataclass

from pandas import DataFrame
import pandas as pd
from pydantic import BaseModel, Field

from queries.test import Test, data_folder, TestParameters, PromptLevel, QueryParameters, PartitionedQueryParameters, \
    CompletenessLevel

type GTT = str
named_index_col = 'City'
score_col = 'Overall Rating'
wights = {
    'Stability': 0.25,
    'Healthcare': 0.20,
    'Culture & Environment': 0.25,
    'Education': 0.10,
    'Infrastructure': 0.20
}
scoring_cols: list[str] = list(wights.keys())

class BestCities(BaseModel):
    most_similar_cities: list[GTT]  = Field(description="Ordered list of the k cities, as required.")

def closest_cities(df: DataFrame, target_city: GTT) -> DataFrame:
    mask = df[named_index_col] == target_city
    if not mask.any():
        raise ValueError(f"City not found: {target_city!r}")

    target_score = df.loc[mask, score_col].iloc[0]

    # Compute absolute distance and take the k smallest excluding the target
    out = (
        df.assign(diff=(df[score_col] - target_score).abs())
        .loc[~mask]
        .sort_values("diff", ascending=True)
        .loc[:, [named_index_col, score_col, "diff"]]
    )
    return out

@dataclass
class global_liveability(Test[GTT, TestParameters]):
    name: str = "Global Liveability Index"
    name_short: str = "GLI"
    json_schema = BestCities
    named_index_col = named_index_col
    prompt_scoring_cols = scoring_cols

    def _load_ds(self) -> DataFrame:
        return pd.read_excel(data_folder / 'cities' / 'global_liveability.xlsx', sheet_name='Foglio2')

    def _prepare_df(self, df: DataFrame) -> DataFrame:
        df = df.drop(columns='Rank')
        df = df.drop_duplicates(subset=score_col, keep='first')
        return df

    def _create_prompt_lotus(self, df: DataFrame, target: GTT | None, q_params: QueryParameters) -> str:
        raise NotImplementedError("il prompt di lotus non è stato aggiornato dopo la modifica del test")
        formula_string = f"GLI = ({" + ".join([f"'{c}' * {str(w)}" for c, w in wights.items()])})"
        # questa stringa fornisce i nomi delle colonne con il formato che LOTUS si aspetta
        target_attributes_string = ", ".join([f"{{{col}}}: {val}" for col, val in df[df[named_index_col] == target].iloc[0].items()])

        match q_params.prompt_level:
            case PromptLevel.instruct:
                prompt = f"Return the most similar {named_index_col} to '{target}', whose attributes are:\n{target_attributes_string}\nbased only on the Global Liveability Index (GLI)."
            case PromptLevel.formula:
                prompt = f"Using only the Global Liveability Index (GLI), which can be computed with the formula {formula_string}, return the most similar {named_index_col} to '{target}' (whose attributes are: {target_attributes_string})."
            case PromptLevel.generic:
                prompt = f"Return the most similar {named_index_col} to '{target}', based only on the provided attributes ({", ".join([f"{{{c}}}" for c in df.columns.tolist()])})."

        return prompt

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given a dataset of cities, with various attributes:"
        output = "Your output must contain only the required list of cities."
        formula_string = f"GLI = ({" + ".join([f"'{c}' * {str(w)}" for c, w in wights.items()])})"

        match q_params.prompt_level:
            case PromptLevel.instruct:
                instruct = f"Return the {q_params.k} cities with the highest Global Liveability Index (GLI), using only the provided data."
            case PromptLevel.formula:
                instruct = f"Return the {q_params.k} cities with the highest Global Liveability Index (GLI). This index can be computed with the formula {formula_string}."
            case PromptLevel.generic:
                instruct = f"Return the {q_params.k} best cities, based only on the provided data."

        return job, f"{instruct}\n\n{output}"

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        sampled_cities: list[str] = random.sample(k=elem_per_query, population=df[named_index_col].tolist())
        return df[df[named_index_col].isin(sampled_cities)].sample(frac=1, random_state=current_seed)

    def _anonymize_query_df(self, df: DataFrame) -> DataFrame:
        df = df.drop(columns='Country')
        fake_name_mapping = {c: "City " + str(i) for i, c in enumerate(df[named_index_col].tolist(), start=1)}
        df[named_index_col] = df[named_index_col].map(fake_name_mapping)
        return df

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], list[float]]:
        gt_df = df.copy()
        gt_df = gt_df.sort_values(by=score_col, ascending=False)
        return gt_df[named_index_col].tolist(), gt_df[score_col].tolist()

    def build_prompt_df(self, df: DataFrame) -> DataFrame:
        return df.drop(columns=score_col)
