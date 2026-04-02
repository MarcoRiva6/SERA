import math
import random
from dataclasses import dataclass, field

from pandas import DataFrame
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from queries.test import Test, data_folder, Query, Evaluations, mark_duplicates, QueryParameters, TestParameters, \
    PromptLevel, NamesLevel
from queries.metrics import ndcg_k, hallucination_rate, mare_k, spearman_rho_k, kendall_tau_k, standard_ndcg_scoring

named_index_col = 'City'
index_col = 'Rank'
score_col = 'Overall Rating'
wights = {
    'Stability': 0.25,
    'Healthcare': 0.20,
    'Culture & Environment': 0.25,
    'Education': 0.10,
    'Infrastructure': 0.20
}
scoring_cols: list[str] = list(wights.keys())

class MostSimilarCities(BaseModel):
    most_similar_cities: list[str]  = Field(description="Ordered list of the k most similar cities to the target city.")

def closest_cities(df: DataFrame, target_city: str) -> DataFrame:
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

def create_prompt(df: DataFrame, target: str, top_k: int, prompt_level: PromptLevel) -> str:
    job = f"You are given a dataset of cities, with various attributes:\n{df.to_string(index=False)}"
    output = "Your output must contain only the required list of cities."

    match prompt_level:
        case PromptLevel.instruct:
            instruct = f"Return the {top_k} most similar cities to '{target}', based only on the Global Liveability Index (GLI) using only the provided data."
        case PromptLevel.formula:
            instruct = f"Using only the Global Liveability Index (GLI), which can be computed with the formula GLI = ({" + ".join(["'"+c+"'"+'*'+str(w) for c, w in wights.items()])}), return the {top_k} most similar cities to '{target}'."
        case PromptLevel.generic:
            instruct = f"Return the {top_k} most similar cities to '{target}', based only on the provided data."

    prompt = f"""{job}

{instruct}

{output}"""
    return prompt

@dataclass
class CityTestParameters(TestParameters):
    cities_per_query: list[int] = field(default_factory=lambda: [80]) #140 tot

@dataclass
class global_liveability(Test[CityTestParameters]):
    name: str = "Global Liveability Index"
    name_short: str = "GLI"
    json_schema = MostSimilarCities
    full_ds: DataFrame = None

    def _load_dataset(self):
        self.full_ds = pd.read_excel(data_folder / 'cities' / 'global_liveability.xlsx', sheet_name='Foglio2', index_col=index_col)

    def _build_prompt(self, df: DataFrame, target: str, top_k: int, prompt_level: PromptLevel) -> str:
        return create_prompt(df, target, top_k, prompt_level)

    def prepare_queries_for_direct(self):
        self.queries = []
        if self.full_ds is None:
            self._load_dataset()
        df = self.full_ds.copy()
        # normalize score
        df[score_col] = df[score_col] / df[score_col].max()

        curr_seed = self.parameters.seed
        counter = 0
        ds_id = 0
        pbar = tqdm(total=self.parameters.n_queries*len(self.parameters.cities_per_query)*len(self.parameters.kp)*len(self.parameters.prompt_levels)*len(self.parameters.names_levels),
                    desc="Generating queries",
                    unit="query",
                    colour='green')

        for _ in range(self.parameters.n_queries):
            for c_per_query in self.parameters.cities_per_query:
                if self.parameters.seed != 0:
                    random.seed(curr_seed)

                sampled_cities: list[str] = random.sample(k=c_per_query, population=df[named_index_col].tolist())
                fake_name_mapping = {c: "City " + str(i) for i, c in enumerate(sampled_cities, start=1)}
                df_sampled = df[df[named_index_col].isin(sampled_cities)]
                target_city = random.choice(sampled_cities)

                for kp in self.parameters.kp:
                    k = max(1, math.ceil(kp * c_per_query))

                    for prompt_level in self.parameters.prompt_levels:
                        for names_level in self.parameters.names_levels:
                            match names_level:
                                case NamesLevel.fake:
                                    curr_df = df_sampled.copy()
                                    curr_df = curr_df.drop(columns='Country')
                                    curr_df[named_index_col] = curr_df[named_index_col].map(fake_name_mapping)
                                    target = fake_name_mapping[target_city]
                                case NamesLevel.real:
                                    curr_df = df_sampled.copy()
                                    target = target_city

                            gt_df: DataFrame = closest_cities(curr_df, target)
                            q = Query(
                                id=counter,
                                ds_id=ds_id,
                                prompt=self._build_prompt(curr_df.drop(columns=score_col), target, k, prompt_level),
                                ground_truth=gt_df[named_index_col].tolist(),
                                ground_truth_scores=(1-gt_df['diff']).tolist(),
                                parameters=QueryParameters(k=k, prompt_level=prompt_level,names_level=names_level, n_elems=c_per_query),
                                response_json_schema=self.json_schema.model_json_schema() if self.parameters.enforce_json_schema else None,
                            )
                            self.queries.append(q)
                            counter += 1
                            pbar.update(1)

                curr_seed += 1
                ds_id += 1

        pbar.close()