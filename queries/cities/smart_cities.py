import random
from dataclasses import field, dataclass

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import Test, data_folder, Query, Evaluations


def closest_cimi_cities(df: pd.DataFrame, target_city: str, k: int) -> pd.DataFrame:
    mask = df["City"] == target_city
    if not mask.any():
        raise ValueError(f"City not found: {target_city!r}")

    target_cimi = df.loc[mask, "CIMI"].iloc[0]

    # Compute absolute distance and take the k smallest excluding the target
    out = (
        df.assign(cimi_diff=(df["CIMI"] - target_cimi).abs())
        .loc[~mask]
        .nsmallest(k, "cimi_diff")
        .loc[:, ["City", "CIMI", "cimi_diff"]]
    )
    return out

def create_prompt(df: DataFrame, target: str, top_k: int) -> str:
    job = "You are given a dataset of cities, with various attributes:"
    instruct = f"Return the {top_k} most similar cities to {target}, based ONLY on the provided data."
    prompt = \
        f"""{job}
{df.to_string(index=False)}

{instruct}"""
    return prompt

@dataclass
class CityQuery(Query):
    target_city: str
    ground_truth: list[str]

class CitiesAnswer(BaseModel):
    most_similar_cities: list[str] = Field(description="Ordered list of the most similar cities to the target city.")

@dataclass
class smart_cities(Test):
    name: str = "Most similar smart cities"
    full_ds: DataFrame = None
    @dataclass
    class Params(Test.Params):
        cities_per_query: int = 80
        n_queries: int = 10
        top_k: int = 5
        enforce_json_schema: bool = True
        pass
    parameters: Params = field(default_factory=Params)

    def _load_dataset(self):
        self.full_ds = pd.read_excel(data_folder / 'cities' / 'smart_cities.xlsx', index_col='Rank')

    def evaluate_query(self, query: Query) -> Evaluations:
        pass

    def prepare_queries_for_direct(self):
        self.queries = []
        df = self.full_ds.copy()
        # Normalize CIMI
        df["CIMI"] = df["CIMI"] / df["CIMI"].max()
        df_uniform = df.copy()
        

        curr_seed = self.params.seed

        for _ in range(self.params.n_queries):
            if self.params.seed != 0:
                random.seed(curr_seed)
            sampled_cities = random.sample(k=self.params.cities_per_query, population=df['City'].tolist())
            df_sampled = df[df['City'].isin(sampled_cities)]
            target_city = random.choice(sampled_cities)
            df_closest = closest_cimi_cities(df_sampled, target_city, self.params.top_k)
            ground_truth = df_closest['City'].tolist()
            q = CityQuery(
                prompt=create_prompt(df_sampled.drop(columns='CIMI'), target_city, self.params.top_k),
                ground_truth=ground_truth,
                target_city=target_city,
                response_json_schema=CitiesAnswer.model_json_schema() if self.params.enforce_json_schema else None,
            )
            self.queries.append(q)

            curr_seed += 1
