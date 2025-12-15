import random
from dataclasses import dataclass, field
from enum import StrEnum, auto

from pandas import DataFrame
import pandas as pd
from pydantic import BaseModel, Field

from queries.test import Test, data_folder, Query, Evaluations, Metric, ndcg_k, mare_k, spearman_rho_k, \
    hallucination_rate, mark_duplicates

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

class PromptLevel(StrEnum):
    compute_GLI = 'compute_GLI'
    similar = 'similar'
    formula_GLI = 'formula_GLI'

class NamesLevels(StrEnum):
    fake = 'fake'
    real = 'real'

class MostSimilarCities(BaseModel):
    most_similar_cities: list[str]  = Field(description="Ordered list of the most similar cities to the target city.")

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
    job = "You are given a dataset of cities, with various attributes:"
    if prompt_level == PromptLevel.compute_GLI:
        instruct = f"Return the {top_k} most similar cities to {target}, based on the Global Liveability Index computed using ONLY the provided data."
    elif prompt_level == PromptLevel.similar:
        instruct = f"Return the {top_k} most similar cities to {target}, based ONLY on the provided data."
    elif prompt_level == PromptLevel.formula_GLI:
        instruct = f"Using the formula GLI = ({" + ".join(["'"+c+"'"+'*'+str(w) for c, w in wights.items()])}), return the {top_k} most similar cities to {target}, based ONLY on the computed GLI scores."
    else:
        raise ValueError(f"Unknown prompt level: {prompt_level}")

    prompt = f"""{job}

{df.to_string(index=False)}

{instruct}"""
    return prompt

class CityMetrics(Metric):
    NDCG_SCORES = auto()
    NDCG_K = auto()
    MARE = auto()
    MARE_K = auto()
    SPEARMAN = auto()
    SPEARMAN_K = auto()
    HALLUCINATION_RATE = auto()

@dataclass
class CityQuery(Query):
    parsed_response: list[str]
    target_city: str
    ground_truth: list[str]
    ground_truth_scores: list[float]
    prompt_level: PromptLevel
    names_level: NamesLevels

@dataclass
class global_liveability(Test):
    name: str = "Global Liveability Index"
    full_ds: DataFrame = None
    @dataclass
    class Params(Test.Params):
        cities_per_query: int = 80
        n_queries: int = 10
        top_k: int = 5
        prompt_levels: list[PromptLevel] = field(default_factory=lambda: [pl for pl in PromptLevel])
        names_levels: list[NamesLevels] = field(default_factory=lambda: [cn for cn in NamesLevels])
        enforce_json_schema: bool = True
    params: Params = field(default_factory=lambda: global_liveability.Params())

    def _load_dataset(self):
        self.full_ds = pd.read_excel(data_folder / 'cities' / 'global_liveability.xlsx', sheet_name='Foglio2', index_col=index_col)

    def _parse_query(self, query: CityQuery) -> bool:
        if query.response is None or query.response == '':
            return False
        if query.response_json_schema is not None:
            parsed_response = MostSimilarCities.model_validate_json(query.response)
            query.parsed_response = parsed_response.most_similar_cities
            return True

        return False

    def evaluate_query(self, query: CityQuery) -> Evaluations:
        failing_scores = Evaluations(ndcg_scores=0.0, ndcg_k=0.0, mare=self.params.top_k, mare_k=self.params.top_k, spearman=-1.0, spearman_k=-1.0, hallucination_rate=self.params.top_k)
        query.parsing_failed = not self._parse_query(query)
        if query.parsing_failed:
            return failing_scores
        marked_duplicate_response = mark_duplicates(query.parsed_response, query.ground_truth[:self.params.top_k])

        return Evaluations(ndcg_scores=ndcg_k([query.ground_truth_scores[query.ground_truth.index(city)] if city in query.ground_truth else 0 for city in marked_duplicate_response], query.ground_truth_scores, self.params.top_k),
                           ndcg_k=ndcg_k(
                               relevance_scores=[self.params.top_k - i if city in query.ground_truth[:self.params.top_k] else 0 for
                                                 i, city in
                                                 enumerate(marked_duplicate_response)], k=self.params.top_k),
                           mare=mare_k(marked_duplicate_response, query.ground_truth),
                           mare_k=mare_k(marked_duplicate_response, query.ground_truth, self.params.top_k),
                           spearman=spearman_rho_k(marked_duplicate_response, query.ground_truth),
                           spearman_k=spearman_rho_k(marked_duplicate_response, query.ground_truth, self.params.top_k),
                           hallucination_rate=hallucination_rate(query.parsed_response, query.ground_truth))

    def prepare_queries_for_direct(self):
        self.queries = []
        if self.full_ds is None:
            self._load_dataset()
        df = self.full_ds.copy()
        # normalize score
        df[score_col] = df[score_col] / df[score_col].max()

        curr_seed = self.params.seed
        counter = 0

        for _ in range(self.params.n_queries):
            if self.params.seed != 0:
                random.seed(curr_seed)
            sampled_cities: list[str] = random.sample(k=self.params.cities_per_query, population=df[named_index_col].tolist())
            fake_name_mapping = {c: "City " + str(i) for i, c in enumerate(sampled_cities, start=1)}
            df_sampled = df[df[named_index_col].isin(sampled_cities)]
            target_city = random.choice(sampled_cities)

            for prompt_level in self.params.prompt_levels:
                for names_level in self.params.names_levels:
                    if names_level == NamesLevels.fake:
                        curr_df = df_sampled.copy()
                        curr_df = curr_df.drop(columns='Country')
                        curr_df[named_index_col] = curr_df[named_index_col].map(fake_name_mapping)
                        target = fake_name_mapping[target_city]
                    elif names_level == NamesLevels.real:
                        curr_df = df_sampled.copy()
                        target = target_city
                    else:
                        raise ValueError(f"Unknown cities name: {names_level}")
                    gt_df: DataFrame = closest_cities(curr_df, target)
                    q = CityQuery(
                        id=counter,
                        prompt=create_prompt(curr_df.drop(columns=score_col), target, self.params.top_k, prompt_level),
                        ground_truth=gt_df[named_index_col].tolist(),
                        ground_truth_scores=(1-gt_df['diff']).tolist(),
                        target_city=target,
                        response=None,
                        parsed_response=None,
                        prompt_level=prompt_level,
                        names_level=names_level,
                        response_json_schema=MostSimilarCities.model_json_schema() if self.params.enforce_json_schema else None,
                        evaluations=None,
                        parsing_failed=None
                    )
                    self.queries.append(q)
                    counter += 1

            curr_seed += 1