from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import TestParameters, Test, data_folder, GroundTruthScoreList, finalize_kp, sample_ds_interesting, \
    CompletenessLevel, PartitionedQueryParameters, PromptLevel, PrintingMode

type GTT = str

class MostRelevantReviews(BaseModel):
    top_k: list[GTT] = Field(description="Top k most relevant reviews")

@dataclass
class rating(Test[GTT, TestParameters]):
    name: str = "Goodreads Ratings"
    name_short: str = "GRR"
    json_schema = MostRelevantReviews
    named_index_col = 'review_id'

    def _load_ds(self) -> DataFrame:
        file_path = data_folder / self.family / "goodreads_reviews_dedup.json"
        return pd.read_json(file_path, lines=True, nrows=100000)

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        k_list = []
        for kp in self.parameters.kp:
            k_list.append(finalize_kp(kp, elem_per_query))
        min_books_needed = max(k_list)
        match elem_per_query:
            case 30:
                add = 3
            case 70:
                add = 8
            case 150:
                add = 20
            case _:
                raise NotImplementedError(f"{elem_per_query} rows not implemented")
        min_books_requested = min_books_needed + add

        return (sample_ds_interesting(df=df, length=elem_per_query, min_unique=min_books_requested, key="book_id")
                    .sample(frac=1, random_state=current_seed)
                    .drop(columns=['user_id','date_added','date_updated','read_at','started_at','n_comments']))

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], list[float]]:
        gt_df = df.copy()
        gt_df = gt_df.sort_values(by=['rating', self.named_index_col], ascending=[False, True])
        return gt_df[self.named_index_col].tolist(), gt_df['rating'].tolist()

    def df_to_string_for_prompt(self, df: DataFrame, printing_mode: PrintingMode) -> str:
        match printing_mode:
            case PrintingMode.plain:
                return "\n".join(
                    f"review_id: {row.review_id}\n"
                    f"book_id: {row.book_id}\n"
                    f"n_votes: {row.n_votes}\n"
                    f"review text:\n"
                    f"{row.review_text}\n"
                    f"---"
                    for row in df.itertuples(index=False)
                )
            case _:
                NotImplementedError(f"{printing_mode} not implemented for test goodreads/rating")

    def build_prompt_df(self, df: DataFrame, completeness_level: CompletenessLevel) -> DataFrame:
        return df.drop(columns=['rating'])

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given a dataset of book reviews:"
        output = "Your output must contain only the required list of reviews_ids."

        match q_params.prompt_level:
            case PromptLevel.generic:
                instruction = f"Return the best {q_params.k} reviews."
            case PromptLevel.instruct:
                instruction = \
                    f"""Return the top {q_params.k} '{self.named_index_col}' from the provided dataset, ranked by their predicted star rating.
First, infer the hidden star rating (from 1 to 5) of each review based on the sentiment of its content. Rank the reviews in descending order based on this predicted rating. To resolve ties between reviews that receive the same predicted rating, use the 'review_id' in alphanumeric order."""
            case PromptLevel.formula:
                instruction = \
                    f"""Return the top {q_params.k} '{self.named_index_col}' from the provided dataset, ranked by their predicted star rating.
use the following algorithm to obtain the final result:
    1. Read the 'review_text' of the current row.
    2. Infer the hidden star rating of the review (an integer from 1 to 5, where 5 is extremely positive and 1 is extremely negative) based purely on the sentiment expressed in the text.
    3. Record this 'predicted_rating' alongside the row's 'n_votes' and 'review_id'.
    4. Repeat steps 1 to 3 for all rows in the dataset.
    5. Sort all reviews in descending order based primarily on their 'predicted_rating' (highest ratings first). For any reviews that share the same 'predicted_rating', apply a secondary sort based on their 'review_id' (alphanumeric order) to completely eliminate any ties."""

        return job, f"{instruction}\n\n{output}"