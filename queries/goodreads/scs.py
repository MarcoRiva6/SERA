import os
import sys

import numpy as np

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)
from rating import *

class BestBooks(BaseModel):
    top_k: list[GTT] = Field(description="Top k most loved books according to SCS")

@dataclass
class scs(rating):
    name: str = "Goodreads Semantic Consensus Score"
    name_short: str = "SCS"
    json_schema = BestBooks
    named_index_col = 'book_id'

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

        while True:
            return (sample_ds_interesting(df=df, length=elem_per_query, min_unique=min_books_requested, key="book_id")
                    .sample(frac=1, random_state=current_seed)
                    .drop(columns=['user_id','date_added','date_updated','read_at','started_at','n_comments']))
            _, gt_scores = self.build_ground_truth(df=final_df, target=None)


    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], list[float]]:
        df = df.copy()
        df['SPS'] = df['rating'] - 3
        base_scs = df.groupby('book_id')['SPS'].sum().reset_index(name='Base_SCS')
        positive_reviews = df[df['SPS'] > 0].copy()
        positive_reviews['vote_weight'] = positive_reviews['n_votes'] + 1
        validation_weight = positive_reviews.groupby('book_id')['vote_weight'].sum().reset_index(name='Validation_Weight')
        risultato = pd.merge(base_scs, validation_weight, on='book_id', how='left')
        risultato['Validation_Weight'] = risultato['Validation_Weight'].fillna(0)
        risultato['Final_SCS_Raw'] = risultato['Base_SCS'] * risultato['Validation_Weight']
        # aggiungo offset
        min_score = risultato['Final_SCS_Raw'].min()
        offset = abs(min_score) + 1 if min_score < 0 else 1
        risultato['Final_SCS_NDCG'] = risultato['Final_SCS_Raw'] + offset

        risultato = risultato.sort_values(by=['Final_SCS_NDCG', self.named_index_col], ascending=[False, True]).reset_index(drop=True)
        return risultato[self.named_index_col].tolist(), risultato['Final_SCS_NDCG'].tolist()

    def build_prompt_df(self, df: DataFrame, completeness_level: CompletenessLevel) -> DataFrame:
        return df.drop(columns=['rating'])

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given a dataset of book reviews:"
        output = f"Your output must contain only the required list of {self.named_index_col}s."

        match q_params.prompt_level:
            case PromptLevel.generic:
                raise NotImplementedError(f"{q_params.prompt_level} not implemented for test goodreads/scs")
            case PromptLevel.instruct:
                instruction = \
                    f"""Return the top {q_params.k} most loved books ('{self.named_index_col}') from the provided dataset using the Semantic Consensus Score (SCS).
To calculate the SCS for a book: first, analyze the content of each review and assign it a Semantic Polarity Score (SPS) ranging from -2 (highly negative) to +2 (highly positive). Next, group the reviews by book and sum the SPS values for each book. Finally, multiply this base sum by the total sum of (n_votes + 1), but only counting the votes from reviews that received a positive SPS (+1 or +2). Rank the books in descending order based on their final SCS.
To resolve ties, order books by their id in descending order."""
            case PromptLevel.formula:
                instruction = \
                    f"""Return the top {q_params.k} most loved books ('{self.named_index_col}') from the provided dataset using the Semantic Consensus Score (SCS).
To calculate the Semantic Consensus Score (SCS) and rank the most loved books, execute this algorithm for the provided dataset:
    1. Read the content of each review and assign a Semantic Polarity Score (SPS): +2 for glowing/enthusiastic, +1 for generally positive, 0 for neutral/mixed, -1 for negative, and -2 for highly negative/hated.
    2. Identify all unique 'book_id' values in the dataset.
    3. For each unique 'book_id', isolate all its associated reviews.
    4. Calculate the 'Base_SCS' by summing the SPS values of all reviews belonging to that 'book_id'.
    5. Filter the reviews of that 'book_id' to isolate only those with a positive SPS.
    6. For this filtered subset of positive reviews, calculate the 'Validation_Weight' by taking the 'n_votes' of each positive review, adding 1 to it, and summing these values together. (If a book has no positive reviews, the Validation_Weight is 0).
    7. Multiply the 'Base_SCS' by the 'Validation_Weight'. This final number is the SCS for that 'book_id'.
    8. Repeat steps 4 through 7 for every unique 'book_id'.
    9. Sort the books based on their final SCS in descending order.
    10.To resolve ties, order books by their id in descending order."""

        return job, f"{instruction}\n\n{output}"