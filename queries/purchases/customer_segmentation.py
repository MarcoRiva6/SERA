import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import timedelta
from enum import auto
from json import JSONDecodeError
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from queries.test import Query, Test, Evaluations, data_folder, extract_json, Metric, extract_list, \
    ensure_kaggle_ds, hallucination_rate, mare_k, spearman_rho_k, ndcg_k, \
    extract_separator_sequence

ALPHA = 0.7                      # weight for basket-content similarity
TOP_N_ITEMS_MIN_PURCHASES = 1    # filter very rare items if needed (set >1 to reduce sparsity)
N_CUSTOMERS_PER_QUERY = 30        # number of similar customers to retrieve per query
N_QUERIES = 50
TOP_K = 5
CHOSEN_SEPARATOR = '|'

def build_basket_matrix(df: pd.DataFrame,
                        min_item_purchases: int) -> pd.DataFrame:
    """
    Build a customer–item matrix (counts of purchased items).
    Optionally filter items bought fewer than `min_item_purchases` times.
    """
    if min_item_purchases > 1:
        # Count how many times each StockCode appears overall
        item_counts = df["StockCode"].value_counts()
        valid_items = item_counts[item_counts >= min_item_purchases].index
        df = df[df["StockCode"].isin(valid_items)]

    # CustomerID x StockCode, values = count of lines (or sum of Quantity if you prefer)
    basket = pd.crosstab(df["CustomerID"], df["StockCode"])

    return basket


def compute_basket_similarity(basket: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between customers based on basket-content.
    Returns a DataFrame (index & columns = CustomerID).
    """
    # Convert to numpy matrix
    X = basket.values

    # Cosine similarity; result is (n_customers x n_customers)
    sim_matrix = cosine_similarity(X)

    # Convert to DataFrame with CustomerID indices
    sim_df = pd.DataFrame(sim_matrix, index=basket.index, columns=basket.index)

    # Replace any NaN (e.g., all-zero rows) with 0
    sim_df = sim_df.fillna(0.0)

    return sim_df


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features per customer.
    """
    # Reference date: one day after last invoice in the dataset
    max_date = df["InvoiceDate"].max()
    ref_date = max_date + timedelta(days=1)
    # Group by customer
    grouped = df.groupby("CustomerID")
    # Recency: days since last purchase
    recency = grouped["InvoiceDate"].max().apply(lambda d: (ref_date - d).days)
    # Frequency: number of unique invoices
    frequency = grouped["InvoiceNo"].nunique()
    # Monetary: total spent
    monetary = grouped["TotalPrice"].sum()

    rfm = pd.DataFrame({
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    })

    return rfm


def compute_rfm_similarity(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between customers based on RFM features.
    Returns a DataFrame (index & columns = CustomerID).
    """

    # We need to transform Recency because lower recency (more recent) should mean "better".
    # Option 1: invert recency by multiplying by -1 (so high = more recent).
    rfm_transformed = rfm.copy()
    rfm_transformed["Recency"] = -rfm_transformed["Recency"]

    # Scale features (z-score) so that Recency, Frequency, Monetary are on comparable scales
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_transformed)

    # Cosine similarity on scaled RFM
    rfm_sim_matrix = cosine_similarity(X_scaled)

    rfm_sim_df = pd.DataFrame(rfm_sim_matrix, index=rfm.index, columns=rfm.index)
    rfm_sim_df = rfm_sim_df.fillna(0.0)

    return rfm_sim_df

def compute_hybrid_similarity(basket_sim: pd.DataFrame,
                              rfm_sim: pd.DataFrame,
                              alpha: float) -> pd.DataFrame:
    """
    Combine basket-content similarity and RFM similarity:
        hybrid = alpha * basket_sim + (1 - alpha) * rfm_sim
    """
    # Align indices just in case (intersection of customers)
    common_customers = basket_sim.index.intersection(rfm_sim.index)

    basket_sim = basket_sim.loc[common_customers, common_customers]
    rfm_sim = rfm_sim.loc[common_customers, common_customers]

    hybrid = alpha * basket_sim + (1.0 - alpha) * rfm_sim

    return hybrid

def get_top_k_similar(sim_df: pd.DataFrame,
                      customer_id: int,
                      k: int = 10) -> pd.Series:
    """
    Given a similarity matrix (index & cols are CustomerID),
    return the top-k most similar customers (excluding the customer itself).
    Returns a Series: index = CustomerID, values = similarity score.
    """
    if customer_id not in sim_df.index:
        raise ValueError(f"CustomerID {customer_id} not found in similarity matrix.")
    # Get row for the customer
    sims = sim_df.loc[customer_id].copy()
    # Remove self-similarity
    sims = sims.drop(index=customer_id, errors="ignore")
    # Sort by similarity descending
    sims_sorted = sims.sort_values(ascending=False)
    # Return top-k
    return sims_sorted.head(k)

def count_row_sums(df: pd.DataFrame) -> int:
    row_sums = df.sum(axis=1, numeric_only=True)
    counter = 0
    for s in row_sums:
        if s > 1:
            counter += 1
    return counter

def select_random_customer_ids(n: int, cid_list):
    cst_ids = []
    for _ in range(n):
        cst_id = random.choice(cid_list)
        while cst_id in cst_ids:
            cst_id = random.choice(cid_list)
        cst_ids.append(cst_id)
    return cst_ids

class MostSimilarCustomers(BaseModel):
    top_k: list[int] = Field(description="The ordered list of top k most similar customers.")

def create_prompt(df: pd.DataFrame, cid: int, top_k: int, alpha: float, level: str, json_schema: bool) -> str:
    if json_schema:
        output_string = f"""Your output MUST contain only a sorted list of the most similar customers (represented by their customer_id),
from most to least similar, as per the following JSON schema:
{json.dumps(MostSimilarCustomers.model_json_schema())}"""
    else:
        output_string = f"""Your output MUST contain only a sorted list of the most similar customers (represented by their customer_id),
from most to least similar, separated by the character '{CHOSEN_SEPARATOR}'."""

    if level == 'formula':
        prompt = \
f"""
your are given the following dataset of customer purchase histories:
{df.to_string(index=False)}

Your job is to find the top {top_k} customers who are most similar to the customer {cid}. The similarity score MUST
be computed following these steps:
1.	Basket-Content Representation
	1.1	Build a customer–item matrix by cross-tabulating customers against the products they purchased.
	1.2	Compute the cosine similarity between customers using these vectors.
2.	RFM Behavioral Features
	2.1	For each customer, compute Recency (days since last purchase), Frequency (number of purchase events), and Monetary value (total spend).
	2.2	Normalize these RFM values so that each feature is on a comparable scale.
	2.3	Compute the cosine similarity between customers based on these normalized RFM vectors.
3.	Final Score
	•	Combine the two similarity measures with a weighted average:
	•	{alpha*100:.0f}% weight for the basket-content similarity
	•	{(1-alpha)*100:.0f}% weight for the RFM similarity

Do NOT guess or hallucinate missing data.
Only reason using the purchase histories provided in the dataset.

Identify the top {top_k} customers who are most similar to the given customer, whose customer_id is {cid}.
{output_string}
"""
    elif level == 'medium':
        prompt = \
f"""
your task is to compare customers only based on their purchasing behavior in the following dataset.
{df.to_string(index=False)}
A customer is more similar if:
1.	They bought many of the same products as the target customer
2.	Their purchase frequency is similar
3.	Their monetary spend is similar
4.	Their recency of last purchase is similar

Do NOT guess or hallucinate missing data.
Only reason using the purchase histories provided in the dataset.

Identify the top {top_k} customers who are most similar to the given customer, whose customer_id is {cid}.
{output_string}
"""
    elif level == 'generic':
        prompt = \
f"""
Your are given the following dataset of customer purchase histories:
{df.to_string(index=False)}

Do NOT guess or hallucinate missing data.
Only reason using the purchase histories provided in the dataset.

Identify the top {top_k} customers who are most similar to the given customer, whose customer_id is {cid}.
{output_string}
"""
    return prompt

@dataclass
class CustomerSegmentationQuery(Query):
    customer_id: int
    ground_truth: list[int]
    ground_truth_values: list[float]
    prompt_level: str
    parsed_response: list[int]

class CustomerSegmentationMetrics(Metric):
    NDCG_SCORES = auto() # NDCG considerando i punteggi reali, indipendentemente da k
    NDCG_K = auto() # NDCG considerando k-i se l'elemento i-esimo dilla lista predetta è presente nella k-ground truth (indipendentemente dalla sua posizione)
    MARE = auto() # MARE considerando tutti gli elementi della ground truth
    MARE_K = auto() # MARE considerando solo i primi k elementi della ground truth
    SPEARMAN = auto() # Spearman considerando tutti gli elementi della ground truth
    SPEARMAN_K = auto() # Spearman considerando solo i primi k elementi della ground truth
    HALLUCINATION_RATE = auto()

@dataclass
class customer_segmentation(Test):
    name: str = "Customer Segmentation"
    full_df: DataFrame = None
    clean_df: DataFrame = None
    pre_queries_df: DataFrame = None
    @dataclass
    class Params(Test.Params):
        alpha: float = ALPHA
        top_n_items_min_purchases: int = TOP_N_ITEMS_MIN_PURCHASES
        n_customers_per_query: int = N_CUSTOMERS_PER_QUERY
        top_k: int = TOP_K
        n_queries: int = N_QUERIES
        rows_in_prompt_limit: int = 2000
        prompt_levels: list[str] = field(default_factory=lambda: ['medium']) # generic, medium, formula
        enforce_json_schema: bool = True
    params: Params = field(default_factory=Params)

    def load_csv(self, file_path: Path = None) -> None:
        if file_path is None:
            file_path = data_folder / self.family / 'Online Retail.xlsx'
        ds_name = "yasserh/customer-segmentation-dataset"
        ensure_kaggle_ds(ds_name, file_path)
        self.full_df = pd.read_excel(file_path)

    def parse_query(self, query: CustomerSegmentationQuery) -> bool:
        """
        Parse the response of a query to extract the list of top-k similar customers.
        It automatically sets the `parsing_failed` and `parsed_response` attributes of the query.
        :param query: CustomerSegmentationQuery
        :return: bool indicating whether parsing was successful
        """
        top_k_list: list[str] = None
        if query.response_json_schema:
            try:
                json_response = extract_json(query.response, query.customer_id)
                top_k_list = json_response['top_k']
            except (JSONDecodeError, KeyError, TypeError):
                try:
                    json_response = {'top_k': extract_list(query.response, query.customer_id)}
                    top_k_list = json_response['top_k']
                except (JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Cannot parse query {query.id}: {e}")
                    query.parsing_failed = True
        else:
            matched_lists: list[list[str]] = extract_separator_sequence(query.response, CHOSEN_SEPARATOR, self.params.top_k)
            matched_lists_len = len(matched_lists)
            if matched_lists_len == 0:
                print(f"Cannot parse query {query.id}: no '{CHOSEN_SEPARATOR}'-separated list found in the response.")
                query.parsing_failed = True
            else:
                candidate_lists: list[list[str]] = []
                for matched_list in matched_lists:
                    valid_parts: list[str] = []
                    for i, part in enumerate(matched_list):
                        found_numbers: list = re.findall(r"\d{3,}", part)
                        if len(found_numbers) == 0:
                            continue # no numbers within an element of the list: skip
                        elif len(found_numbers) == 1:
                            valid_parts.append(found_numbers[0])
                        elif len(found_numbers) > 1:
                            if i == 0: # if multiple numbers in the first part, take the last one
                                valid_parts.append(found_numbers[-1])
                            elif i == matched_lists_len - 1: # if multiple numbers in the last part, take the first one
                                valid_parts.append(found_numbers[0])
                            else: # ambiguous part in the middle of the list
                                valid_parts = [] # invalidate the entire list
                                break
                    if len(valid_parts) == self.params.top_k:
                        candidate_lists.append(valid_parts)
                if len(candidate_lists) == 0:
                    print(f"Cannot parse query {query.id}: no valid '{CHOSEN_SEPARATOR}'-separated list with enough customer IDs found in the response.")
                    query.parsing_failed = True
                elif len(candidate_lists) > 1:
                    print(f"Warning for query {query.id}: multiple valid '{CHOSEN_SEPARATOR}'-separated lists found in the response. Using the last one.")
                    top_k_list: list[str] = candidate_lists[-1] # use the last valid
                else:
                    top_k_list: list[str] = candidate_lists[0]

        if query.parsing_failed or top_k_list is None or len(top_k_list) != self.params.top_k:
            return False
        try:
            query.parsed_response = [int(s) for s in top_k_list]
            query.parsing_failed = False
            return True
        except (TypeError, ValueError) as e:
            print(f"Cannot parse query {query.id}: invalid customer IDs in the extracted list. {e}")
            query.parsing_failed = True
            return False


    def evaluate_query(self, query: CustomerSegmentationQuery) -> Evaluations:
        failing_scores = Evaluations(ndcg_scores=0.0, ndcg_k=0.0, mare=self.params.top_k, mare_k=self.params.top_k, spearman=-1.0, spearman_k=-1.0, hallucination_rate=self.params.top_k)

        if not self.parse_query(query):
            return failing_scores

        temp_vals: list[float] = [x + 1 for x in query.ground_truth_values]
        ndcg_scores: list[float] = [temp_vals[query.ground_truth.index(cust)]
                                    if cust in query.ground_truth
                                    else 0
                                    for cust in query.parsed_response]

        return Evaluations(ndcg_scores=ndcg_k(ndcg_scores, temp_vals, self.params.top_k),
                           ndcg_k=ndcg_k(
                               relevance_scores=[self.params.top_k - i if cust in query.ground_truth[:self.params.top_k] else 0 for
                                                 i, cust in
                                                 enumerate(query.parsed_response)], k=self.params.top_k),
                           mare=mare_k(query.parsed_response, query.ground_truth),
                           mare_k=mare_k(query.parsed_response, query.ground_truth, self.params.top_k),
                           spearman=spearman_rho_k(query.parsed_response, query.ground_truth),
                           spearman_k=spearman_rho_k(query.parsed_response, query.ground_truth, self.params.top_k),
                           hallucination_rate=hallucination_rate(query.parsed_response, query.ground_truth))

    def init_queries(self) -> None:
        current_seed = self.params.seed

        self.queries: list[CustomerSegmentationQuery] = []
        cids_unique_full = self.clean_df['CustomerID'].unique().tolist()
        counter = 0

        for _ in range(self.params.n_queries):
            if self.params.seed != 0:
                random.seed(current_seed)
            # trim dataset to N_CUSTOMERS_PER_QUERY customers and verify it is interesting
            while True:
                selected_cids = select_random_customer_ids(self.params.n_customers_per_query, cids_unique_full)

                temp_df = self.clean_df[self.clean_df['CustomerID'].isin(selected_cids)]
                if len(temp_df) > self.params.rows_in_prompt_limit:
                    current_seed += 1
                    continue

                basket = build_basket_matrix(temp_df, self.params.top_n_items_min_purchases)
                if count_row_sums(basket) < 15: # n of interesting rows
                    current_seed += 1
                    continue
                df = temp_df
                break

            basket_sim = compute_basket_similarity(basket)

            rfm = build_rfm_features(df)
            rfm_sim = compute_rfm_similarity(rfm)

            hybrid_sim = compute_hybrid_similarity(basket_sim, rfm_sim, alpha=self.params.alpha)

            selected_cid = random.choice(selected_cids)
            top_similar_df = get_top_k_similar(hybrid_sim, selected_cid, k=self.params.n_customers_per_query)
            sorted_cids = top_similar_df.index.tolist()
            ground_truth_vals = top_similar_df.values.tolist()

            for p_level in self.params.prompt_levels:
                self.queries.append(CustomerSegmentationQuery(
                    id=counter,
                    customer_id=selected_cid,
                    prompt=create_prompt(df, selected_cid, self.params.top_k, self.params.alpha, p_level, self.params.enforce_json_schema),
                    ground_truth=sorted_cids,
                    ground_truth_values=ground_truth_vals,
                    prompt_level=p_level,
                    response=None,
                    evaluations=Evaluations(),
                    response_json_schema=MostSimilarCustomers.model_json_schema() if self.params.enforce_json_schema else None,
                    parsing_failed=None,
                    parsed_response=None
                ))
                counter += 1

            current_seed += 1

    def prepare_df(self) -> None:
        # Drop unnecessary columns
        self.clean_df = self.full_df.drop(columns=["Description", "Country"])
        # Remove rows with missing CustomerID
        self.clean_df = self.clean_df.dropna(subset=["CustomerID"])
        # Convert CustomerID to int
        self.clean_df["CustomerID"] = self.clean_df["CustomerID"].astype(int)
        # Remove cancellations (InvoiceNo that start with 'C')
        self.clean_df = self.clean_df[~self.clean_df["InvoiceNo"].astype(str).str.startswith("C")]
        # Remove negative or zero quantities / unit prices
        self.clean_df = self.clean_df[(self.clean_df["Quantity"] > 0) & (self.clean_df["UnitPrice"] > 0)]
        # Total price per line
        self.clean_df["TotalPrice"] = self.clean_df["Quantity"] * self.clean_df["UnitPrice"]

    # def prepare_lotus(self) -> None:
    #     self.load_csv()
    #     self.prepare_df()
    #     pass

    def prepare_queries_for_direct(self) -> None:
        print('loading dataset...')
        self.load_csv()
        print('preparing DF...')
        self.prepare_df()
        print('initializing queries...')
        self.init_queries()
