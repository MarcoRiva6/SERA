import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import timedelta
from enum import auto
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from queries.test import Query, Test, Evaluations, data_folder, extract_json, Metric, precision_at_k, \
    spearman_rank_correlation, ndcg_at_k_scores, extract_list

ALPHA = 0.7                      # weight for basket-content similarity
TOP_N_ITEMS_MIN_PURCHASES = 1    # filter very rare items if needed (set >1 to reduce sparsity)
N_CUSTOMERS_PER_QUERY = 30        # number of similar customers to retrieve per query
N_QUERIES = 50
TOP_K = 5

def build_basket_matrix(df: pd.DataFrame,
                        min_item_purchases: int) -> pd.DataFrame:
    """
    Build a customer–item matrix (counts of purchased items).
    Optionally filter items bought fewer than `min_item_purchases` times.
    """
    # Count how many times each StockCode appears overall
    item_counts = df["StockCode"].value_counts()

    if min_item_purchases > 1:
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

class ResponseSchema(BaseModel):
    top_k: list[int] = Field(description="The ordered list of top k most similar customers.")

def create_prompt(df: pd.DataFrame, cid: int, top_k: int) -> str:
    prompt = \
f"""
Your task is to compare customers only based on their purchasing behavior in the following dataset.
{df.to_string(index=False)}
A customer is more similar if:
1.	They bought many of the same products as the target customer
2.	Their purchase frequency is similar
3.	Their monetary spend is similar
4.	Their recency of last purchase is similar

Do NOT guess or hallucinate missing data.
Only reason using the purchase histories provided in the dataset.

Identify the top {top_k} customers who are most similar to the customer {cid}.
Your output MUST be exactly a sorted list of the most similar customers (represented by their customer_id),
from most to least similar, as per the following JSON schema:
{json.dumps(ResponseSchema.model_json_schema())}
and nothing else.
Remember that you MUST only provide the final answer, not the reasoning steps.
"""
    return prompt

@dataclass
class CustomerSegmentationQuery(Query):
    customer_id: int
    ground_truth: list[int]
    ground_truth_values: list[float]

class CustomerSegmentationMetrics(Metric):
    NDCG_K = auto()
    PRECISION_K = auto()
    SPEARMAN_K = auto()

@dataclass
class customer_segmentation(Test):
    name = "Customer Segmentation"
    full_df: DataFrame = None
    clean_df: DataFrame = None
    pre_queries_df: DataFrame = None
    alpha: float = ALPHA
    top_n_items_min_purchases: int = TOP_N_ITEMS_MIN_PURCHASES
    n_customers_per_query: int = N_CUSTOMERS_PER_QUERY
    n_queries: int = N_QUERIES
    top_k: int = TOP_K

    def load_csv(self, file_path: Path = None) -> None:
        if file_path is None:
            file_path = data_folder / self.family / 'Online Retail.xlsx'
        if not os.path.exists(file_path):
            ds_folder = kagglehub.dataset_download("yasserh/customer-segmentation-dataset", force_download=True)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(Path(ds_folder) / file_path.name, file_path)
        self.full_df = pd.read_excel(file_path)

    def evaluate_query(self, query: CustomerSegmentationQuery) -> Evaluations:
        try:
            json_response = extract_json(query.response)
            if json_response == "": # try extracting directly the list ("[1,2,3]")
                json_response = {'top_k': extract_list(query.response)}

            top_k_list = json_response['top_k']
        except KeyError as e:
            print(f"KeyError while evaluating query for customer_id {query.customer_id}: {e}")
            return Evaluations(ndcg_5=0.0)
        except Exception as e:
            print(f"Error while evaluating query for customer_id {query.customer_id}: {e}")
            return Evaluations(ndcg_5=0.0, precision_5=0.0, spearman_5=0.0)
        temp_vals = [x + 1 for x in query.ground_truth_values]
        ndcg_scores = np.array([temp_vals[query.ground_truth.index(cust)]
                                      if cust in query.ground_truth
                                      else 0
                                      for cust in top_k_list])
        return {
            CustomerSegmentationMetrics.NDCG_K: ndcg_at_k_scores(temp_vals, ndcg_scores, k=self.top_k),
           # CustomerSegmentationMetrics.NDCG_K: sklearn.metrics.ndcg_score(temp_vals, ndcg_scores, k=TOP_K),
           # CustomerSegmentationMetrics.PRECISION_K: precision_at_k(query.ground_truth, top_k_list, k=TOP_K),
           # CustomerSegmentationMetrics.SPEARMAN_K: spearman_rank_correlation(query.ground_truth, top_k_list)
        }

    def init_queries(self) -> None:
        file_name = 'prepared_queries.csv'
        if os.path.exists(self.run_folder / file_name):
            self.csv_to_queries(CustomerSegmentationQuery, file_name, self.run_folder)
            return

        current_seed = self.seed
        row_limit = 2000

        self.queries = []
        cids_unique_full = self.clean_df['CustomerID'].unique().tolist()

        for _ in range(self.n_queries):
            if self.seed != 0:
                random.seed(current_seed)
            # trim dataset to N_CUSTOMERS_PER_QUERY customers and verify it is interesting
            while True:
                selected_cids = select_random_customer_ids(self.n_customers_per_query, cids_unique_full)

                temp_df = self.clean_df[self.clean_df['CustomerID'].isin(selected_cids)]
                if len(temp_df) > row_limit:
                    current_seed += 1
                    continue

                basket = build_basket_matrix(temp_df, self.top_n_items_min_purchases)
                if count_row_sums(basket) < 15: # n of interesting rows
                    current_seed += 1
                    continue
                df = temp_df
                break

            basket_sim = compute_basket_similarity(basket)

            rfm = build_rfm_features(df)
            rfm_sim = compute_rfm_similarity(rfm)

            hybrid_sim = compute_hybrid_similarity(basket_sim, rfm_sim, alpha=self.alpha)

            selected_cid = random.choice(selected_cids)
            top_similar_df = get_top_k_similar(hybrid_sim, selected_cid, k=self.n_customers_per_query)
            sorted_cids = top_similar_df.index.tolist()
            ground_truth_vals = top_similar_df.values.tolist()

            self.queries.append(CustomerSegmentationQuery(
                customer_id=selected_cid,
                prompt=create_prompt(df, selected_cid, self.top_k),
                ground_truth=sorted_cids,
                ground_truth_values=ground_truth_vals,
                response=None,
                evaluations=Evaluations(),
                response_json_schema=json.dumps(ResponseSchema.model_json_schema())
            ))

            current_seed += 1

        self.queries_to_csv(file_name)

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
        print('loading CSV...')
        self.load_csv()
        print('preparing DF...')
        self.prepare_df()
        print('initializing queries...')
        self.init_queries()
