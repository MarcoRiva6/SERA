import math
import random
import re
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from experiments.run_type import RunType
from queries.test import Query, Test, Evaluations, data_folder, \
    ensure_kaggle_ds, extract_separator_sequence, TestParameters, PromptLevel, NamesLevel, sample_ds_interesting

index_name = 'CustomerID'
ALPHA = 0.7                      # weight for basket-content similarity
TOP_N_ITEMS_MIN_PURCHASES = 1    # filter very rare items if needed (set >1 to reduce sparsity)
N_CUSTOMERS_PER_QUERY = 30        # number of similar customers to retrieve per query
N_QUERIES = 50
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

def count_row_sums(df: pd.DataFrame, threshold=1) -> int:
    row_sums = df.sum(axis=1, numeric_only=True)
    counter = 0
    for s in row_sums:
        if s > threshold:
            counter += 1
    return counter

class MostSimilarCustomers(BaseModel):
    top_k: list[int] = Field(description="The ordered list of top k most similar customers.")

class NElemsType(StrEnum):
    customers = 'customers'
    rows = 'rows'

@dataclass
class CustomerSegmentationTestParameters(TestParameters):
    alpha: float = ALPHA
    top_n_items_min_purchases: int = TOP_N_ITEMS_MIN_PURCHASES
    n_elems_type: NElemsType = NElemsType.rows # customers | rows
    rows_in_prompt_limit: int = 5500
    names_levels: tuple[NamesLevel, ...] = tuple(NamesLevel.fake)

@dataclass
class customer_segmentation(Test[CustomerSegmentationTestParameters]):
    name: str = "Customer Segmentation"
    name_short: str = "RFM"
    json_schema = MostSimilarCustomers
    stats_df: DataFrame = None
    named_index_col = index_name

    def _load_ds(self) -> DataFrame:
        file_path = data_folder / self.family / 'Online Retail.xlsx'
        ds_name = "yasserh/customer-segmentation-dataset"
        ensure_kaggle_ds(ds_name, file_path)
        return pd.read_excel(file_path)

    def _parse_query_manual(self, query: Query) -> bool:
        matched_lists: list[list[str]] = extract_separator_sequence(query.response, CHOSEN_SEPARATOR, query.parameters.k)
        matched_lists_len = len(matched_lists)
        if matched_lists_len == 0:
            print(f"Cannot parse query {query.id}: no '{CHOSEN_SEPARATOR}'-separated list found in the response.")
            return True
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
                if len(valid_parts) == query.parameters.k:
                    candidate_lists.append(valid_parts)
            if len(candidate_lists) == 0:
                print(f"Cannot parse query {query.id}: no valid '{CHOSEN_SEPARATOR}'-separated list with enough customer IDs found in the response.")
                return True
            elif len(candidate_lists) > 1:
                print(f"Warning for query {query.id}: multiple valid '{CHOSEN_SEPARATOR}'-separated lists found in the response. Using the last one.")
                top_k_list: list[str] = candidate_lists[-1] # use the last valid
            else:
                top_k_list: list[str] = candidate_lists[0]

        if top_k_list is None:
            return True
        else:
            query.parsed_response = top_k_list
            return False

    def evaluate_query(self, query: Query) -> Evaluations:
        pre_vals = query.ground_truth_scores.copy()
        temp_vals: list[float] = [x + 1 for x in query.ground_truth_scores]
        query.ground_truth_scores = temp_vals

        result = super().evaluate_query(query)

        query.ground_truth_scores = pre_vals

        return result

    def _init_query(self, seed: int, df: DataFrame, elem_per_query: int) -> tuple[tuple[DataFrame, str | int | None, list[int | str], list[float]],tuple[DataFrame, str | int | None, list[int | str], list[float]]]:
        cids_unique_full = df[self.named_index_col].unique().tolist()
        k_list = [max(1, math.ceil(kp * elem_per_query)) for kp in self.parameters.kp]
        min_customers_needed = max(k_list) + 1 #+1: perché se ne chiediamo K simili ad 1 significa che ce ne devono essere K+1
        min_customers_requested = min_customers_needed + 3
        while True:
            # --- 1. CAMPIONAMENTO ---
            match self.parameters.n_elems_type:
                case NElemsType.customers:
                    selected_cids = random.sample(cids_unique_full, elem_per_query)
                    final_df = df[df[self.named_index_col].isin(selected_cids)]

                    if len(final_df) > self.parameters.rows_in_prompt_limit:
                        continue

                case NElemsType.rows:
                    final_df = sample_ds_interesting(
                        df,
                        length=elem_per_query,
                        min_unique=min_customers_requested,
                        key=self.named_index_col
                    )
            # --- 2. CONTROLLO BASKET ---
            basket = build_basket_matrix(final_df, self.parameters.top_n_items_min_purchases)
            threshold = 1 if self.parameters.n_elems_type == NElemsType.rows else 2
            if count_row_sums(basket, threshold) < min_customers_requested:
                continue
            # --- 3. CALCOLO SIMILARITÀ ---
            basket_sim = compute_basket_similarity(basket)
            rfm = build_rfm_features(final_df)
            rfm_sim = compute_rfm_similarity(rfm)
            hybrid_sim = compute_hybrid_similarity(basket_sim, rfm_sim, alpha=self.parameters.alpha)
            # --- 4. RICERCA DEL CLIENTE INTERESSANTE ---
            # Mischiamo la lista clienti. Li testiamo uno a uno finché non troviamo
            # il primo che ha un vicino >= 0.4. Appena lo troviamo, abbiamo vinto.
            tutti_i_clienti = final_df[self.named_index_col].unique().tolist()
            random.shuffle(tutti_i_clienti)

            trovato_interessante = False
            selected_cid = None

            for cid in tutti_i_clienti:
                top_similar_df = get_top_k_similar(hybrid_sim, cid, k=elem_per_query)

                # Controlliamo il livello di similarità del più vicino
                if top_similar_df.values.tolist()[0] >= 0.3:
                    selected_cid = cid
                    trovato_interessante = True
                    break

            # Se NESSUN cliente del batch ha similarità >= 0.3, buttiamo il dataset
            if not trovato_interessante:
                continue

            sorted_cids = top_similar_df.index.tolist()
            ground_truth_vals = top_similar_df.values.tolist()

            result = (final_df, selected_cid, sorted_cids, ground_truth_vals)
            return result, result


    def _create_prompt(self, df: DataFrame, target: str|int|None, k: int, prompt_level: PromptLevel) -> str:
        if not self.parameters.enforce_json_schema:
            raise NotImplementedError("il caso senza json_schema non è supportato.")

        output = "Your output must contain only the required list of customers."
        #PROBLEMA: questi sono gli attributi solo della prima riga del target CID! -> LOTUS è icompatibile con questo dataset
        attributes_without_index = ", ".join(f"{{{col}}}: {val}" for col, val in df[df[index_name] == target].iloc[0].items() if col != index_name)

        match self.run_type:
            case RunType.LOTUS:
                raise Exception("LOTUS è incompatibile con il DS customers")
                match level:
                    case PromptLevel.formula:
                        prompt = \
                            f"""Return the {{{index_name}}} most similar to the customer {cid}, whose attributes are: {attributes_without_index}. The similarity score must be computed following these steps:
    1.	Basket-Content Representation
        1.1	Build a customer–item matrix by cross-tabulating customers against the products they purchased.
        1.2	Compute the cosine similarity between customers using these vectors.
    2.	RFM Behavioral Features
        2.1	For each customer, compute Recency (days since last purchase), Frequency (number of purchase events), and Monetary value (total spend).
        2.2	Normalize these RFM values so that each feature is on a comparable scale.
        2.3	Compute the cosine similarity between customers based on these normalized RFM vectors.
    3.	Final Score
        Combine the two similarity measures as follows:
        •	{alpha*100:.0f}% weight for the basket-content similarity
        •	{(1-alpha)*100:.0f}% weight for the RFM similarity"""
                    case PromptLevel.instruct:
                        prompt = \
                            f"""Return the {{{index_name}}} most similar to customer {cid}, whose attributes are: {attributes_without_index}.
    To determine similarity, evaluate customers across two standard retail dimensions, weighting them respectively {alpha*100:.0f}% and {(1-alpha)*100:.0f}%:
    1. Product Affinity (basket-content similarity).
    2. RFM Profile (Recency, Frequency, Monetary)."""
                    case PromptLevel.generic:
                        prompt = f"Return the {{{index_name}}} most similar to customer {cid}, whose attributes are: {attributes_without_index}."

            case RunType.DIRECT:
                match prompt_level:
                    case PromptLevel.formula:
                        instruction = \
                            f"""Return the {k} customers most similar to the customer {target}. The similarity score must be computed following these steps:
    1.	Basket-Content Representation
        1.1	Build a customer–item matrix by cross-tabulating customers against the products they purchased.
        1.2	Compute the cosine similarity between customers using these vectors.
    2.	RFM Behavioral Features
        2.1	For each customer, compute Recency (days since last purchase), Frequency (number of purchase events), and Monetary value (total spend).
        2.2	Normalize these RFM values so that each feature is on a comparable scale.
        2.3	Compute the cosine similarity between customers based on these normalized RFM vectors.
    3.	Final Score
        Combine the two similarity measures as follows:
        •	{self.parameters.alpha*100:.0f}% weight for the basket-content similarity
        •	{(1-self.parameters.alpha)*100:.0f}% weight for the RFM similarity"""
                    case PromptLevel.instruct:
                        instruction = \
                            f"""Return {k} customers most similar to customer {target} based on their purchasing behavior.
To determine similarity, evaluate customers across two standard retail dimensions, weighting them respectively {self.parameters.alpha*100:.0f}% and {(1-self.parameters.alpha)*100:.0f}%:
    1. Product Affinity (basket-content similarity).
    2. RFM Profile (Recency, Frequency, Monetary)."""
                    case PromptLevel.generic:
                        instruction = f"Return the {k} customers most similar to customer {target}."
                prompt = f"You are given the following dataset of customer purchase histories:\n{df.to_string(index=False)}\n\n{instruction}\n\n{output}"

        return prompt

    def _prepare_df(self, df: DataFrame) -> DataFrame:
        # Drop unnecessary columns
        clean_df = df.drop(columns=["Description", "Country"])
        # Remove rows with missing CustomerID
        clean_df = clean_df.dropna(subset=["CustomerID"])
        # Convert CustomerID to int
        clean_df["CustomerID"] = clean_df["CustomerID"].astype(int)
        # Remove cancellations (InvoiceNo that start with 'C')
        clean_df = clean_df[~clean_df["InvoiceNo"].astype(str).str.startswith("C")]
        # Remove negative or zero quantities / unit prices
        clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["UnitPrice"] > 0)]
        # Total price per line
        clean_df["TotalPrice"] = clean_df["Quantity"] * clean_df["UnitPrice"]

        # 2. Raggruppiamo per cliente per calcolare le loro statistiche base nel dataset completo
        df_valid = clean_df.copy()
        self.stats_df = df_valid.groupby('CustomerID').agg(
            NumRows=('InvoiceNo', 'count'),
            NumInvoices=('InvoiceNo', 'nunique')
        )

        return clean_df
