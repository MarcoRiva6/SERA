import os
import sys

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)

from customer_segmentation import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import random


def get_top_similar_customers(df: DataFrame, target_cid: int, alpha: float) -> pd.Series:
    """
    Calcola i clienti più simili secondo la metrica CBS (Customer Buying Signature).
    """
    # Lavoriamo su una copia per non alterare il dataframe originale
    data = df.copy()

    # -------------------------------------------------------------------------
    # DATA PREPARATION (Parsing stringhe e date)
    # -------------------------------------------------------------------------

    # 1. Parsing del formato europeo per UnitPrice (es. "2,55" -> 2.55)
    if data['UnitPrice'].dtype == 'object':
        data['UnitPrice'] = data['UnitPrice'].str.replace(',', '.').astype(float)

    # 2. Parsing della data (formato dd/mm/yy HH:MM)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d/%m/%y %H:%M')

    # Calcolo spesa totale per riga (utile per la metrica 'A')
    data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

    # -------------------------------------------------------------------------
    # 1. PRICE-TIER AFFINITY
    # -------------------------------------------------------------------------
    # Assegnazione delle fasce di prezzo
    conditions = [
        data['UnitPrice'] < 2.00,
        (data['UnitPrice'] >= 2.00) & (data['UnitPrice'] <= 5.00),
        data['UnitPrice'] > 5.00
    ]
    choices = ['Budget', 'Standard', 'Premium']
    data['PriceTier'] = np.select(conditions, choices, default='Other')

    # Creazione della matrice Cliente-FasciaPrezzo (somma delle Quantità)
    pt_matrix = data.pivot_table(
        index='CustomerID',
        columns='PriceTier',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    # Assicuriamoci che tutte e 3 le colonne esistano sempre, anche se vuote
    for tier in choices:
        if tier not in pt_matrix.columns:
            pt_matrix[tier] = 0
    pt_matrix = pt_matrix[choices] # Ordiniamo le colonne

    # Calcolo della Cosine Similarity per Price-Tier
    pt_sim_matrix = cosine_similarity(pt_matrix)
    pt_sim_df = pd.DataFrame(pt_sim_matrix, index=pt_matrix.index, columns=pt_matrix.index)

    # -------------------------------------------------------------------------
    # 2. TVA BEHAVIORAL FEATURES (Tenure, Variety, Average Basket)
    # -------------------------------------------------------------------------
    # Raggruppiamo per cliente e calcoliamo le metriche grezze
    tva_df = data.groupby('CustomerID').agg(
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max'),
        Variety_V=('StockCode', 'nunique'),
        UniqueInvoices=('InvoiceNo', 'nunique'),
        TotalMonetary=('TotalSpend', 'sum')
    )

    # Calcolo esatto di T, V, A
    tva_df['Tenure_T'] = (tva_df['LastPurchase'] - tva_df['FirstPurchase']).dt.days
    tva_df['AverageBasket_A'] = tva_df['TotalMonetary'] / tva_df['UniqueInvoices']

    # Selezioniamo solo i 3 vettori finali
    tva_vectors = tva_df[['Tenure_T', 'Variety_V', 'AverageBasket_A']].copy()

    # Normalizzazione Min-Max (su scala 0-1)
    scaler = MinMaxScaler()
    tva_scaled = scaler.fit_transform(tva_vectors)

    # Calcolo della Cosine Similarity per TVA
    tva_sim_matrix = cosine_similarity(tva_scaled)
    tva_sim_df = pd.DataFrame(tva_sim_matrix, index=tva_vectors.index, columns=tva_vectors.index)

    # -------------------------------------------------------------------------
    # 3. FINAL SCORE (Combinazione con peso alpha)
    # -------------------------------------------------------------------------
    # Assicuriamoci che gli indici combacino perfettamente prima di sommare
    final_sim_df = (alpha * pt_sim_df) + ((1 - alpha) * tva_sim_df)

    # -------------------------------------------------------------------------
    # ESTRAZIONE DEI TOP CLIENTI
    # -------------------------------------------------------------------------
    if target_cid not in final_sim_df.index:
        raise ValueError(f"Il CustomerID {target_cid} non è presente nel dataset.")
    # Otteniamo i punteggi di similarità per il cliente target
    target_scores = final_sim_df.loc[target_cid]
    # Rimuoviamo il cliente stesso dalla lista (avrà similarity = 1 o vicina a 1)
    target_scores = target_scores.drop(target_cid)
    # Ordiniamo in modo decrescente (i più simili prima) e prendiamo i primi K
    top_customers = target_scores.sort_values(ascending=False)

    return top_customers

def create_prompt(df: pd.DataFrame, cid: int, top_k: int, alpha: float, level: PromptLevel, json_schema: bool) -> str:
    if not json_schema:
        raise NotImplementedError("Il caso senza json_schema non è supportato")

    intro_and_dataset = f"""your are given the following dataset of customer purchase histories:\n{df.to_string(index=False)}"""
    output = "Your output must contain only the required list of customers."

    match level:
        case PromptLevel.generic: # non ha senso
            raise NotImplementedError("generic prompt not implemented")

        case PromptLevel.formula:
            prompt = \
f"""{intro_and_dataset}

Return the {top_k} customers who are most similar to the customer {cid}. The similarity score must be computed strictly following these steps:

1. Price-Tier Affinity Representation
1.1 Classify every purchased item into one of three tiers based on UnitPrice: "Budget" (< 2.00), "Standard" (>= 2.00 and <= 5.00), and "Premium" (> 5.00).
1.2 For each customer, build a 3-dimensional vector representing the total Quantity of items purchased in each of the three tiers.
1.3 Compute the cosine similarity between customers using these 3D Price-Tier vectors.

2. TVA Behavioral Features
2.1 For each customer, compute:
    - Tenure (T): Number of days between their very first purchase and their latest purchase in the dataset.
    - Variety (V): Total number of unique StockCodes purchased.
    - Average Basket (A): Total monetary spend divided by the number of unique InvoiceNos.
2.2 Normalize these T, V, and A values so that each feature is on a comparable scale (e.g., Min-Max scaling).
2.3 Compute the cosine similarity between customers based on these normalized TVA vectors.

3. Final Score
Combine the two similarity measures as follows:
    - {alpha*100:.0f}% weight for the Price-Tier Affinity similarity
    - {(1-alpha)*100:.0f}% weight for the TVA similarity

{output}"""

        case PromptLevel.instruct:
            prompt = \
f"""{intro_and_dataset}

Return the {top_k} customers who are most similar to customer {cid} based on their 'Customer Buying Signature' (CBS).
The CBS is a score based on two behavioral pillars. You must calculate the cosine similarity between customers for each pillar and then combine them with an {alpha*100:.0f}% weight for the first pillar (and {(1-alpha)*100:.0f}% to the second one):
1. Price-Tier Affinity: Compare customers based on the total volume (quantity) of items they purchase across three price segments: Budget (under 2.00), Standard (2.00 to 5.00), and Premium (over 5.00).
2. Behavioral Profile (TVA): Compare customers based on a normalized vector of three business metrics:
    - Tenure: The total duration of their relationship as a customer (in days).
    - Variety: The breadth of their product catalog interests.
    - Average Basket: The average monetary value of their shopping carts.

{output}"""

    return prompt

@dataclass
class customer_CBS(customer_segmentation):
    name: str = "Customer CBS"

    def _build_prompt(self, df: pd.DataFrame, cid: int, top_k: int, alpha: float, level: PromptLevel, json_schema: bool) -> str:
        return create_prompt(df, cid, top_k, alpha, level, json_schema)


    def init_queries(self) -> None:
        current_seed = self.parameters.seed

        self.queries: list[CustomerSegmentationQuery] = []
        counter = 0
        ds_id = 0
        pbar = tqdm(total=self.parameters.n_queries * len(self.parameters.n_elems_per_query) * len(self.parameters.kp) * len(self.parameters.names_levels) * len(self.parameters.prompt_levels),
                    desc="Generating queries",
                    unit="query",
                    colour='green')

        for _ in range(self.parameters.n_queries):
            for elem_per_query in self.parameters.n_elems_per_query:
                if self.parameters.seed != 0:
                    random.seed(current_seed)
                k_list = [max(1, math.ceil(kp * elem_per_query)) for kp in self.parameters.kp]
                min_customers_needed = max(k_list) + 1

                temp_df = pd.DataFrame()
                while True:
                    match self.parameters.n_elems_type:
                        case NElemsType.customers:
                            cids_unique_full = self.clean_df['CustomerID'].unique().tolist()
                            selected_cids = random.sample(cids_unique_full, elem_per_query)
                            temp_df = self.clean_df[self.clean_df['CustomerID'].isin(selected_cids)]
                        case NElemsType.rows:
                            temp_df = self.generate_prompt_dataset(target_rows=elem_per_query, min_customers=min_customers_needed, seed=current_seed)

                    if len(temp_df) > self.parameters.rows_in_prompt_limit:
                        current_seed += 1
                        continue

                    df = temp_df
                    selected_cid = random.choice(df['CustomerID'].unique().tolist())
                    top_similar_df = get_top_similar_customers(df, selected_cid, self.parameters.alpha)
                    if top_similar_df.values.tolist()[0] < 0.4:
                        current_seed += 1
                        continue
                    break

                sorted_cids = top_similar_df.index.tolist()
                ground_truth_vals = top_similar_df.values.tolist()

                for k in k_list:

                    for n_level in self.parameters.names_levels:
                        if n_level != NamesLevel.fake:
                            Exception(f"Unsupported names_level {n_level} in parameters.")

                        for p_level in self.parameters.prompt_levels:
                            self.queries.append(CustomerSegmentationQuery(
                                id=counter,
                                ds_id=ds_id,
                                customer_id=selected_cid,
                                prompt=self._build_prompt(df, selected_cid, k, self.parameters.alpha, p_level, self.parameters.enforce_json_schema),
                                ground_truth=sorted_cids,
                                ground_truth_values=ground_truth_vals,
                                parameters=CustomerSegmentationParameters(k=k, prompt_level=p_level, names_level=n_level, n_elems=elem_per_query),
                                response=None,
                                evaluations=None,
                                response_json_schema=MostSimilarCustomers.model_json_schema() if self.parameters.enforce_json_schema else None,
                                parsing_failed=None,
                                parsed_response=None
                            ))
                            counter += 1
                            pbar.update(1)

                current_seed += 1
                ds_id += 1

        pbar.close()