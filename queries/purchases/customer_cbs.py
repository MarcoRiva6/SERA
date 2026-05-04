import os
import sys

from queries.test import QueryParameters, PartitionedQueryParameters

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

@dataclass
class customer_CBS(customer_segmentation):
    name: str = "Customer CBS"
    name_short = "TVA"

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str | None]:
        job = "you are given the following dataset of customer purchase histories:"
        output = "Your output must contain only the required list of customers."

        match q_params.prompt_level:
            case PromptLevel.generic:
                raise NotImplementedError("generic prompt not implemented")

            case PromptLevel.formula:
                instruction = \
                    f"""Return the {q_params.k} customers who are most similar to the customer {target}. The similarity score must be computed strictly following these steps:
    
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
        - {self.parameters.alpha*100:.0f}% weight for the Price-Tier Affinity similarity
        - {(1-self.parameters.alpha)*100:.0f}% weight for the TVA similarity"""

            case PromptLevel.instruct:
                instruction = \
                    f"""Return the {q_params.k} customers who are most similar to customer {target} based on their 'Customer Buying Signature' (CBS).
    The CBS is a score based on two behavioral pillars. You must calculate the cosine similarity between customers for each pillar and then combine them with an {self.parameters.alpha*100:.0f}% weight for the first pillar (and {(1-self.parameters.alpha)*100:.0f}% to the second one):
    1. Price-Tier Affinity: Compare customers based on the total volume (quantity) of items they purchase across three price segments: Budget (under 2.00), Standard (2.00 to 5.00), and Premium (over 5.00).
    2. Behavioral Profile (TVA): Compare customers based on a normalized vector of three business metrics:
        - Tenure: The total duration of their relationship as a customer (in days).
        - Variety: The breadth of their product catalog interests.
        - Average Basket: The average monetary value of their shopping carts."""

        return job, f"{instruction}\n\n{output}"

    def _init_query(self, seed: int, df: DataFrame, elem_per_query: int) -> tuple[tuple[DataFrame, DataFrame, GTT|None, list[GTT], GroundTruthScoreList],tuple[DataFrame, DataFrame, GTT|None, list[GTT], GroundTruthScoreList]]:
        k_list = [max(1, math.ceil(kp * elem_per_query)) for kp in self.parameters.kp]
        min_customers_needed = max(k_list) + 1 #+1: perché se ne chiediamo K simili ad 1 significa che ce ne devono essere K+1

        temp_df = pd.DataFrame()
        while True:
            match self.parameters.n_elems_type:
                case NElemsType.customers:
                    cids_unique_full = df[self.named_index_col].unique().tolist()
                    selected_cids = random.sample(cids_unique_full, elem_per_query)
                    temp_df = df[df[self.named_index_col].isin(selected_cids)]
                case NElemsType.rows:
                    temp_df = sample_ds_interesting(df, length=elem_per_query, min_unique=min_customers_needed+3, key=self.named_index_col)
                    #temp_df = self.generate_prompt_dataset(df=df, target_rows=elem_per_query,
                    #                                       min_customers=min_customers_needed, seed=seed)

            if len(temp_df) > self.parameters.rows_in_prompt_limit:
                continue

            df = temp_df
            selected_cid = random.choice(df[self.named_index_col].unique().tolist())
            top_similar_df = get_top_similar_customers(df, selected_cid, self.parameters.alpha)
            if top_similar_df.values.tolist()[0] < 0.3:
                continue
            break

        sorted_cids = top_similar_df.index.tolist()
        ground_truth_vals = top_similar_df.values.tolist()

        result = (df, df, selected_cid, sorted_cids, ground_truth_vals)
        return result, result

    # non è più utilizzato
    def generate_prompt_dataset(self, df, target_rows: int, min_customers: int, seed: int) -> DataFrame:
        """
        Estrae un sotto-dataset di esattamente 'target_rows' righe e almeno
        'min_customers' clienti, garantendo che le metriche comportamentali siano calcolabili.
        Viene utilizzato "random", di cui NON viene ri-settato il seed, che invece viene utilizzato per le funzioni random
        di pandas.
        """
        # 3. Creiamo un "bacino" di clienti ideali per il test.
        # Devono avere almeno 2 fatture (per calcolare la Tenure) e non troppe righe
        # (altrimenti un solo cliente occuperebbe tutto il test da 30 righe).
        max_rows_per_cust = (target_rows // min_customers) + 3

        clienti_idonei = self.stats_df[
            (self.stats_df['NumInvoices'] >= 2) &
            (self.stats_df['NumRows'] >= 2) &
            (self.stats_df['NumRows'] <= max_rows_per_cust)
            ].index.tolist()

        if len(clienti_idonei) < min_customers:
            raise ValueError("Non ci sono abbastanza clienti con questi requisiti nel dataset.")

        # 4. Ricerca della combinazione esatta (ciclo veloce basato sulla casualità)
        # Poiché il dataset è enorme, troverà la combinazione in frazioni di secondo.
        while True:
            # Decidiamo quanti clienti pescare (tra il minimo richiesto e il massimo possibile)
            num_clienti_da_pescare = random.randint(min_customers, target_rows // 2)
            # Peschiamo casualmente i clienti dal nostro bacino idoneo
            clienti_scelti = random.sample(clienti_idonei, num_clienti_da_pescare)
            # Contiamo quante righe totali occupano questi clienti
            righe_totali = self.stats_df.loc[clienti_scelti, 'NumRows'].sum()
            # Se la somma fa ESATTAMENTE il numero di righe che vogliamo (es. 30), ci fermiamo!
            if righe_totali == target_rows:
                # Estraiamo le righe reali di questi clienti dal dataset originale
                df_finale = df[df['CustomerID'].isin(clienti_scelti)].copy()
                # Mischiamo le righe in modo casuale
                df_finale = df_finale.sample(frac=1, random_state=seed).reset_index(drop=True)
                return df_finale
