import math
from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import TestParameters, Test, data_folder, PromptLevel, sample_ds_interesting, \
    PartitionedQueryParameters, NamesLevel

type GTT = str

class MostConsumingFamilies(BaseModel):
    top_k: list[GTT] = Field(description="")

@dataclass
class SPATestParameters(TestParameters):
    names_levels: tuple[NamesLevel, ...] = tuple([NamesLevel.fake])

@dataclass
class spa(Test[GTT, SPATestParameters]):
    name: str = "Sustained Peak Average"
    name_short: str = "SPA"
    json_schema = MostConsumingFamilies
    named_index_col = "LCLid"

    def _load_ds(self) -> DataFrame:
        folder = data_folder / self.family
        return pd.read_csv(folder / 'block_0.csv')

    def _prepare_df(self, df: DataFrame) -> DataFrame:
        colonne_hh = [f'hh_{i}' for i in range(48)]
        dati_mezzore = df[colonne_hh].to_numpy()
        # - reshape(-1, 8, 6) prende le 48 colonne e le divide in 8 gruppi da 6 per ogni riga
        # - sum(axis=2) somma i 6 valori (le 3 ore) all'interno di ogni gruppo
        dati_aggregati = dati_mezzore.reshape(-1, 8, 6).sum(axis=2)
        # Creiamo la nuova colonna.
        # .tolist() trasforma la matrice (N, 8) in una lista di liste, perfetta per la singola colonna
        df['3h_intervals'] = dati_aggregati.tolist()
        # Manteniamo solo le 3 colonne finali
        return df[['LCLid', 'day', '3h_intervals']]

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        k_list = []
        for kp in self.parameters.kp:
            if kp >= 1:
                k_list.append(int(kp))
            else:
                k_list.append(max(1, math.ceil(kp * elem_per_query)))
        return sample_ds_interesting(df, length=elem_per_query, min_unique=max(k_list)+3, key=self.named_index_col)

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], list[float]]:
        df_calc = df.copy()
        # Daily Peak Average
        # sorted(x, reverse=True)[:3] prende i 3 valori più alti dell'array.
        # sum(...) / 3 ne calcola la media aritmetica.
        df_calc['Daily_Peak_Average'] = df_calc['3h_intervals'].apply(
            lambda x: sum(sorted(x, reverse=True)[:3]) / 3
        )
        # Calcolo inter-riga (Group By): Calcoliamo l'SPA per utente
        # Raggruppiamo per LCLid e facciamo la media di tutti i suoi Daily Peak Average
        df_spa = (
            df_calc.groupby(self.named_index_col)['Daily_Peak_Average']
            .mean()
            .reset_index(name='SPA') # Rinominare direttamente la colonna risultante
        )
        df_classifica = df_spa.sort_values(by='SPA', ascending=False).reset_index()

        return df_classifica[self.named_index_col].tolist(), df_classifica['SPA'].tolist()

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given a dataset of household electricity consumptions:"
        output = f"Your output must contain only the required list of household ids ({self.named_index_col})."

        match q_params.prompt_level:
            case PromptLevel.generic:
                raise NotImplementedError()
            case PromptLevel.instruct:
                instruction = f"""Return the {q_params.k} {self.named_index_col} who have the highest Sustained Peak Average (SPA) from the provided dataset.
To calculate the SPA for a user: first, evaluate the '3h_intervals' array for every row associated with them. For each array, identify the 3 highest numerical values and calculate their arithmetic mean (this is the Daily Peak Average). Then, group by '{self.named_index_col}' and finally calculate the overall mean of these Daily Peak Averages across all recorded days for each household."""
            case PromptLevel.formula:
                instruction = f"""Return the {q_params.k} {self.named_index_col} who have the highest Sustained Peak Average (SPA) from the provided dataset.
The Sustained Peak Average (SPA) score for a household can be computed as follows:
    1. Initialize an empty list called 'daily_averages'.
    2. for each of the household's rows, append the arithmetic mean of the 3 highest values of the array contained in the '3h_intervals' column to the 'daily_averages' list.
    3. Compute the arithmetic mean of all values within the 'daily_averages' list: this value is the final SPA score for the household."""

        return job, f"{instruction}\n\n{output}"