import string
from dataclasses import field, dataclass

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from experiments.run_type import RunType
from queries.test import TestParameters, Test, data_folder, PromptLevel

named_index_col = 'smart_id'
scoring_cols = ['Nominal Battery Capacity', 'CPU Clock', 'Mass', 'Memory Capacity', 'Display Diagonal']

class BestSmartphones(BaseModel):
    top_k: list[int] = Field(description=f"Ordered list of the k best smartphones (by {named_index_col}) according to the specified metric.")

@dataclass
class SmartphoneTestParameters(TestParameters):
    elems_per_query: list[int] = field(default_factory=lambda: [50])

@dataclass
class hw_eff_score(Test[SmartphoneTestParameters]):
    name: str = 'hw_eff_score'
    name_short: str = 'HES'
    json_schema = BestSmartphones
    full_df: DataFrame = None
    named_index_col = named_index_col

    def _load_ds(self) -> DataFrame:
        df_path = data_folder / self.family / 'mobile.csv'
        df = pd.read_csv(df_path, index_col=0)
        df = df.dropna(subset=scoring_cols)
        return df.rename_axis(self.named_index_col).reset_index()

    def _anonymize_query_df(self, df: DataFrame) -> DataFrame:
        result_df = df.copy()
        # 2. Anonimizzazione dei Modelli ('model 1', 'model 2', ecc.)
        # Estraiamo i modelli univoci e creiamo un dizionario di associazione
        modelli_univoci = result_df['Model'].unique()
        mappa_modelli = {modello: f"model {i+1}" for i, modello in enumerate(modelli_univoci)}

        # 3. Anonimizzazione dei Brand ('brand A', 'brand B', ecc.)
        # Usiamo string.ascii_uppercase per avere l'alfabeto (A, B, C...)
        brand_univoci = result_df['Brand'].unique()
        alfabeto = string.ascii_uppercase
        mappa_brand = {brand: f"brand {alfabeto[i % 26]}" for i, brand in enumerate(brand_univoci)}

        # 4. Applichiamo le mappature al DataFrame sostituendo le colonne originali
        result_df['Brand'] = result_df['Brand'].map(mappa_brand)
        result_df['Model'] = result_df['Model'].map(mappa_modelli)

        return result_df

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        return df.sample(n=elem_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)

    def _build_ground_truth(self, df: DataFrame, target: str|int|None) -> tuple[list[str | int], list[float]]:
        gt_df = df.copy()
        gt_df['HES'] = (
                (gt_df['Nominal Battery Capacity'] / (gt_df['CPU Clock'] * gt_df['Mass'])) * 100000
                + gt_df['Memory Capacity']
                - (gt_df['Display Diagonal'] * 2)
        )
        gt_df = gt_df.sort_values(by='HES', ascending=False)
        return gt_df[named_index_col].to_list(), gt_df['HES'].to_list()

    def _create_prompt(self, df: DataFrame, target: str|int|None, k: int, prompt_level: PromptLevel) -> str:
        if not self.parameters.enforce_json_schema:
            raise NotImplementedError("il caso senza json_schema non è più supportato")
        job = f"You are given a dataset of smartphones specs:\n{df.to_markdown(index=False)}"
        output = f"Your output must contain only the required list of {named_index_col}."
        match self.run_type:
            case RunType.LOTUS:
                raise NotImplementedError("LOTUS non è ancora implementato")
            case RunType.DIRECT:
                match prompt_level:
                    case PromptLevel.generic:
                        raise NotImplementedError("generic non ancora implementato")
                    case PromptLevel.instruct:
                        instruction = \
                            f"""Return the {k} smartphone models by '{named_index_col}' from the provided dataset that have the highest Hardware Efficiency Score (HES).
    To calculate the HES for each device: divide its 'Nominal Battery Capacity' by the product of its 'CPU Clock' and 'Mass'. Multiply this result by 100,000 to normalize the scale. Then, add the 'Memory Capacity' and subtract twice the value of the 'Display Diagonal'.
    Rank the smartphones in descending order based on their HES score."""
                    case PromptLevel.formula:
                        instruction = \
                            f"""Return the {k} smartphone models by '{named_index_col}' from the provided dataset that have the highest Hardware Efficiency Score (HES).
    The Hardware Efficiency Score (HES) can be calculate with the following steps:
        1.	Calculate the product of 'CPU Clock' and 'Mass' (call this 'power_weight').
        2.	Divide 'Nominal Battery Capacity' by 'power_weight' (call this 'base_efficiency').
        3.	Multiply 'base_efficiency' by 100000 (call this 'scaled_efficiency').
        4.	Add the 'Memory Capacity' to 'scaled_efficiency'.
        5.	Multiply the 'Display Diagonal' by 2.
        6.	Subtract the result of step 6 from the result of step 5. This final number is the HES for the current device.
    Rank the smartphones in descending order based on their HES score."""
                prompt = f"{job}\n{instruction}\n{output}"
        return prompt
