import math
import string
from dataclasses import field, dataclass

import pandas as pd
from openpyxl.styles.builtins import total
from pandas import DataFrame
from pydantic import BaseModel, Field
from tqdm import tqdm

from experiments.run_type import RunType
from queries.test import TestParameters, Test, data_folder, NamesLevel, Query, QueryParameters, PromptLevel

named_index_col = 'smart_id'
scoring_cols = ['Nominal Battery Capacity', 'CPU Clock', 'Mass', 'Memory Capacity', 'Display Diagonal']

class BestSmartphones(BaseModel):
    top_k: list[int] = Field(description=f"Ordered list of the k best smartphones (by {named_index_col}) according to the specified metric.")

def anonymize_df(df: DataFrame) -> DataFrame:
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

def compute_ground_truth(input_df: DataFrame) -> tuple[list[str], list[float]]:
    df = input_df.copy()
    df['HES'] = (
            (df['Nominal Battery Capacity'] / (df['CPU Clock'] * df['Mass'])) * 100000
            + df['Memory Capacity']
            - (df['Display Diagonal'] * 2)
    )
    df = df.sort_values(by='HES', ascending=False)

    return df[named_index_col].to_list(), df['HES'].to_list()

def create_prompt(df: DataFrame, prompt_level: PromptLevel, top_k: int, json_schema: bool, run_type: RunType) -> str:
    if not json_schema:
        raise NotImplementedError("il caso senza json_schema non è più supportato")

    job = f"You are given a dataset of smartphones specs:\n{df.to_markdown(index=False)}"
    output = f"Your output must contain only the required list of {named_index_col}."

    match run_type:
        case RunType.LOTUS:
            raise NotImplementedError("LOTUS non è ancora implementato")
        case RunType.DIRECT:
            match prompt_level:
                case PromptLevel.generic:
                    raise NotImplementedError("generic non ancora implementato")
                case PromptLevel.instruct:
                    instruction = \
            f"""Return the {top_k} smartphone models by '{named_index_col}' from the provided dataset that have the highest Hardware Efficiency Score (HES).
To calculate the HES for each device: divide its 'Nominal Battery Capacity' by the product of its 'CPU Clock' and 'Mass'. Multiply this result by 100,000 to normalize the scale. Then, add the 'Memory Capacity' and subtract twice the value of the 'Display Diagonal'.
Rank the smartphones in descending order based on their HES score."""
                case PromptLevel.formula:
                    instruction = \
            f"""Return the {top_k} smartphone models by '{named_index_col}' from the provided dataset that have the highest Hardware Efficiency Score (HES).
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

@dataclass
class SmartphoneTestParameters(TestParameters):
    items_per_query: list[int] = field(default_factory=lambda: [50])

@dataclass
class hw_eff_score(Test[SmartphoneTestParameters]):
    name: str = 'hw_eff_score'
    name_short: str = 'HES'
    json_schema = BestSmartphones
    full_df: DataFrame = None
    named_index_col = named_index_col

    def load_csv(self) -> None:
        df_path = data_folder / self.family / 'mobile.csv'
        df = pd.read_csv(df_path, index_col=0)
        df = df.dropna(subset=scoring_cols)
        self.full_df = df.rename_axis(self.named_index_col).reset_index()

    def init_queries(self) -> None:
        if self.full_df is None:
            self.load_csv()

        self.queries = []
        current_seed = self.parameters.seed
        counter = 0
        ds_id = 0
        pbar = tqdm(total=self.parameters.n_queries*len(self.parameters.items_per_query)*len(self.parameters.kp)*len(self.parameters.prompt_levels)*len(self.parameters.names_levels),
                    desc="Generating queries",
                    unit='query',
                    colour='green')

        for _ in range(self.parameters.n_queries):
            for sp_per_query in self.parameters.items_per_query:
                selected_sp = self.full_df.sample(n=sp_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)

                for kp in self.parameters.kp:
                    k = max(1, math.ceil(kp * sp_per_query))

                    for name_mode in self.parameters.names_levels:
                        match name_mode:
                            case NamesLevel.real:
                                q_df = selected_sp
                            case NamesLevel.fake:
                                q_df = anonymize_df(selected_sp)
                        ground_truth_ids, ground_truth_scores = compute_ground_truth(q_df)
                        prompt_df = q_df

                        for prompt_level in self.parameters.prompt_levels:
                            query = Query(
                                id=counter,
                                ds_id=ds_id,
                                prompt=create_prompt(df=prompt_df, prompt_level=prompt_level, top_k=k, json_schema=self.parameters.enforce_json_schema, run_type=self.run_type),
                                prompt_df=prompt_df if self.run_type==RunType.LOTUS else None,
                                parameters=QueryParameters(k=k, prompt_level=prompt_level, names_level=name_mode, n_elems=sp_per_query),
                                ground_truth=ground_truth_ids,
                                ground_truth_scores=ground_truth_scores,
                                response_json_schema=self.json_schema.model_json_schema() if self.parameters.enforce_json_schema else None
                            )
                            self.queries.append(query)
                            counter += 1
                            pbar.update(1)

                current_seed = current_seed + 1
                ds_id += 1

        pbar.close()

    def prepare_queries_for_direct(self) -> None:
        print('loading dataset...')
        self.load_csv()
        print('initializing queries...')
        self.init_queries()
