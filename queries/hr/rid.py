import random
from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import TestParameters, Test, data_folder, PromptLevel, PartitionedQueryParameters, NamesLevel

type GTT = int
named_index_col = 'employee_id'

class FarthestEmployees(BaseModel):
    top_k: list[GTT] = Field(description=f"Ordered list of the k farthest employees (by {named_index_col}) according to the specified metric.")

def compute_rid(input_df: pd.DataFrame, target_id) -> pd.DataFrame:
    df = input_df.copy()
    target_mask = df[named_index_col] == target_id
    if target_mask.sum() == 0:
        raise ValueError(f"Impiegato con ID '{target_id}' non trovato nel DataFrame.")

    target_A = df[target_mask].iloc[0]
    # 2. Penalità Dipartimento
    # Se è diverso True diventa 1 (1 * 10 = 10). Se è uguale False diventa 0.
    penalita_dipartimento = (df['Department'] != target_A['Department']).astype(int) * 10
    # 3. Penalità Formazione Asimmetrica
    # Condizione: Campo diverso AND livello di education del target <= a quello dell'impiegato B
    campo_diverso = df['EducationField'] != target_A['EducationField']
    target_meno_qualificato = target_A['Education'] <= df['Education']
    penalita_formazione = (campo_diverso & target_meno_qualificato).astype(int) * 10
    # 4. Distanza di Esperienza
    distanza_esperienza = (df['TotalWorkingYears'] - target_A['TotalWorkingYears']).abs()
    # 5. Distanza Salariale Normalizzata
    distanza_salariale = (df['MonthlyIncome'] - target_A['MonthlyIncome']).abs() / 1000
    # 6. Somma finale (Punteggio RID)
    df['RID_Score'] = (
            penalita_dipartimento +
            penalita_formazione +
            distanza_esperienza +
            distanza_salariale
    )

    return df

@dataclass
class RIDTestParameters(TestParameters):
    names_levels: tuple[NamesLevel, ...] = tuple([NamesLevel.fake])

@dataclass
class rid(Test[GTT, RIDTestParameters]):
    name: str = 'Role Interchangeability Distance'
    name_short: str = 'RID'
    json_schema = FarthestEmployees
    full_df: DataFrame = None
    named_index_col = named_index_col

    def _load_ds(self) -> DataFrame:
        df_path = data_folder / self.family / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
        df = pd.read_csv(df_path)
        columns_to_keep = ['Age','Gender','Department','EducationField','Education','TotalWorkingYears','MonthlyIncome','MonthlyRate']
        return df[columns_to_keep].rename_axis(self.named_index_col).reset_index()

    def _sample_for_query(self, current_seed: int, df: DataFrame, elem_per_query: int) -> DataFrame:
        return df.sample(n=elem_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)

    def _select_query_target(self, current_seed, real_df: DataFrame, anon_df) -> tuple[GTT|None, GTT|None]:
        target_employee = random.choice(real_df[named_index_col].tolist())
        return target_employee, target_employee

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], list[float]]:
        gt_df = compute_rid(df, target)
        target_mask = gt_df[named_index_col] == target
        gt_df = gt_df[~target_mask].sort_values('RID_Score', ascending=False)
        return gt_df[named_index_col].tolist(), gt_df['RID_Score'].tolist()

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"Your are given a dataset of employees:"
        output = f"Your output must contain only the required list of {named_index_col}."
        match q_params.prompt_level:
            case PromptLevel.generic:
                raise NotImplementedError("generic non ancora implementato")
            case PromptLevel.instruct:
                instruction = \
                    f"""Return the {q_params.k} least interchangeable employees to the target employee '{target}' from the provided dataset, using the Role Interchangeability Distance (RID). Calculate the RID by summing four components:
    1.	A 10-point penalty if their 'Department' values are different.
    2.	An education penalty: 0 if they share the same 'EducationField' OR if the target employee's 'Education' level (numeric) is strictly greater than the candidate's. Otherwise, add 10 points.
    3.	The absolute difference in their 'TotalWorkingYears'.
    4.	The absolute difference in their 'MonthlyIncome', divided by 1000."""
            case PromptLevel.formula:
                instruction = \
                    f"""Return the {q_params.k} least interchangeable employees to the target employee '{target}', based on the Role Interchangeability Distance (RID). The RID between two employees can be computed as follows:
    1.	Initialize a variable 'total_rid' to 0.
    2.	If their 'Department' differs, add 10 to 'total_rid'.
    3.	If the 'EducationField' differs, compare their numeric 'Education' values. If target employee's 'Education' value is less than or equal to other employee's one, add 10 to 'total_rid'.
    4.	Add the absolute difference between the two employees' 'TotalWorkingYears' to 'total_rid'.
    5.	Add the absolute difference between the two employees' 'MonthlyIncome', divided this by 1000, to 'total_rid'.
    6.	The final value of 'total_rid' is the RID score."""

        return job, f"{instruction}\n{output}"