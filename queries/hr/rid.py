import math
import random
from dataclasses import dataclass, field

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field
from tqdm import tqdm

from experiments.run_type import RunType
from queries.test import TestParameters, Test, data_folder, NamesLevel, Query, QueryParameters, PromptLevel

named_index_col = 'employee_id'

class ClosestEmployee(BaseModel):
    top_k: list[int] = Field(description=f"Ordered list of the k closest employees (by {named_index_col}) according to the specified metric.")

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
    df['RID_Score'] = (1 - 1/( 1 +
            penalita_dipartimento +
            penalita_formazione +
            distanza_esperienza +
            distanza_salariale)
    )

    return df

def compute_ground_truth(input_df: DataFrame, target) -> tuple[list[str], list[float]]:
    gt_df = compute_rid(input_df, target)
    target_mask = gt_df[named_index_col] == target
    gt_df = gt_df[~target_mask].sort_values('RID_Score', ascending=False)

    return gt_df[named_index_col].tolist(), gt_df['RID_Score'].tolist()


def create_prompt(df: DataFrame, target, prompt_level: PromptLevel, top_k: int, json_schema: bool, run_type: RunType) -> str:
    if not json_schema:
        raise NotImplementedError("il caso senza json_schema non è più supportato")

    job = f"Your are given a dataset of employees:\n{df.to_markdown(index=False)}"
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
            f"""Return the {top_k} most interchangeable employees to the target employee '{target}' from the provided dataset, using the Role Interchangeability Distance (RID). Calculate the RID by summing four components:
	1.	A 10-point penalty if their 'Department' values are different.
	2.	An education penalty: 0 if they share the same 'EducationField' OR if the target employee's 'Education' level (numeric) is strictly greater than the candidate's. Otherwise, add 10 points.
	3.	The absolute difference in their 'TotalWorkingYears'.
	4.	The absolute difference in their 'MonthlyIncome', divided by 1000."""
                case PromptLevel.formula:
                    instruction = \
            f"""Return the {top_k} most interchangeable employees to the target employee '{target}', based on the Role Interchangeability Distance (RID). The RID between two employees can be computed as follows:
	1.	Initialize a variable 'total_rid' to 0.
	2.	If their 'Department' differs, add 10 to 'total_rid'.
	3.	If the 'EducationField' differs, compare their numeric 'Education' values. If target employee's 'Education' value is less than or equal to other employee's one, add 10 to 'total_rid'.
	4.	Add the absolute difference between the two employees' 'TotalWorkingYears' to 'total_rid'.
	5.	Add the absolute difference between the two employees' 'MonthlyIncome', divided this by 1000, to 'total_rid'.
	6.	The final value of 'total_rid' is the RID score."""

            prompt = f"{job}\n{instruction}\n{output}"

    return prompt

@dataclass
class HRTestParameters(TestParameters):
    items_per_query: list[int] = field(default_factory=lambda: [50])

@dataclass
class rid(Test[HRTestParameters]):
    name: str = 'Role Interchangeability Distance'
    name_short: str = 'RID'
    json_schema = ClosestEmployee
    full_df: DataFrame = None
    named_index_col = named_index_col

    def load_csv(self) -> None:
        df_path = data_folder / self.family / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
        df = pd.read_csv(df_path)
        columns_to_keep = ['Age','Gender','Department','EducationField','Education','TotalWorkingYears','MonthlyIncome','MonthlyRate']
        self.full_df = df[columns_to_keep].rename_axis(self.named_index_col).reset_index()

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
            for empl_per_query in self.parameters.items_per_query:
                if self.parameters.seed != 0:
                    random.seed(current_seed)

                sampled_empl = self.full_df.sample(n=empl_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)
                target_employee = random.choice(sampled_empl[named_index_col].tolist())

                for kp in self.parameters.kp:
                    k = max(1, math.ceil(kp * empl_per_query))

                    for name_mode in self.parameters.names_levels:
                        match name_mode:
                            case NamesLevel.real:
                                q_df = sampled_empl
                            case NamesLevel.fake:
                                raise NotImplementedError("il dataset è incompatibile")
                        ground_truth_ids, ground_truth_scores = compute_ground_truth(q_df, target_employee)
                        prompt_df = q_df

                        for prompt_level in self.parameters.prompt_levels:
                            query = Query(
                                id=counter,
                                ds_id=ds_id,
                                prompt=create_prompt(df=prompt_df, target=target_employee, prompt_level=prompt_level, top_k=k, json_schema=self.parameters.enforce_json_schema, run_type=self.run_type),
                                prompt_df=prompt_df if self.run_type==RunType.LOTUS else None,
                                parameters=QueryParameters(k=k, prompt_level=prompt_level, names_level=name_mode, n_elems=empl_per_query),
                                ground_truth=ground_truth_ids,
                                ground_truth_scores=ground_truth_scores,
                                response_json_schema=self.json_schema.model_json_schema() if self.parameters.enforce_json_schema else None
                            )
                            self.queries.append(query)
                            counter += 1
                            pbar.update(1)

                current_seed += 1
                ds_id += 1

        pbar.close()

    def prepare_queries_for_direct(self) -> None:
        print('loading dataset...')
        self.load_csv()
        print('initializing queries...')
        self.init_queries()