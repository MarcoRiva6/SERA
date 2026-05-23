import sys
import os

import numpy as np

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)
from esi import *

class TopDIFPlanets(BaseModel):
    top_k: list[GTT] = Field(description="The ordered list of the planets with the highest DIF score.")

@dataclass
class dif(esi):
    name: str = "DIF"
    name_short: str = "DIF"
    json_schema = TopDIFPlanets
    prompt_scoring_cols = ['Mass (Me)', 'Radius (Re)', 'Flux (Se)']

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], GroundTruthScoreList]:
        result_df = df.copy()
        result_df['DIF'] = (result_df['Mass (Me)'] / (result_df['Radius (Re)'] ** 3)) * np.sqrt(result_df['Flux (Se)'])
        ground_truth_df = result_df.sort_values(by=['DIF'], ascending=False)
        return ground_truth_df[self.named_index_col].tolist(), ground_truth_df['DIF'].tolist()

    def _create_prompt_lotus(self, df: DataFrame, target: GTT | None, q_params: QueryParameters) -> str:
        raise NotImplementedError("il prompt di lotus non è stato aggiornato dopo la modifica del test")
        # shadowing voluto
        index_name = "Planet"
        attributes_without_index = ", ".join([f"{{{col}}}: {val}" for col, val in df[df[index_name] == target].iloc[0].items() if col != index_name])

        match q_params.prompt_level:
            case PromptLevel.generic:
                raise NotImplementedError("PromptLevel.generic is not implemented for esi_free_score.")
            case PromptLevel.instruct:
                prompt = f"Return the most similar {{{index_name}}} to planet '{target}' (whose attributes are: {attributes_without_index}) using only the 'Density-Irradiance Factor' (DIF). This factor is computed as the planet's average density (expressed in Earth units, assuming a spherical shape) and multiplying it by the square root of the incident stellar flux."
            case PromptLevel.formula:
                prompt = \
                    f"""Return the most similar {{{index_name}}} to planet '{target}' (whose attributes are: {attributes_without_index}) using only the 'Density-Irradiance Factor' (DIF). The DIF can be computed as follows:

DIF = (M / R^3) * sqrt(S)

Where:
- 'M' is the planet's Mass;
- 'R' is the planet's Radius;
- 'S' is the planet's Flux."""

        return prompt

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = "You are given a dataset of planets, with various attributes:"
        output = "Your output must contain only the required list of planets."

        match q_params.prompt_level:
            case PromptLevel.generic:
                raise NotImplementedError("PromptLevel.generic is not implemented for esi_free_score.")
            case PromptLevel.instruct:
                instruction = f"Return the sorted list of the top {q_params.k} planets having the highest 'Density-Irradiance Factor' (DIF). This factor is computed as the planet's average density (expressed in Earth units, assuming a spherical shape) and multiplying it by the square root of the incident stellar flux."
            case PromptLevel.formula:
                instruction = \
                    f"""Return the sorted list of top {q_params.k} planets having the highest 'Density-Irradiance Factor' (DIF). The DIF can be computed as follows:

DIF = (M / R^3) * sqrt(S)

Where:
- 'M' is the planet's Mass;
- 'R' is the planet's Radius;
- 'S' is the planet's Flux."""

        return job, f"{instruction}\n\n{output}"
