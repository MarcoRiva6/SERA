import random
import sys
import os

import numpy as np


cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)
from esi import *

@dataclass
class dif(esi):
    name: str = "DIF"
    name_short: str = "DIF"

    def _select_query_target(self, current_seed, real_df: DataFrame, anon_df: DataFrame):
        chosen = random.choice(range(real_df.shape[0]))
        return real_df.iloc[chosen]['Name'], anon_df.iloc[chosen]['Name']

    def _build_ground_truth(self, df: DataFrame, target=None) -> tuple[list[str | int], list[float]]:
        assert target is not None
        result_df = df.copy()
        result_df['DIF'] = (result_df['Mass (Me)'] / (result_df['Radius (Re)'] ** 3)) * np.sqrt(result_df['Flux (Se)'])
        mask = result_df['Name'] == target
        target_dif = result_df.loc[result_df['Name'] == target, 'DIF'].values[0]
        out = (
            result_df.assign(diff_invers=1 / (1 + (result_df['DIF'] - target_dif).abs()))
            .loc[~mask]
            .sort_values("diff_invers", ascending=False)
            .loc[:, ['Name', 'DIF', "diff_invers"]]
        )
        ground_truth_df = out
        return ground_truth_df['Name'].tolist(), ground_truth_df['diff_invers'].tolist()

    def _create_prompt(self, df: DataFrame, target: str|int|None, k: int, prompt_level: PromptLevel) -> str:
        # shadowing voluto
        index_name = "Planet"
        if not self.parameters.enforce_json_schema:
            NotImplementedError("Non è implementato il caso senza json_schema per esi_free_score.")

        job = "You are given a dataset of planets, with various attributes:"
        output = "Your output must contain only the required list of planets."
        attributes_without_index = ", ".join([f"{{{col}}}: {val}" for col, val in df[df[index_name] == target].iloc[0].items() if col != index_name])

        match self.run_type:
            case RunType.LOTUS:
                match prompt_level:
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

            case RunType.DIRECT:
                match prompt_level:
                    case PromptLevel.generic:
                        raise NotImplementedError("PromptLevel.generic is not implemented for esi_free_score.")
                    case PromptLevel.instruct:
                        instruction = f"Return the sorted list of top {k} most similar planets to planet '{target}' using only the 'Density-Irradiance Factor' (DIF). This factor is computed as the planet's average density (expressed in Earth units, assuming a spherical shape) and multiplying it by the square root of the incident stellar flux."
                    case PromptLevel.formula:
                        instruction = \
                            f"""Return the sorted list of top {k} most similar planets to planet '{target}' using only the 'Density-Irradiance Factor' (DIF). The DIF can be computed as follows:

DIF = (M / R^3) * sqrt(S)

Where:
- 'M' is the planet's Mass;
- 'R' is the planet's Radius;
- 'S' is the planet's Flux."""
                prompt = f"""{job}
    {df.to_string(index=False)}
    
    {instruction}
    
    {output}"""

        return prompt
