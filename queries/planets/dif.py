import random
import sys
import os

import numpy as np

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)
from esi import *

def _build_prompt(df: DataFrame, target: str, prompt_level: PromptLevel, top_k: int, json_schema: bool) -> str:
    if not json_schema:
        NotImplementedError("Non è implementato il caso senza json_schema per esi_free_score.")

    job = "You are given a dataset of planets, with various attributes:"
    output = "Your output must contain only the required list of planets."

    match prompt_level:
        case PromptLevel.generic:
            raise NotImplementedError("PromptLevel.generic is not implemented for esi_free_score.")
        case PromptLevel.instruct:
            prompt = f"Return the sorted list of top {top_k} most similar planets to planet '{target}' using only the 'Density-Irradiance Factor' (DIF). This factor is computed as the planet's average density (expressed in Earth units, assuming a spherical shape) and multiplying it by the square root of the incident stellar flux."
        case PromptLevel.formula:
            prompt = \
f"""Return the sorted list of top {top_k} most similar planets to planet '{target}' using only the 'Density-Irradiance Factor' (DIF). THe DIF can be computed as follows:

DIF = (M / R^3) * sqrt(S)

Where:
- 'M' is the planet's Mass;
- 'R' is the planet's Radius;
- 'S' is the planet's Flux."""

    return f"""{job}
{df.to_string(index=False)}

{prompt}

{output}"""

def compute_ground_truth(df: DataFrame, target_planet: str) -> DataFrame:
    result_df = df.copy()
    result_df['DIF'] = (result_df['Mass (Me)'] / (result_df['Radius (Re)'] ** 3)) * np.sqrt(result_df['Flux (Se)'])
    mask = result_df['Name'] == target_planet
    target_dif = result_df.loc[result_df['Name'] == target_planet, 'DIF'].values[0]

    out = (
        result_df.assign(diff_invers=1 / (1 + (result_df['DIF'] - target_dif).abs()))
        .loc[~mask]
        .sort_values("diff_invers", ascending=False)
        .loc[:, ['Name', 'DIF', "diff_invers"]]
    )
    return out

@dataclass
class dif(esi):
    name: str = "DIF"

    def init_queries(self) -> None:
        for p_per_query in self.parameters.planets_per_query:
            if math.comb(len(self.clean_df), p_per_query) < self.parameters.n_queries:
                raise ValueError(f"Not enough unique combinations of planets to generate the requested number of queries (planets per query: {p_per_query}).")

        self.queries = []
        current_seed = self.parameters.seed
        counter = 0
        ds_id = 0
        pbar = tqdm(total=self.parameters.n_queries*len(self.parameters.planets_per_query)*len(self.parameters.kp)*len(self.parameters.prompt_levels)*len(self.parameters.names_levels),
                    desc="Generating queries",
                    unit="query",
                    colour='green')

        for _ in range(self.parameters.n_queries):
            for p_per_query in self.parameters.planets_per_query:
                if self.parameters.seed != 0:
                    random.seed(self.parameters.seed)

                selected_planets = self.clean_df.sample(n=p_per_query, replace=False, random_state=current_seed if self.parameters.seed != 0 else None)

                for kp in self.parameters.kp:
                    k = max(1, math.ceil(kp * p_per_query))

                    for planet_name_mod in self.parameters.names_levels:
                        match planet_name_mod:
                            case NamesLevel.real:
                                q_df = selected_planets
                            case NamesLevel.fake:
                                fake_selected_planets = selected_planets.copy()
                                fake_selected_planets['Name'] = "Planet " + fake_selected_planets.index.astype(str)
                                q_df = fake_selected_planets

                        target_planet = random.choice(list(q_df['Name']))
                        ground_truth_df = compute_ground_truth(q_df, target_planet)
                        prompt_df = q_df.drop(columns=['ESI'], inplace=False)
                        ground_truth = [{'planet_name': row['Name'],
                                         'esi': row['diff_invers']} for _, row # chiamato ancora 'esi' per non dover modificare la funzione di valutaiozne delle query
                                        in ground_truth_df.iterrows()]

                        for prompt_level in self.parameters.prompt_levels:
                            match prompt_level:
                                case PromptLevel.generic:
                                    response_schema = MostSimilarPlanets.model_json_schema()
                                case PromptLevel.instruct | PromptLevel.formula:
                                    response_schema = MostSimilarPlanetsScore.model_json_schema()
                            query = PlanetQuery(
                                id=counter,
                                ds_id=ds_id,
                                prompt=_build_prompt(prompt_df, target_planet, prompt_level, k, self.parameters.enforce_json_schema),
                                parameters=PlanetQueryParameters(k=k, prompt_level=prompt_level, names_level=planet_name_mod, n_elems=p_per_query),
                                ground_truth=ground_truth,
                                response=None,
                                evaluations=None,
                                response_json_schema=MostSimilarPlanets.model_json_schema() if self.parameters.enforce_json_schema else None,
                                parsing_failed=None,
                                parsed_response=None
                            )
                            self.queries.append(query)
                            counter += 1
                            pbar.update(1)

                current_seed = current_seed + 1
                ds_id += 1

        pbar.close()