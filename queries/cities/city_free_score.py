import sys
import os
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
if cartella_corrente not in sys.path:
    sys.path.append(cartella_corrente)
from global_liveability import *

ignoring_column = 'Education'

def redefine_ranking(df: DataFrame):
    result = df.copy()
    SC = scoring_cols.copy()
    SC.remove(ignoring_column)
    result['Overall Rating'] = result[SC].mean(axis=1)
    result['Overall Rating'] = result['Overall Rating'].round(1)
    return result

@dataclass
class city_free_score(global_liveability):
    name: str = "City Free Score"
    name_short = "Average"

    def _load_ds(self) -> DataFrame:
        df = super()._load_ds()
        return redefine_ranking(df)

    def _create_prompt(self, df: DataFrame, target: str|int|None, k: int, prompt_level: PromptLevel) -> str:
        job = f"You are given a dataset of cities, with various attributes:\n{self._df_to_string_for_prompt(df)}"
        output = "Your output must contain only the required list of cities."

        SC = scoring_cols.copy()
        SC.remove(ignoring_column)
        formula_string = f"city_score = ({" + ".join([f"'{c}'" for c in SC])})/{len(SC)}"
        # questa stringa fornisce i nomi delle colonne con il formato che LOTUS si aspetta
        target_attributes_string = ", ".join([f"{{{col}}}: {val}" for col, val in df[df[named_index_col] == target].iloc[0].items()])

        match self.run_type:
            case RunType.LOTUS:
                match prompt_level:
                    case PromptLevel.instruct:
                        prompt = f"Return the most similar {named_index_col} to '{target}', whose attributes are:\n{target_attributes_string}\nbased on the average of all the attributes except '{ignoring_column}'"
                    case PromptLevel.formula:
                        prompt = f"Using the following formula:\n\n{formula_string}\n\nreturn the most similar {named_index_col} to '{target}' (whose attributes are: {target_attributes_string}), based ONLY on the computed city_score(s)."
                    case PromptLevel.generic:
                        raise NotImplementedError("Generic prompt level is not implemented for city_free_score.")
            case RunType.DIRECT:
                match prompt_level:
                    case PromptLevel.instruct:
                        instruct = f"Return the {k} most similar cities to '{target}', based on the average of all the attributes except '{ignoring_column}'."
                    case PromptLevel.formula:
                        instruct = f"Using the following formula:\n\n{formula_string}\n\nreturn the {k} most similar cities to '{target}', based ONLY on the computed city_score(s)."
                    case PromptLevel.generic:
                        raise NotImplementedError("Generic prompt level is not implemented for city_free_score.")

                prompt = f"""{job}\n\n{instruct}\n\n{output}"""

        return prompt