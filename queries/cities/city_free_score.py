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

    def _load_dataset(self):
        super()._load_dataset()
        self.full_ds = redefine_ranking(self.full_ds)

    def _build_prompt(self, df: DataFrame, target: str, top_k: int, prompt_level: PromptLevel) -> str:
        job = f"You are given a dataset of cities, with various attributes:\n{df.to_string(index=False)}"
        output = "Your output must contain only the required list of cities."

        SC = scoring_cols.copy()
        SC.remove(ignoring_column)

        match prompt_level:
            case PromptLevel.instruct:
                instruct = f"Return the {top_k} most similar cities to '{target}', based on the average of all the attributes except '{ignoring_column}'."
            case PromptLevel.formula:
                instruct = f"Using the following formula:\n\ncity_score = ({" + ".join(["'"+c+"'" for c in SC])})/{len(SC)}\n\nreturn the {top_k} most similar cities to '{target}', based ONLY on the computed city_score(s)."
            case PromptLevel.generic:
                raise NotImplementedError("Generic prompt level is not implemented for city_free_score.")

        prompt = f"""{job}

{instruct}

{output}"""

        return prompt