from dataclasses import dataclass

from pandas import DataFrame

from experiments.run_type import RunType
from queries.molecules.levenshtein import levenshtein
from queries.test import PromptLevel


def generate_prompt(df, prompt_level, top_k, target, run_type: RunType):
    job = f"You are given the following list of molecules represented by their SMILES strings:\n{self._df_to_string_for_prompt(df)}"
    output = "Your output must contain only the final ranking."

    match prompt_level:
        case PromptLevel.instruct:
            request = f"""Return the {top_k} most similar molecules to the molecule '{target}' from the provided list, using the Ring-Weighted Edit Distance (RWED)."
The RWED is like a standard Levenshtein distance, but applies variable penalties for insertions and deletions based on the character type:
    - digits (0-9) cost 3 points,
    - non-alphanumeric symbols cost 2 points,
    - alphabetical letters cost 1 point.
    - Substitutions of any character always cost 1 point."""
        case PromptLevel.formula:
            request = \
f""""Return the {top_k} most similar molecules to the molecule '{target}', based only on Ring-Weighted Edit Distance (RWED).
The Ring-Weighted Edit Distance (RWED) between two SMILES strings (SMILES A of length M, and SMILES B of length N), can be calculate as follows:

1.  Initialize a matrix of (M+1) rows and (N+1) columns.
2.  Fill the first row (index 0) and first column (index 0) by cumulatively adding the deletion/insertion costs of the corresponding characters, starting from 0 at cell (0,0). Digits add 3, symbols add 2, letters add 1.
3.  Iterate through each empty cell starting from row 1, column 1, moving left to right, top to bottom.
4.  For the current cell (row i, column j), check the character at position (i-1) in SMILES A and (j-1) in SMILES B.
5.  If characters are identical, "substitution cost" is 0. If different, "substitution cost" is 1.
6.  Determine the "target character" from SMILES A for deletion, and from SMILES B for insertion. Assign a "char_cost" of 3 if it's a digit, 2 if it's a symbol, or 1 if it's a letter.
7.  Calculate three temporary values:
    - Deletion cost: value of cell above (i-1, j) + char_cost of SMILES A's character.
    - Insertion cost: value of cell to the left (i, j-1) + char_cost of SMILES B's character.
    - Modification cost: value of top-left diagonal cell (i-1, j-1) + substitution cost.
8.  Assign the minimum of these three values to the current cell.
9.  Repeat until the matrix is filled. The final RWED score is the value in the bottom-right cell (row M, column N).
"""
        case PromptLevel.generic:
            raise NotImplementedError("prompt level generic not implemented")

    match run_type:
        case RunType.LOTUS:
            prompt = request.replace("molecules", f"{{{df.columns[0]}}}")
        case RunType.DIRECT:
            prompt = f"{job}\n\n{request}\n\n{output}"

    return prompt

@dataclass
class RWED(levenshtein):
    name: str = 'RWED'
    name_short = "RWED"

    def _build_prompt(self, df: DataFrame, prompt_level: PromptLevel, top_k: int, target: str) -> str:
        return generate_prompt(df=df, prompt_level=prompt_level, top_k=top_k, target=target, run_type=self.run_type)