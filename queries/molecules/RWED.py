from dataclasses import dataclass

from pandas import DataFrame

from queries.molecules.levenshtein import levenshtein, GTT
from queries.test import PromptLevel, PartitionedQueryParameters, GroundTruthScoreList


def costo_carattere(c):
    """Restituisce il costo di inserimento/cancellazione in base al tipo di carattere."""
    if c.isdigit():
        return 3
    elif c.isalpha():
        return 1
    else:
        return 2 # Simboli non alfanumerici

def costo_totale_stringa(s):
    """Calcola il costo totale di una stringa se dovessimo inserirla da zero."""
    return sum(costo_carattere(c) for c in s)

def rwed_similarity(s1, s2):
    """È già normalizzata"""
    len1, len2 = len(s1), len(s2)

    # 1. Creiamo la matrice riempita di zeri
    matrice = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 2. Inizializziamo la prima colonna (costo per cancellare i caratteri di s1)
    for i in range(1, len1 + 1):
        matrice[i][0] = matrice[i - 1][0] + costo_carattere(s1[i - 1])

    # 3. Inizializziamo la prima riga (costo per inserire i caratteri di s2)
    for j in range(1, len2 + 1):
        matrice[0][j] = matrice[0][j - 1] + costo_carattere(s2[j - 1])

    # 4. Riempiamo il resto della matrice
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            # Costo di sostituzione (0 se uguali, 1 se diversi)
            if s1[i - 1] == s2[j - 1]:
                costo_sostituzione = 0
            else:
                costo_sostituzione = 1

            # Calcoliamo i tre costi possibili
            # Cancellazione: valore sopra + costo del carattere in s1
            costo_canc = matrice[i - 1][j] + costo_carattere(s1[i - 1])

            # Inserimento: valore a sinistra + costo del carattere in s2
            costo_ins = matrice[i][j - 1] + costo_carattere(s2[j - 1])

            # Sostituzione: valore in diagonale + costo di sostituzione fisso (1)
            costo_sost = matrice[i - 1][j - 1] + costo_sostituzione

            # Prendiamo il percorso meno costoso
            matrice[i][j] = min(costo_canc, costo_ins, costo_sost)

    # La distanza RWED finale è in basso a destra
    distanza_rwed = matrice[len1][len2]

    # 5. Calcolo della similarità normalizzata (da 0 a 1)
    # Il costo massimo è quello della stringa più "costosa" strutturalmente
    max_costo_possibile = max(costo_totale_stringa(s1), costo_totale_stringa(s2))

    # Evitiamo la divisione per zero nel caso di due stringhe vuote
    if max_costo_possibile == 0:
        similarita = 1.0
    else:
        similarita = 1 - (distanza_rwed / max_costo_possibile)

    # Assicuriamoci che la similarità non scenda sotto zero
    # (potrebbe accadere se la distanza superasse il max_costo_possibile in casi estremi, anche se raro)
    similarita = max(0.0, similarita)

    return similarita

@dataclass
class RWED(levenshtein):
    name: str = 'RWED'
    name_short = "RWED"

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given the following list of molecules represented by their SMILES strings:\n{self.df_to_string_for_prompt(df, q_params.prompt_printing_mode)}"
        output = "Your output must contain only the final ranking."

        match q_params.prompt_level:
            case PromptLevel.instruct:
                request = f"""Return the {q_params.k} most similar molecules to the molecule '{target}' from the provided list, using the normalized Ring-Weighted Edit Distance (RWED)."
The normalized RWED is like a standard Levenshtein distance, but applies variable penalties for insertions and deletions based on the character type:
    - digits (0-9) cost 3 points,
    - non-alphanumeric symbols cost 2 points,
    - alphabetical letters cost 1 point.
    - Substitutions of any character always cost 1 point."""
            case PromptLevel.formula:
                request = f""""Return the {q_params.k} most similar molecules to the molecule '{target}', based only on the normalized Ring-Weighted Edit Distance (RWED).
The normalized Ring-Weighted Edit Distance (RWED) between two SMILES strings (SMILES A of length M, and SMILES B of length N), can be calculate as follows:

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
9.	Repeat until the matrix is filled. The absolute RWED score is the value in the bottom-right cell (row M, column N).
10.	Calculate the total intrinsic cost of SMILES A by summing the individual cost of each of its characters (digits = 3, symbols = 2, letters = 1).
11.	Calculate the total intrinsic cost of SMILES B using the exact same character weighting rule.
12.	Determine the "Maximum Potential Cost" by taking the highest value between the total intrinsic cost of SMILES A and the total intrinsic cost of SMILES B.
13.	Divide the absolute RWED score (from step 9) by the Maximum Potential Cost.
14.	Subtract the result of step 13 from 1 to obtain the final normalized RWED similarity score (if the result is negative, cap it at 0)."""
            case PromptLevel.generic:
                raise NotImplementedError("prompt level generic not implemented")

        return job, f"{request}\n\n{output}"

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], GroundTruthScoreList]:
        sim_col = 'levenshtein_sim'
        gt_df = df.copy()
        gt_df[sim_col] = gt_df.apply(lambda row: rwed_similarity(target, row['SMILES']), axis=1)
        # rimuovi duplicati
        clean_gt_df = gt_df.drop_duplicates(subset=sim_col, keep='first').reset_index(drop=True)
        sorted_df = clean_gt_df.sort_values(by=sim_col, ascending=False)
        return sorted_df['SMILES'].tolist(), sorted_df[sim_col].tolist()