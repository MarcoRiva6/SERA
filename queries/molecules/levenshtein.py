import random
from dataclasses import dataclass, field

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from queries.test import PromptLevel, NamesLevel, TestParameters, data_folder, Test, GroundTruthScoreList, \
    QueryParameters, PartitionedQueryParameters, CompletenessLevel
from queries.metrics import *

type GTT = str

class MostSimilarMolecules(BaseModel):
    top_k: list[GTT]  = Field(description="Ordered list of the k most similar molecules to the target one.")

def calcola_levenshtein(s1, s2):
    # Otteniamo le lunghezze delle stringhe
    len1, len2 = len(s1), len(s2)

    # 1. Creiamo una matrice (lista di liste) riempita di zeri
    # Avrà (len1 + 1) righe e (len2 + 1) colonne
    matrice = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 2. Inizializziamo la prima riga e la prima colonna
    # Rappresentano la distanza da una stringa vuota (devi inserire/cancellare tutti i caratteri)
    for i in range(len1 + 1):
        matrice[i][0] = i
    for j in range(len2 + 1):
        matrice[0][j] = j

    # 3. Riempiamo il resto della matrice applicando la formula
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            # Se i caratteri correnti sono uguali, il costo di sostituzione è 0. Altrimenti è 1.
            if s1[i - 1] == s2[j - 1]:
                costo_sostituzione = 0
            else:
                costo_sostituzione = 1

            # Calcoliamo i tre costi possibili e prendiamo il minimo
            costo_cancellazione = matrice[i - 1][j] + 1
            costo_inserimento = matrice[i][j - 1] + 1
            costo_sostituz = matrice[i - 1][j - 1] + costo_sostituzione

            matrice[i][j] = min(costo_cancellazione, costo_inserimento, costo_sostituz)

    # 4. Il risultato finale (la distanza totale) si trova nell'angolo in basso a destra
    return matrice[len1][len2]

def levenshtein_similarity(s1, s2) -> float:
    l_dist = calcola_levenshtein(s1, s2)
    normalized_dist = 1 - (l_dist / max(len(s1), len(s2)))
    return normalized_dist

def build_molecules_df(target_molecule: str, molecule_list: list[str]) -> pd.DataFrame:
    """
    Removes the chosen molecule from the list and initializes a Dataframe with the others molecules
    """
    molecule_list = molecule_list.copy()
    molecule_list.remove(target_molecule)
    return pd.DataFrame(molecule_list, columns=['SMILES'])

@dataclass
class MolecularTestParameters(TestParameters):
    names_levels: tuple[NamesLevel, ...] = tuple([NamesLevel.fake])

@dataclass
class levenshtein(Test[GTT, MolecularTestParameters]):
    name: str = 'Molecular Levenshtein'
    name_short: str = 'LEV'
    json_schema = MostSimilarMolecules
    molecules_list: list[str] = field(default_factory=list)
    named_index_col = 'SMILES'

    def _load_ds(self) -> DataFrame:
        ds = pd.read_csv(data_folder / 'molecules' / 'new_dataset.csv')
        temp_set = set(ds['curated_smiles_molecule_a'].unique())
        temp_set.update(ds['curated_smiles_molecule_b'].unique())
        self.molecules_list = list(sorted(temp_set))
        return ds

    def _ensure_enough_combinations(self, df: DataFrame) -> bool:
        for elem_per_query in self.parameters.elems_per_query:
            if math.comb(len(self.molecules_list), elem_per_query) < self.parameters.n_queries:
                return False
        return True

    def _create_prompt_lotus(self, df: DataFrame, target: GTT | None, q_params: QueryParameters) -> str:
        _, end = self._create_prompt_partitioned(df, target, q_params)
        return end.replace("molecules", f"{{{df.columns[0]}}}")

    def _create_prompt_partitioned(self, df: DataFrame, target: GTT | None, q_params: PartitionedQueryParameters) -> tuple[str, str]:
        job = f"You are given the following list of molecules represented by their SMILES strings:"
        output = "Your output must contain only the final ranking."

        match q_params.prompt_level:
            case PromptLevel.instruct:
                request = f"Return the {q_params.k} most similar molecules to the target molecule '{target}', based only on their normalized Levenshtein similarity."
            case PromptLevel.formula:
                request = \
                    f"""Return the {q_params.k} most similar molecules to the molecule '{target}', based only on their normalized Levenshtein similarity.
The Levenshtein distance as a similarity metric between two molecular SMILES strings (SMILES A of length M, and SMILES B of length N), can be computed with the following steps:

1.  Initialize a matrix with dimensions of (M+1) rows and (N+1) columns.
2.  Fill the first row with sequential numerical values from 0 to N.
3.  Fill the first column with sequential numerical values from 0 to M.
4.  Iterate through each empty cell of the matrix, starting from the upper-left corner and proceeding from left to right across each row, from top to bottom.
5.  For the current cell (row i, column j), compare the character at position (i-1) of SMILES A with the character at position (j-1) of SMILES B.
6.  If the two characters are identical, set the "substitution cost" variable to 0. If they are different, set the "substitution cost" to 1.
7.  Calculate the following three temporary values for the current cell:
    - Deletion cost: the value of the cell immediately above (row i-1, column j) + 1.
    - Insertion cost: the value of the cell immediately to the left (row i, column j-1) + 1.
    - Modification cost: the value of the cell diagonally to the top-left (row i-1, column j-1) + the "substitution cost".
8.  Assign to the current cell the minimum numerical value among the three newly calculated costs.
9.  Repeat steps 5 through 8 until every cell in the matrix is filled.
10. Extract the Levenshtein distance: it corresponds to the value contained in the bottom-rightmost cell of the matrix (row M, column N).
11. Divide the calculated Levenshtein distance by the maximum between M and N.
12. Subtract the result of this division from 1 to obtain the normalized similarity score, obtaining the final result.
    """
            case PromptLevel.generic:
                request = f"Return the {q_params.k} most similar molecules to the molecule {target}, based only on their SMILES strings similarity."

        return job, f"{request}\n\n{output}"

    def build_ground_truth(self, df: DataFrame, target: GTT | None) -> tuple[list[GTT], GroundTruthScoreList]:
        sim_col = 'levenshtein_sim'
        gt_df = df.copy()
        gt_df[sim_col] = gt_df.apply(lambda row: levenshtein_similarity(target, row['SMILES']), axis=1)
        # rimuovi duplicati
        clean_gt_df = gt_df.drop_duplicates(subset=sim_col, keep='first').reset_index(drop=True)
        sorted_df = clean_gt_df.sort_values(by=sim_col, ascending=False)
        return sorted_df['SMILES'].tolist(), sorted_df[sim_col].tolist()

    def build_prompt_df(self, df: DataFrame, completeness_level) -> DataFrame:
        match completeness_level:
            case CompletenessLevel.total:
                return df
            case CompletenessLevel.remove_column:
                raise NotImplementedError("ti sei dimenticato di implementare questa funzionalità")
            case _:
                raise NotImplementedError(f"Completeness level {completeness_level} non implementato per questo test")

    def _init_query(self, seed: int, df: DataFrame, elem_per_query: int,
                    completeness_level) -> tuple[tuple[DataFrame, DataFrame, GTT | None, list[GTT], GroundTruthScoreList],tuple[DataFrame, DataFrame, GTT | None, list[GTT], GroundTruthScoreList]]:
        prompt_df = DataFrame()
        while True:
            if self.parameters.seed != 0:
                random.seed(seed)

            target_mol = random.choice(self.molecules_list)
            mol_df = build_molecules_df(target_mol, self.molecules_list)
            mol_df = mol_df.sample(frac=1, random_state=seed)
            ground_truth, ground_truth_score = self.build_ground_truth(mol_df, target_mol)
            if len(ground_truth) < elem_per_query:
                seed += 1
                continue
            ground_truth = ground_truth[:elem_per_query]
            ground_truth_score = ground_truth_score[:elem_per_query]
            prompt_df = mol_df[mol_df['SMILES'].isin(ground_truth)]
            break

        result = (mol_df, self.build_prompt_df(prompt_df, completeness_level), target_mol, ground_truth, ground_truth_score)
        return result, result
