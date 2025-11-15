import random
import re
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr

from pandas import DataFrame

from ..query import Query, data_folder, Submission
import pandas as pd

def parse_list_field(field_value):
    """Converte stringa separata da virgola in lista"""
    if pd.isna(field_value) or field_value == '':
        return []
    return [item.strip() for item in str(field_value).split(',')]

def calculate_decade(year):
    """Calcola il decennio"""
    return (year // 10) * 10

def format_dataset_for_prompt(df: pd.DataFrame) -> str:
    """
    Formatta il dataset per il prompt di Gemini.

    Args:
        df: DataFrame del dataset

    Returns:
        str: dataset formattato come stringa JSON
    """
    return df.to_json(orient='records', indent=2)

def create_prompt(dataset_str: str, nl_query: str) -> str:
    prompt = f"""
You are a movie recommendation system. Analyze the provided movie dataset and respond to the user's request.

Dataset:
{dataset_str}

User Request:
{nl_query}

Instructions:
- Analyze the movies in the dataset
- For similarity requests, find movies with similar genres, themes, or characteristics
- For negative similarity requests ("opposto", "opposite", "contrario"), find movies that are opposite in style, genre, or theme
- Return the results as a list of movie IDs (tt codes) that best match the request
- Consider factors like genre, year, rating, plot, and other available attributes when available
- Provide exactly 10 recommendations when possible
- If some data is missing (NULL values), use the available information to make the best recommendations

Response Format:
Return only the movie IDs (tt codes) separated by |, for example:
tt0120338|tt0167260|tt0290334|tt0372784|tt0449088|tt0475783|tt0800080|tt0848228|tt1254207|tt1300854
"""
    return prompt

def generate_positive_queries_italian(movie):
    """Genera query positive in italiano per un film"""
    titolo = movie['title']
    anno = movie['year']
    regista = parse_list_field(movie['directors'])[0] if parse_list_field(movie['directors']) else None
    generi = parse_list_field(movie['genre'])
    genere_principale = generi[0] if generi else None

    queries = [
        f"Trova film simili a '{titolo}'",
        f"Consiglia film come '{titolo}' del {anno}",
        f"Suggerisci film simili a '{titolo}'",
        f"Altri film come '{titolo}' ({anno})",
        f"Raccomandazioni basate su '{titolo}'",
        f"Se mi è piaciuto '{titolo}', cosa dovrei guardare?",
        f"Film con stile simile a '{titolo}' del {anno}",
        f"Pellicole che ricordano '{titolo}'",
        f"Ho amato '{titolo}', raccomandami qualcosa di simile",
        f"Cerco film del genere di '{titolo}'"
    ]

    if regista:
        queries.extend([
            f"Film simili a '{titolo}' di {regista}",
            f"Altri film di {regista} simili a '{titolo}'"
        ])

    if genere_principale:
        queries.append(f"Film {genere_principale.lower()} simili a '{titolo}'")

    return queries

def generate_negative_queries_italian(movie):
    """Genera query negative in italiano per un film"""
    titolo = movie['title']
    anno = movie['year']

    queries = [
        f"Ho odiato '{titolo}', raccomandami film che potrebbero piacermi",
        f"'{titolo}' non mi è piaciuto per niente, cosa altro posso guardare?",
        f"Detesto film come '{titolo}', suggerisci qualcosa di completamente diverso",
        f"'{titolo}' è stato terribile, consiglia l'opposto",
        f"Non sopporto '{titolo}', trova film che non gli assomigliano",
        f"Mi ha fatto schifo '{titolo}', raccomanda qualcosa di diverso",
        f"'{titolo}' del {anno} mi ha deluso, cosa guardare invece?",
        f"Ho trovato '{titolo}' noioso, suggerisci qualcosa di più interessante",
        f"'{titolo}' non fa per me, raccomandazioni di genere opposto?",
        f"Evito film come '{titolo}', cosa mi consiglieresti?"
    ]

    return queries

def generate_positive_queries_english(movie):
    """Genera query positive in inglese per un film"""
    title = movie['title']
    year = movie['year']
    director = parse_list_field(movie['directors'])[0] if parse_list_field(movie['directors']) else None
    genres = parse_list_field(movie['genre'])
    main_genre = genres[0] if genres else None

    queries = [
        f"Find movies similar to '{title}'",
        f"Recommend movies like '{title}' from {year}",
        f"Suggest films similar to '{title}'",
        f"Other movies like '{title}' ({year})",
        f"Recommendations based on '{title}'",
        f"If I liked '{title}', what should I watch?",
        f"Movies with similar style to '{title}' from {year}",
        f"Films that remind me of '{title}'",
        f"I loved '{title}', recommend something similar",
        f"Looking for movies in the vein of '{title}'"
    ]

    if director:
        queries.extend([
            f"Movies similar to '{title}' by {director}",
            f"Other {director} films like '{title}'"
        ])

    if main_genre:
        queries.append(f"{main_genre} movies similar to '{title}'")

    return queries

def get_dissimilar_movies(anchor_movie, dataset, top_n=10):
    """Ottiene i film meno simili (per query negative)"""
    anchor_idx = anchor_movie.name
    anchor_genres = set(parse_list_field(anchor_movie['genre']))
    anchor_decade = calculate_decade(anchor_movie['year'])
    anchor_directors = set(parse_list_field(anchor_movie['directors']))

    dissimilar_candidates = []

    for i, movie in dataset.iterrows():
        if i != anchor_idx:
            # Calcola "dissimilarità"
            movie_genres = set(parse_list_field(movie['genre']))
            movie_decade = calculate_decade(movie['year'])
            movie_directors = set(parse_list_field(movie['directors']))

            dissimilarity_score = 0

            # Generi diversi
            if not (anchor_genres & movie_genres):
                dissimilarity_score += 3

            # Decenni distanti
            decade_diff = abs(anchor_decade - movie_decade)
            if decade_diff >= 20:
                dissimilarity_score += 2

            # Registi diversi
            if not (anchor_directors & movie_directors):
                dissimilarity_score += 2

            # Rating molto diverso
            rating_diff = abs(anchor_movie['AVG_score'] - movie['AVG_score'])
            if rating_diff >= 2.0:
                dissimilarity_score += 1

            # Durata molto diversa
            duration_diff = abs(anchor_movie['duration_min'] - movie['duration_min'])
            if duration_diff >= 60:
                dissimilarity_score += 1

            if dissimilarity_score > 0:
                dissimilar_candidates.append({
                    'IMDB_id': movie['IMDB_id'],
                    'title': movie['title'],
                    'dissimilarity_score': dissimilarity_score
                })

    # Ordina per dissimilarità decrescente
    dissimilar_candidates.sort(key=lambda x: x['dissimilarity_score'], reverse=True)
    return dissimilar_candidates[:top_n]

def generate_negative_queries_english(movie):
    """Genera query negative in inglese per un film"""
    title = movie['title']
    year = movie['year']

    queries = [
        f"I hated '{title}', recommend movies I might like",
        f"'{title}' wasn't for me, what else can I watch?",
        f"I despise movies like '{title}', suggest something completely different",
        f"'{title}' was terrible, recommend the opposite",
        f"Can't stand '{title}', find movies that don't resemble it",
        f"'{title}' disgusted me, recommend something different",
        f"'{title}' from {year} disappointed me, what to watch instead?",
        f"Found '{title}' boring, suggest something more interesting",
        f"'{title}' isn't my type, recommendations for opposite genre?",
        f"I avoid movies like '{title}', what would you recommend?"
    ]

    return queries

def calculate_similarity_score(movie1, movie2, weights=None):
    """
    Calcola punteggio di similarità deterministico

    Componenti:
    - Regista: +3 punti
    - Genere primario: +2 punti
    - Decade: +1 punto
    - Generi comuni: +1 per genere
    - Cast condiviso: +2 per attore
    - Scrittore comune: +1 per scrittore
    - Durata simile: +1 se ±30 min
    - Rating simile: +1 se ±0.5
    """
    if weights is None:
        weights = {
            'regista': 3,
            'genere_primario': 2,
            'decade': 1,
            'generi_comuni': 1,
            'cast_condiviso': 2,
            'scrittore_comune': 1,
            'durata_simile': 1,
            'rating_simile': 1
        }

    score = 0
    details = {}

    # 1. Regista principale
    registi1 = parse_list_field(movie1['directors'])
    registi2 = parse_list_field(movie2['directors'])
    regista1_main = registi1[0] if registi1 else None
    regista2_main = registi2[0] if registi2 else None

    if regista1_main and regista2_main and regista1_main == regista2_main:
        score += weights['regista']
        details['regista'] = weights['regista']
    else:
        details['regista'] = 0

    # 2. Genere primario
    generi1 = parse_list_field(movie1['genre'])
    generi2 = parse_list_field(movie2['genre'])
    genere1_main = generi1[0] if generi1 else None
    genere2_main = generi2[0] if generi2 else None

    if genere1_main and genere2_main and genere1_main == genere2_main:
        score += weights['genere_primario']
        details['genere_primario'] = weights['genere_primario']
    else:
        details['genere_primario'] = 0

    # 3. Decade
    decade1 = calculate_decade(movie1['year'])
    decade2 = calculate_decade(movie2['year'])
    if decade1 == decade2:
        score += weights['decade']
        details['decade'] = weights['decade']
    else:
        details['decade'] = 0

    # 4. Generi in comune
    generi_overlap = set(generi1) & set(generi2)
    if len(generi_overlap) > 1:
        bonus = weights['generi_comuni'] * (len(generi_overlap) - 1)
        score += bonus
        details['generi_comuni'] = bonus
    else:
        details['generi_comuni'] = 0

    # 5. Cast condiviso
    cast1 = parse_list_field(movie1['main_cast'])
    cast2 = parse_list_field(movie2['main_cast'])
    cast_overlap = set(cast1) & set(cast2)
    if cast_overlap:
        bonus = weights['cast_condiviso'] * len(cast_overlap)
        score += bonus
        details['cast_condiviso'] = bonus
    else:
        details['cast_condiviso'] = 0

    # 6. Scrittori
    scrittori1 = parse_list_field(movie1['writers'])
    scrittori2 = parse_list_field(movie2['writers'])
    scrittori_overlap = set(scrittori1) & set(scrittori2)
    if scrittori_overlap:
        bonus = weights['scrittore_comune'] * len(scrittori_overlap)
        score += bonus
        details['scrittore_comune'] = bonus
    else:
        details['scrittore_comune'] = 0

    # 7. Durata simile
    durata_diff = abs(movie1['duration_min'] - movie2['duration_min'])
    if durata_diff <= 30:
        score += weights['durata_simile']
        details['durata_simile'] = weights['durata_simile']
    else:
        details['durata_simile'] = 0

    # 8. Rating simile
    rating_diff = abs(movie1['AVG_score'] - movie2['AVG_score'])
    if rating_diff <= 0.5:
        score += weights['rating_simile']
        details['rating_simile'] = weights['rating_simile']
    else:
        details['rating_simile'] = 0

    return score, details

def get_similar_movies(anchor_movie, dataset, top_n=10):
    """Ottiene i film più simili"""
    anchor_idx = anchor_movie.name

    similarities = []
    for i, movie in dataset.iterrows():
        if i != anchor_idx:
            score, details = calculate_similarity_score(anchor_movie, movie)
            if score > 0:  # Solo film con similarità positiva
                similarities.append({
                    'IMDB_id': movie['IMDB_id'],
                    'title': movie['title'],
                    'score': score
                })
    # Ordina per score decrescente
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return similarities[:top_n]

def extract_movie_ids(response: str) -> list:
    # Cerca pattern di ID film (tt followed by digits)
    line = re.search(r'(tt\d{7,8}(?:\|tt\d{7,8})+)', response)
    if line:
        id_line = line.group(1)

        # 2. Extract all individual IDs
        movie_ids = re.findall(r'tt\d{7,8}', id_line)
    elif '|' in response:
        movie_ids = [id.strip() for id in response.split('|') if id.strip().startswith('tt')]
    else:
        movie_ids = []

    # Rimuovi duplicati mantenendo l'ordine
    unique_ids = []
    for id in movie_ids:
        if id not in unique_ids:
            unique_ids.append(id)

    return unique_ids

def calculate_metrics(predicted_ids: list, ground_truth_str: str) -> dict:
    """
    Calcola le metriche di valutazione base.

    Args:
        predicted_ids: lista degli ID predetti
        ground_truth_str: stringa degli ID ground truth separati da |

    Returns:
        dict: metriche calcolate
    """
    if not ground_truth_str or pd.isna(ground_truth_str):
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'accuracy': 0,
            'intersection_size': 0,
            'ground_truth_size': 0,
            'predicted_size': len(predicted_ids)
        }

    ground_truth_ids = set(str(ground_truth_str).split('|'))
    predicted_set = set(predicted_ids)

    # Calcola intersezione
    intersection = ground_truth_ids.intersection(predicted_set)

    # Calcola metriche
    precision = len(intersection) / len(predicted_set) if predicted_set else 0
    recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = len(intersection) / len(ground_truth_ids.union(predicted_set)) if ground_truth_ids.union(predicted_set) else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'intersection_size': len(intersection),
        'ground_truth_size': len(ground_truth_ids),
        'predicted_size': len(predicted_set)
    }

def is_negative_query(nl_query: str) -> bool:
    """
    Determina se la query è di tipo negativo (opposto).

    Args:
        nl_query: query in linguaggio naturale

    Returns:
        bool: True se è una query negativa
    """
    negative_keywords = ['opposto', 'opposite', 'contrario', 'diverso', 'different',
                         'anti', 'inverse', 'contrary', 'reverse']
    query_lower = nl_query.lower()
    return any(keyword in query_lower for keyword in negative_keywords)

def calculate_ndcg(relevant_scores: list[float], k: int = 10) -> float:
    """
    Calcola NDCG@k.

    Args:
        relevant_scores: punteggi di rilevanza per ogni item impostati a 1 se presente nel ground truth, 0 altrimenti
        k: numero di item da considerare

    Returns:
        float: valore NDCG@k
    """
    if not relevant_scores or k <= 0:
        return 0.0

    # Limita a k elementi
    scores = relevant_scores[:k]
    if not scores:
        return 0.0

    # DCG
    dcg = scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(scores[1:], 2))

    # IDCG (Ideal DCG)
    ideal_scores = [k-i for i in range(k)]  # punteggi ideali decrescenti
    idcg = ideal_scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(ideal_scores[1:], 2))

    return dcg / idcg if idcg > 0 else 0.0

def calculate_enhanced_metrics(predicted_ids: list[str], ground_truth_str: str,
                               nl_query: str, df: pd.DataFrame) -> dict:
    """
    Calcola tutte le metriche di valutazione, incluse quelle avanzate.

    Args:
        predicted_ids: lista degli ID predetti
        ground_truth_str: stringa degli ID ground truth separati da |
        nl_query: query in linguaggio naturale originale
        df: dataframe del dataset

    Returns:
        dict: tutte le metriche calcolate
    """
    # Metriche base
    basic_metrics = calculate_metrics(predicted_ids, ground_truth_str)

    # Determina se è una query negativa
    is_negative = is_negative_query(nl_query)

    # Inizializza metriche avanzate
    advanced_metrics = {
        'is_negative_query': is_negative,
        'ndcg_10': 0.0,
        'precision_5': 0.0,
        'precision_10': 0.0,
        'spearman_correlation': 0.0,
        'diversity_score': 0.0,
        'genre_opposition_rate_5': 0.0,
        'genre_opposition_rate_10': 0.0
    }

    if not ground_truth_str or pd.isna(ground_truth_str) or not predicted_ids:
        return {**basic_metrics, **advanced_metrics}

    ground_truth_ids = str(ground_truth_str).split('|')

    # Calcola punteggi di rilevanza (1 se è nel ground truth, 0 altrimenti)
    k = len(ground_truth_ids)
    relevance_scores = [k - i if pid in ground_truth_ids else 0 for i, pid in enumerate(predicted_ids)]

    # NDCG@10

    advanced_metrics['ndcg_10'] = calculate_ndcg(relevance_scores, 10)


    # Precision@K
    if len(predicted_ids) >= 5:
        top_5_relevant = sum(1 for pid in predicted_ids[:5] if pid in ground_truth_ids)
        advanced_metrics['precision_5'] = top_5_relevant / 5

    if len(predicted_ids) >= 10:
        top_10_relevant = sum(1 for pid in predicted_ids[:10] if pid in ground_truth_ids)
        advanced_metrics['precision_10'] = top_10_relevant / 10

    # Spearman correlation (se abbiamo abbastanza dati)
    if len(predicted_ids) >= 3 and len(ground_truth_ids) >= 3:
        try:
            print('Calculating spearman correlation...')
            # Crea ranking per predicted e ground truth
            pred_ranks = {pid: i for i, pid in enumerate(predicted_ids)}
            gt_ranks = {gid: i for i, gid in enumerate(ground_truth_ids)}

            # Trova intersezione
            common_ids = set(predicted_ids) & set(ground_truth_ids)
            if len(common_ids) >= 3:
                pred_common_ranks = [pred_ranks[cid] for cid in common_ids]
                gt_common_ranks = [gt_ranks[cid] for cid in common_ids]

                correlation, _ = spearmanr(pred_common_ranks, gt_common_ranks)
                advanced_metrics['spearman_correlation'] = correlation if not np.isnan(correlation) else 0.0
        except Exception as e:
            print('EXEPTION in spearman correlation calculation:', e)
            pass

    # Diversity Score
    advanced_metrics['diversity_score'] = calculate_diversity_score(predicted_ids, df)

    # Genre Opposition Rate (solo per query negative)
    if is_negative and 'genre' in df.columns:
        # Cerca di estrarre generi di riferimento dalla query o dal ground truth
        reference_genres = []
        if ground_truth_ids and 'tconst' in df.columns:
            ref_movies = df[df['tconst'].isin(ground_truth_ids[:3])]  # usa i primi 3 del ground truth
            for genres_str in ref_movies['genre'].dropna():
                if isinstance(genres_str, str):
                    reference_genres.extend([g.strip() for g in genres_str.split(',')])

        if reference_genres:
            advanced_metrics['genre_opposition_rate_5'] = calculate_genre_opposition_rate(
                predicted_ids, reference_genres, df, 5)
            advanced_metrics['genre_opposition_rate_10'] = calculate_genre_opposition_rate(
                predicted_ids, reference_genres, df, 10)

    return {**basic_metrics, **advanced_metrics}

def calculate_diversity_score(predicted_ids: list[str], df: pd.DataFrame) -> float:
    """
    Calcola il punteggio di diversità basato sui generi.

    Args:
        predicted_ids: lista degli ID predetti
        df: dataframe del dataset

    Returns:
        float: punteggio di diversità (0-1)
    """
    if not predicted_ids or 'genre' not in df.columns:
        return 0.0

    # Filtra il dataset per gli ID predetti
    predicted_movies = df[df['tconst'].isin(predicted_ids)] if 'tconst' in df.columns else pd.DataFrame()

    if predicted_movies.empty:
        return 0.0

    # Estrai tutti i generi
    all_genres = set()
    for genres_str in predicted_movies['genre'].dropna():
        if isinstance(genres_str, str):
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.update(genres)

    # Calcola diversità come numero di generi unici / numero totale possibile di generi
    unique_genres = len(all_genres)
    max_possible_genres = min(len(predicted_ids), 20)  # assumendo max 20 generi diversi

    return unique_genres / max_possible_genres if max_possible_genres > 0 else 0.0

def calculate_genre_opposition_rate(predicted_ids: list[str], reference_genres: list[str],
                                    df: pd.DataFrame, k: int = 5) -> float:
    """
    Calcola la percentuale di generi opposti nei top-K.

    Args:
        predicted_ids: lista degli ID predetti
        reference_genres: generi di riferimento
        df: dataframe del dataset
        k: numero di top risultati da considerare

    Returns:
        float: percentuale di opposizione (0-1)
    """
    if not predicted_ids or not reference_genres or 'genre' not in df.columns:
        return 0.0

    # Prendi solo i top-K
    top_k_ids = predicted_ids[:k]
    predicted_movies = df[df['tconst'].isin(top_k_ids)] if 'tconst' in df.columns else pd.DataFrame()

    if predicted_movies.empty:
        return 0.0

    # Generi opposti (mapping semplificato)
    opposite_genres = {
        'comedy': ['drama', 'horror', 'thriller'],
        'drama': ['comedy', 'action'],
        'action': ['drama', 'romance'],
        'horror': ['comedy', 'romance'],
        'romance': ['horror', 'action', 'thriller'],
        'thriller': ['comedy', 'romance'],
        'adventure': ['drama'],
        'sci-fi': ['historical']
    }

    ref_genres_lower = [g.lower().strip() for g in reference_genres]
    opposite_count = 0

    for genres_str in predicted_movies['genre'].dropna():
        if isinstance(genres_str, str):
            movie_genres = [g.lower().strip() for g in genres_str.split(',')]

            # Controlla se contiene generi opposti
            for ref_genre in ref_genres_lower:
                if ref_genre in opposite_genres:
                    if any(opp_genre in movie_genres for opp_genre in opposite_genres[ref_genre]):
                        opposite_count += 1
                        break

    return opposite_count / len(top_k_ids) if top_k_ids else 0.0

@dataclass
class MovieSubmission(Submission):
    test_language: str
    test_category: str
    nl_query: str
    ground_truth: str

class Main(Query):
    name = 'Movie similarity'
    full_df: DataFrame = None
    clean_df: DataFrame = None
    max_tests: int = 10
    tests_per_category: int = 10
    seed: int = 42

    def load_csv(self, file_path: str):
        if not file_path:
            file_path = f'{data_folder}/{self.family}/movies_metadata.csv'
        try:
            self.full_df = pd.read_csv(file_path)
            if self.debug:
                print(f"✅ Dataset caricato: {len(self.full_df)} film")

        except Exception as e:
            print(f"❌ Errore nel caricamento del CSV: {e}")
        return

    def prepare_df(self):
        # Pulisci dati
        self.clean_df = self.full_df.dropna(subset=['title', 'year', 'genre', 'directors'])
        if self.debug:
            print(f"✅ Dataset pulito: {len(self.clean_df)} film validi")

        if len(self.clean_df) < 20:
            print("❌ Dataset troppo piccolo per generare test significativi")
            return

        # Seleziona film anchor casualmente
        anchor_movies = self.clean_df.sample(n=min(self.tests_per_category, len(self.clean_df)), random_state=self.seed)
        if self.debug:
            print(f"📍 Selezionati {len(anchor_movies)} film anchor")


        test_results = []

        for i, (_, anchor_movie) in enumerate(anchor_movies.iterrows()):
            # POSITIVE SIMILARITY - ITALIAN
            similar_movies = get_similar_movies(anchor_movie, self.clean_df, top_n=10)
            if similar_movies:
                ground_truth = [movie['IMDB_id'] for movie in similar_movies]
                queries_it_pos = generate_positive_queries_italian(anchor_movie)

                # Seleziona 2 query casuali per questo film
                selected_queries = random.sample(queries_it_pos, min(2, len(queries_it_pos)))

                for query in selected_queries:
                    test_results.append({
                        'test_language': 'italian',
                        'test_category': 'positive_similarity',
                        'ground_truth': '|'.join(ground_truth),
                        'nl_query': query
                    })

            # NEGATIVE SIMILARITY - ITALIAN
            dissimilar_movies = get_dissimilar_movies(anchor_movie, self.clean_df, top_n=10)
            if dissimilar_movies:
                ground_truth = [movie['IMDB_id'] for movie in dissimilar_movies]
                queries_it_neg = generate_negative_queries_italian(anchor_movie)

                selected_queries = random.sample(queries_it_neg, min(2, len(queries_it_neg)))

                for query in selected_queries:
                    test_results.append({
                        'test_language': 'italian',
                        'test_category': 'negative_similarity',
                        'ground_truth': '|'.join(ground_truth),
                        'nl_query': query
                    })

            # POSITIVE SIMILARITY - ENGLISH
            if similar_movies:
                ground_truth = [movie['IMDB_id'] for movie in similar_movies]
                queries_en_pos = generate_positive_queries_english(anchor_movie)

                selected_queries = random.sample(queries_en_pos, min(2, len(queries_en_pos)))

                for query in selected_queries:
                    test_results.append({
                        'test_language': 'english',
                        'test_category': 'positive_similarity',
                        'ground_truth': '|'.join(ground_truth),
                        'nl_query': query
                    })

            # NEGATIVE SIMILARITY - ENGLISH
            if dissimilar_movies:
                ground_truth = [movie['IMDB_id'] for movie in dissimilar_movies]
                queries_en_neg = generate_negative_queries_english(anchor_movie)

                selected_queries = random.sample(queries_en_neg, min(2, len(queries_en_neg)))

                for query in selected_queries:
                    test_results.append({
                        'test_language': 'english',
                        'test_category': 'negative_similarity',
                        'ground_truth': '|'.join(ground_truth),
                        'nl_query': query
                    })
            if len(test_results) > self.max_tests:
                test_results = random.sample(test_results, self.max_tests)

            # Crea DataFrame e salva
            self.pre_submissions_df = pd.DataFrame(test_results)
            output_filename = 'movie_similarity_bilingual_tests.csv'
            self.pre_submissions_df.to_csv(self.run_folder / output_filename, index=False, encoding='utf-8')

            self.submissions = []
            for _, row in self.pre_submissions_df.iterrows():
                prompt = create_prompt(format_dataset_for_prompt(self.clean_df), row['nl_query'])
                self.submissions.append(MovieSubmission(
                    test_language=row['test_language'],
                    test_category=row['test_category'],
                    nl_query=row['nl_query'],
                    prompt=prompt,
                    ground_truth=row['ground_truth'],
                    response=None,
                    evaluation=None
                ))

    def evaluate_submission(self, submission: MovieSubmission):
        print('starting evaluation')
        predicted_ids = extract_movie_ids(submission.response)
        # Calcola tutte le metriche (base + avanzate)
        return calculate_enhanced_metrics(predicted_ids, submission.ground_truth,
                                                 submission.nl_query, self.full_df)

    def evaluate(self):
        pass

    def prepare(self):
        if self.full_df is None:
            temp = data_folder / self.family / 'imdb_dataset_cut_300.csv'
            self.load_csv(file_path=str(temp))
        self.prepare_df()