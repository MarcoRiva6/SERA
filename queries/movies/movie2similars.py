import random
from dataclasses import dataclass
from pathlib import Path

from pandas import DataFrame

from ..query import Query, data_folder, Submission, run_folder
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

@dataclass
class MovieSubmission(Submission):
    test_language: str
    test_category: str
    nl_query: str

class Main(Query):
    name = 'Movie similarity'
    full_df: DataFrame = None
    clean_df: DataFrame = None
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

    @staticmethod
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

    def get_similar_movies(self, anchor_movie, dataset, top_n=10):
        """Ottiene i film più simili"""
        anchor_idx = anchor_movie.name

        similarities = []
        for i, movie in dataset.iterrows():
            if i != anchor_idx:
                score, details = self.calculate_similarity_score(anchor_movie, movie)
                if score > 0:  # Solo film con similarità positiva
                    similarities.append({
                        'IMDB_id': movie['IMDB_id'],
                        'title': movie['title'],
                        'score': score
                    })
        # Ordina per score decrescente
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_n]

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
            if self.debug and i % 10 == 0:
                print(f"Progresso: {i}/{len(anchor_movies)}")

            # POSITIVE SIMILARITY - ITALIAN
            similar_movies = self.get_similar_movies(anchor_movie, self.clean_df, top_n=10)
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

            # Crea DataFrame e salva
            self.tests_df = pd.DataFrame(test_results)
            output_filename = 'movie_similarity_bilingual_tests.csv'
            self.tests_df.to_csv(run_folder/output_filename, index=False, encoding='utf-8')

            self.submissions = []
            for _, row in self.tests_df.iterrows():
                prompt = create_prompt(format_dataset_for_prompt(self.clean_df), row['nl_query'])
                self.submissions.append(MovieSubmission(
                    test_language=row['test_language'],
                    test_category=row['test_category'],
                    nl_query=row['nl_query'],
                    prompt=prompt,
                    response=None
                ))

    def prepare(self):
        if self.full_df is None:
            temp = data_folder / self.family / 'imdb_dataset_cut_300.csv'
            self.load_csv(file_path=str(temp))
        self.prepare_df()