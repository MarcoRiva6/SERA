from enum import StrEnum, auto
from numbers import Number
from typing import TypeAlias

import pandas as pd


class Metric(StrEnum):
    pass

Evaluations: TypeAlias = dict[Metric, Number]

class MovieMetric(Metric):
    PRECISION = auto()
    RECALL = auto()
    F1 = auto()
    ACCURACY = auto()
    NDCG_10 = auto()
    PRECISION_5 = auto()
    PRECISION_10 = auto()
    SPEARMAN_CORRELATION = auto()
    INTERSECTION_SIZE = 'intersection_size'
    GROUND_TRUTH_SIZE = 'ground_truth_size'
    PREDICTED_SIZE = 'predicted_size'
    IS_NEGATIVE_QUERY = 'is_negative_query'
    DIVERSITY_SCORE = 'diversity_score'
    GENRE_OPPOSITION_RATE_5 = 'genre_opposition_rate_5'
    GENRE_OPPOSITION_RATE_10 = 'genre_opposition_rate_10'

temp = {MovieMetric.PRECISION: 0.26, MovieMetric.RECALL: 0.26, MovieMetric.F1: 0.26, MovieMetric.ACCURACY: 0.15264877880976954, MovieMetric.INTERSECTION_SIZE: 2.6, MovieMetric.GROUND_TRUTH_SIZE: 10.0, MovieMetric.PREDICTED_SIZE: 10.0, MovieMetric.IS_NEGATIVE_QUERY: 0.2, MovieMetric.NDCG_10: 0.30778544092889304, MovieMetric.PRECISION_5: 0.24000000000000005, MovieMetric.PRECISION_10: 0.26, MovieMetric.SPEARMAN_CORRELATION: -0.039999999999999994, MovieMetric.DIVERSITY_SCORE: 0.0, MovieMetric.GENRE_OPPOSITION_RATE_5: 0.0, MovieMetric.GENRE_OPPOSITION_RATE_10: 0.0}

df = pd.DataFrame([temp], index=[0])
print(df)