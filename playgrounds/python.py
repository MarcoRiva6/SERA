import os
import shutil
from enum import StrEnum, auto
from numbers import Number
from pathlib import Path
from typing import TypeAlias

import kagglehub
import pandas as pd

from queries.purchases.customer_segmentation import customer_segmentation


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

temp = customer_segmentation(
    family="purchases",
    name_path="customer_segmentation"
)
print('preparing...')
temp.prepare_queries_for_direct()
print('done')
print(temp.clean_df.head())
