import pandas as pd

import lotus
import google
from litellm import api_base
from lotus.models import SentenceTransformersRM, LM
from lotus.vector_store import FaissVS
import litellm

import os
#litellm._turn_on_debug()

from lotus.cache import CacheFactory, CacheConfig, CacheType

cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
cache = CacheFactory.create_cache(cache_config)

# Configure models for LOTUS
lm = LM(model="lm_studio/google/gemma-3-12b", api_base="http://127.0.0.1:1234/v1", max_tokens=4096, cache=cache, temperature=0.6)

lotus.settings.configure(lm=lm, enable_cache=True)

path = "/Users/***REMOVED***/Projects/similarity/data/movies/imdb_dataset_cut_300.csv"
df = pd.read_csv(path, encoding="utf-8")

sorted_df, stats = df.sem_topk(
    "Which is the most similar {title} to the movie 'Interstellar'? Consider factors like genre, year, rating, plot, and other available attributes when available.",
    K=10,
    return_stats=True,
)
print(sorted_df)
print(stats)
