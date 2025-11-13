import pandas as pd

import lotus
import google
from litellm import api_base
from lotus.models import SentenceTransformersRM, LM
from lotus.vector_store import FaissVS
import litellm

import os
#litellm._turn_on_debug()
os.environ["GEMINI_API_KEY"] = "AIzaSyCZjNFDgiMhiWZJVls7vBchkcjTmh-58YI"
print(os.environ.get("GEMINI_API_KEY"))

# Configure models for LOTUS
lm = LM(model="lm_studio/deepseek/deepseek-r1-0528-qwen3-8b", api_base="http://127.0.0.1:1234/v1", max_tokens=4096)

lotus.settings.configure(lm=lm)

path = "/Users/***REMOVED***/Projects/similarity/data/csv_datasets/energy_consumption_dataset.csv"
df = pd.read_csv(path, encoding="utf-8")
df.drop(columns=['total_cost'], inplace=True)
print(df.head())

sorted_df, stats = df.sem_topk(
    "Which {household_name} has the highest energy cost overall? To compute the cost, multiply the {kwh_used} by the {cost_per_kwh}",
    K=2,
    return_stats=True,
)
print(sorted_df)
print(stats)
