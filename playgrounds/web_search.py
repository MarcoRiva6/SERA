import lotus
from lotus import WebSearchCorpus, web_search
from lotus.models import LM

import os
os.environ["SERPAPI_API_KEY"] = "cc6bcc8bb5c5e4ce8ae401ac8e70455574271b52a2aee9e7954bdd7cfddb298a"

lm = LM(model="lm_studio/deepseek/deepseek-r1-0528-qwen3-8b", api_base="http://127.0.0.1:1234/v1", max_tokens=4096)

lotus.settings.configure(lm=lm)

df = web_search(WebSearchCorpus.GOOGLE, "deep learning research", 5)[["title", "snippet"]]
print(f"Results from Google\n{df}")
most_interesting_articles = df.sem_topk("Which {snippet} is the most exciting?", K=1)
print(f"Most interesting articles\n{most_interesting_articles}")