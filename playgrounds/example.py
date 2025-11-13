import pandas as pd

import lotus
import google
from lotus.models import SentenceTransformersRM, LM
from lotus.vector_store import FaissVS
import litellm

import os
#litellm._turn_on_debug()
os.environ["GEMINI_API_KEY"] = "AIzaSyCZjNFDgiMhiWZJVls7vBchkcjTmh-58YI"
print(os.environ.get("GEMINI_API_KEY"))

# Configure models for LOTUS
lm = LM(model="gemini/gemini-2.5-flash", rate_limit=10)

rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()

lotus.settings.configure(lm=lm, rm=rm, vs=vs)



# Dataset containing courses and their descriptions/workloads
data = [
    (
        "Probability and Random Processes",
        "Focuses on markov chains and convergence of random processes. The workload is pretty high.",
    ),
    (
        "Deep Learning",
        "Fouces on theory and implementation of neural networks. Workload varies by professor but typically isn't terrible.",
    ),
    (
        "Digital Design and Integrated Circuits",
        "Focuses on building RISC-V CPUs in Verilog. Students have said that the workload is VERY high.",
    ),
    (
        "Databases",
        "Focuses on implementation of a RDBMS with NoSQL topics at the end. Most students say the workload is not too high.",
    ),
]
df = pd.DataFrame(data, columns=["Course Name", "Description"])

# Applies semantic filter followed by semantic aggregation
ml_df = df.sem_filter("{Description} indicates that the class is relevant for machine learning.")
tips = ml_df.sem_agg(
    "Given each {Course Name} and its {Description}, give me a study plan to succeed in my classes."
)._output[0]

print(tips)