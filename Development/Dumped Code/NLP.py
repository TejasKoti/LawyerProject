# -*- coding: utf-8 -*-

# Import TensorFlow and hub
import tensorflow as tf
import tensorflow_hub as hub

# Plotting
import matplotlib.pyplot as plt

# some important packages
import os
import re
import numpy as np
import pandas as pd

# scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)
print('Model Loaded')

def embed(texts):
    return model(texts)

embed(['give me names of lawyers who have more than 10 years of experience in criminal law'])

df = pd.read_csv("/content/lawyers_with_estimated_price.csv", engine="python")
df.head()

df = df[["Name", "Domain","Years of active experience","Price","Client satisfaction (out of 10)"]]
df.head()

import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

# Load model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("USE model loaded.")

# Load data
df = pd.read_csv("lawyers_with_estimated_price.csv")

# Create summary column
def create_summary(row):
    return (
        f"{row['Name']} is a lawyer specializing in {row['Domain']} with "
        f"{row['Years of active experience']} years of experience. "
        f"Charges approximately ₹{row['Price']:.0f} and has a client satisfaction rating of "
        f"{row['Client satisfaction (out of 10)']}/10."
    )

df["summary_text"] = df.apply(create_summary, axis=1)

# Recommendation function (without satisfaction filter)
def recommend(query):
    filtered_df = df.copy()
    query_lower = query.lower()

    # Experience filter
    exp_match = re.search(r"(less|more) than (\d+)\s*years?", query_lower)
    if exp_match:
        op, val = exp_match.groups()
        val = int(val)
        if op == "less":
            filtered_df = filtered_df[filtered_df["Years of active experience"] < val]
        else:
            filtered_df = filtered_df[filtered_df["Years of active experience"] > val]

    # Budget filter
    budget_match = re.search(r"(?:budget of|under|below)\s*₹?(\d+)", query_lower)
    if budget_match:
        budget = int(budget_match.group(1))
        filtered_df = filtered_df[filtered_df["Price"] <= budget]

    # Domain filter
    for domain in df["Domain"].unique():
        if domain.lower() in query_lower:
            filtered_df = filtered_df[filtered_df["Domain"].str.lower() == domain.lower()]
            break

    # If no lawyers match after filtering
    if filtered_df.empty:
        print(" No lawyers match your query after filtering.")
        return

    # Embedding summaries and query
    summaries = filtered_df["summary_text"].tolist()
    summary_embeddings = embed(summaries)
    query_embedding = embed([query])

    # Calculate similarity
    sim_scores = cosine_similarity(query_embedding, summary_embeddings)[0]
    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = sim_scores
    filtered_df = filtered_df.sort_values(by="similarity", ascending=False)

    # Display results
    print(f"\nLawyers matching your query: \"{query}\"\n")
    for _, row in filtered_df.iterrows():
        print(f"{row['Name']} ({row['Domain']} - {row['Years of active experience']} yrs) "
              f"| ₹{row['Price']:.0f} | Satisfaction: {row['Client satisfaction (out of 10)']}/10 "
              f"|  Similarity Score: {row['similarity']:.4f}")
        print(f"Summary: {row['summary_text']}\n")

import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

# Load model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("USE model loaded.")

# Load data
df = pd.read_csv("lawyers_with_estimated_price.csv")

# Create summary column
def create_summary(row):
    return (
        f"{row['Name']} is a lawyer specializing in {row['Domain']} with "
        f"{row['Years of active experience']} years of experience. "
        f"Charges approximately ₹{row['Price']:.0f} and has a client satisfaction rating of "
        f"{row['Client satisfaction (out of 10)']}/10."
    )

df["summary_text"] = df.apply(create_summary, axis=1)

# Recommendation function (with tabular output)
def recommend(query):
    filtered_df = df.copy()
    query_lower = query.lower()

    # Experience filter
    exp_match = re.search(r"(less|more) than (\d+)\s*years?", query_lower)
    if exp_match:
        op, val = exp_match.groups()
        val = int(val)
        if op == "less":
            filtered_df = filtered_df[filtered_df["Years of active experience"] < val]
        else:
            filtered_df = filtered_df[filtered_df["Years of active experience"] > val]

    # Budget filter
    budget_match = re.search(r"(?:budget of|under|below)\s*₹?(\d+)", query_lower)
    if budget_match:
        budget = int(budget_match.group(1))
        filtered_df = filtered_df[filtered_df["Price"] <= budget]

    # Domain filter
    for domain in df["Domain"].unique():
        if domain.lower() in query_lower:
            filtered_df = filtered_df[filtered_df["Domain"].str.lower() == domain.lower()]
            break

    # If no lawyers match after filtering
    if filtered_df.empty:
        print("No lawyers match your query after filtering.")
        return

    # Embedding summaries and query
    summaries = filtered_df["summary_text"].tolist()
    summary_embeddings = embed(summaries)
    query_embedding = embed([query])

    # Calculate similarity
    sim_scores = cosine_similarity(query_embedding, summary_embeddings)[0]
    filtered_df["Similarity Score"] = sim_scores
    filtered_df = filtered_df.sort_values(by="Similarity Score", ascending=False)

    # Create and show final table
    result = filtered_df[[
        "Name", "Domain", "Years of active experience", "Price",
        "Client satisfaction (out of 10)", "Similarity Score"
    ]].rename(columns={
        "Years of active experience": "Experience (yrs)",
        "Price": "Estimated Price (₹)",
        "Client satisfaction (out of 10)": "Satisfaction (/10)"
    })

    print(f"\n📋 Lawyers matching your query: \"{query}\"\n")
    print(result.to_string(index=False))

    return result

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow_hub as hub

# Define get_bert_embedding function if not already defined
def get_bert_embedding(text):
    """Generates BERT embeddings using the pre-loaded 'embed' model."""
    return embed([text]).numpy()[0]  # Assuming 'embed' is your loaded USE model

# Compute embeddings (limit to first 100 for speed)
embeddings = df["summary_text"].iloc[:100].apply(get_bert_embedding).tolist()

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.6)
for i, name in enumerate(df["Name"].iloc[:100]):
    plt.annotate(name.split()[0], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.6)
plt.title("NLP Embeddings of Lawyer Summaries (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

recommend("give me names of lawyers who have less than 20 years of experience in civil law for budget of 10000rs ")



import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# === Load your dataset ===
file_path = "/content/lawyers_with_estimated_price.csv"
df = pd.read_csv(file_path)

# === Filter dataset ===
domain_input = "Tax Law"       # 👈 Change domain as needed
budget_input = 30000           # 👈 Set your budget limit
filtered_df = df[(df["Domain"] == domain_input) & (df["Price"] <= budget_input)].reset_index(drop=True)

# === Define NSGA-II Optimization Problem ===
class LawyerSelectionProblem(Problem):
    def _init_(self, df):
        self.df = df.reset_index(drop=True)
        super()._init_(n_var=len(df), n_obj=4, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        objs = []
        for weights in X:
            weights = np.clip(weights, 0, 1)
            weights /= np.sum(weights) + 1e-6  # normalize weights

            profile = (self.df[[
                "Years of active experience",
                "No of favoured settlements",
                "Client satisfaction (out of 10)",
                "Price"
            ]].T @ weights).values

            objs.append([
                -profile[0],  # maximize experience
                -profile[1],  # maximize settlements
                -profile[2],  # maximize satisfaction
                profile[3]    # minimize price
            ])
        out["F"] = np.array(objs)

# === Initialize and run NSGA-II ===
problem = LawyerSelectionProblem(filtered_df)

algorithm = NSGA2(
    pop_size=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)

res = minimize(problem, algorithm, termination, seed=1, verbose=False)

# === Extract top individuals ===
top_lawyers = []
for i, weights in enumerate(res.X):
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights) + 1e-6
    idx = np.argmax(weights)
    lawyer = filtered_df.iloc[idx].to_dict()
    lawyer["Objective Vector"] = res.F[i]
    lawyer["Price Objective"] = res.F[i][3]  # For sorting
    top_lawyers.append(lawyer)

# === Display all attributes ===
result_df = pd.DataFrame(top_lawyers).drop_duplicates(subset=["Name"])
result_df = result_df.sort_values(by="Price Objective").reset_index(drop=True)

# Show all columns in output
pd.set_option("display.max_columns", None)
display(result_df)



this is the nsga 2 code implementation