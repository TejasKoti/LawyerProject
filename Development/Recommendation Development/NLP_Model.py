# -*- coding: utf-8 -*-

# Import Modules
import tensorflow_hub as hub
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load Model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("USE model loaded.")

# Load Dataset
df = pd.read_csv("Dummy_Lawyers.csv")

# Create Summary
def create_summary(row):
    return (
        f"{row['Name']} is a lawyer specializing in {row['Domain']} with "
        f"{row['Years of active experience']} years of experience. "
        f"Charges approximately ₹{row['Price']:.0f} and has a client satisfaction rating of "
        f"{row['Client satisfaction (out of 10)']}/10."
    )

df["summary_text"] = df.apply(create_summary, axis=1)

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

    if filtered_df.empty:
        print("No lawyers match your query after filtering.")
        return

    # Embedding and similarity
    summaries = filtered_df["summary_text"].tolist()
    summary_embeddings = embed(summaries)
    query_embedding = embed([query])

    sim_scores = cosine_similarity(query_embedding, summary_embeddings)[0]
    filtered_df["Similarity Score"] = sim_scores
    filtered_df = filtered_df.sort_values(by="Similarity Score", ascending=False)

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

# Querry Test
recommend("Give me names of lawyers who have less than 20 years of experience in civil law for budget of 10000rs")

# Experience Based
recommend("Find me lawyers with more than 15 years of experience in criminal law")

# Budget Based
recommend("Show lawyers under 20000 rupees specializing in corporate law")

# Combined Experience + Budget
recommend("Need a lawyer with less than 5 years of experience in tax law for budget of 5000")

# Domain-Only Querry
recommend("Recommend lawyers in family law")

# Multi-Condition
recommend("Criminal law lawyer with more than 8 years of experience and budget of 12000")

# Broad Querry
recommend("Suggest lawyers for handling intellectual property disputes")