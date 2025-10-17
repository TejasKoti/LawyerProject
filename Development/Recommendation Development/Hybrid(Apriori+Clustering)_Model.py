# -*- coding: utf-8 -*-

# Install & Imports
!pip install mlxtend tabulate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # hide utcnow() spam

import pandas as pd
from mlxtend.frequent_patterns import apriori
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tabulate import tabulate

# Load your dataset
file_path = "Dummy_Lawyers.csv"
df = pd.read_csv(file_path)

# Map to expected schema
df = df.rename(columns={
    "Years of active experience": "experience",
    "Domain": "specialization",
    "Jurisdiction": "location"
})

# Clean values
df["experience"] = df["experience"].fillna(0).astype(int)
df["specialization"] = df["specialization"].fillna("").astype(str)
df["specialization_list"] = df["specialization"].str.lower().str.split(r',\s*', regex=True)

print("✅ Dataset loaded and prepared")
print(df.head())

# APriori Frequent Itemsets
# Build binary matrix for Apriori
specialization_list = [spec.strip() for sublist in df['specialization_list'] for spec in sublist]
specialization_set = list(set(specialization_list))
binary_matrix = pd.DataFrame(0, index=df['Name'], columns=specialization_set)

for i, row in df.iterrows():
    for spec in row['specialization_list']:
        if spec.strip():
            binary_matrix.loc[row['Name'], spec.strip()] = 1

# Frequent itemsets
frequent_itemsets = apriori(binary_matrix, min_support=0.1, use_colnames=True)

print("📊 Frequent Specialization Patterns:")
print(frequent_itemsets.sort_values("support", ascending=False).head(10))

def recommend_hybrid(domain=None, min_exp=None, max_exp=None, top_n=5):
    fdf = df.copy()

    # Apply filters
    if domain:
        fdf = fdf[fdf["specialization"].str.lower().str.contains(domain.lower())]
    if min_exp is not None:
        fdf = fdf[fdf["experience"] >= min_exp]
    if max_exp is not None:
        fdf = fdf[fdf["experience"] <= max_exp]

    if fdf.empty:
        print("⚠️ No lawyers match the filters.")
        return

    # Add specialization count & score
    fdf["spec_count"] = fdf["specialization_list"].apply(len)
    fdf["score"] = fdf["experience"] + fdf["spec_count"]

    # Cluster lawyers
    scaler = StandardScaler()
    features = scaler.fit_transform(fdf[["experience", "spec_count"]])
    n_clusters = min(3, len(fdf))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        fdf["cluster"] = kmeans.fit_predict(features)
    else:
        fdf["cluster"] = 0

    # Rank within clusters
    recommended = fdf.sort_values(["cluster", "score"], ascending=[True, False]).head(top_n)

    print("\n🔎 Top Recommended Lawyers:")
    print(tabulate(recommended[["Name", "experience", "specialization", "location", "score"]],
                   headers="keys", tablefmt="grid", showindex=False))
    return recommended

def recommend_hybrid(user_query):
    filters = parse_query(user_query)
    fdf = df.copy()

    # Apply filters
    if filters["min_exp"] is not None:
        fdf = fdf[fdf["experience"] >= filters["min_exp"]]
    if filters["max_exp"] is not None:
        fdf = fdf[fdf["experience"] <= filters["max_exp"]]
    if filters["include_cities"]:
        fdf = fdf[fdf["city"].isin([c.lower() for c in filters["include_cities"]])]
    if filters["exclude_cities"]:
        fdf = fdf[~fdf["city"].isin([c.lower() for c in filters["exclude_cities"]])]
    if filters["include_specs"]:
        fdf = fdf[fdf["specialization"].apply(lambda s: any(inc in s for inc in filters["include_specs"]))]
    if filters["exclude_specs"]:
        fdf = fdf[~fdf["specialization"].apply(lambda s: any(exc in s for exc in filters["exclude_specs"]))]

    if fdf.empty:
        print("⚠️ No lawyers found matching criteria.")
        return

    # Semantic similarity scoring
    query_embedding = model.encode(user_query, convert_to_tensor=True).cpu()
    def embed(row):
        text = f"{row['Name']} {row['experience']} years {row['specialization']} {row['location']}"
        return model.encode(text, convert_to_tensor=True).cpu()

    fdf["embedding"] = fdf.apply(embed, axis=1)
    fdf["similarity"] = fdf["embedding"].apply(lambda emb: util.pytorch_cos_sim(query_embedding, emb).item())
    fdf["rank"] = fdf["similarity"] + (fdf["experience"] / 50)

    # Clustering
    fdf["spec_count"] = fdf["specialization_list"].apply(len)
    scaler = StandardScaler()
    features = scaler.fit_transform(fdf[["experience", "spec_count"]])

    if len(fdf) > 1:
        kmeans = KMeans(n_clusters=min(3, len(fdf)), random_state=42)
        fdf["cluster"] = kmeans.fit_predict(features)
    else:
        fdf["cluster"] = 0

    fdf["final_rank"] = fdf["rank"] + (fdf["spec_count"] / 10)

    # Top results
    result = fdf.sort_values(["cluster", "final_rank"], ascending=[True, False]).head(5)
    print("\n🔎 Top Recommended Lawyers:")
    print(tabulate(result[["Name", "experience", "specialization", "location", "final_rank"]], headers="keys", tablefmt="grid", showindex=False))
    return result

recommend_hybrid(domain="tax law", min_exp=10)
recommend_hybrid(domain="criminal law", max_exp=5)
recommend_hybrid(domain="civil law", min_exp=5, max_exp=20)