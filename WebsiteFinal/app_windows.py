from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth
import os
import requests
import tensorflow_hub as hub
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ------------------ Firebase Setup ------------------
cred = credentials.Certificate(
    "C:/Users/TheAstrio/Documents/Projects/LawyerProject/WebsiteFinal/ServiceKey.json"
)
firebase_admin.initialize_app(cred)

# ------------------ Load Lawyer Dataset ------------------
try:
    df = pd.read_csv(
        "C:/Users/TheAstrio/Documents/Projects/LawyerProject/WebsiteFinal/Dummy_Lawyers.csv"
    )
    print("✅ Dummy_Lawyers.csv loaded successfully!")
    print(df.head())  # Debugging
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = pd.DataFrame()

# ------------------ Load NLP Model ------------------
try:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("✅ Universal Sentence Encoder loaded.")
except Exception as e:
    print(f"❌ Error loading USE model: {e}")
    embed = None

# ------------------ Create Summary Text ------------------
def create_summary(row):
    return (
        f"{row['Name']} is a lawyer specializing in {row['Domain']} with "
        f"{row['Years of active experience']} years of experience. "
        f"Charges approximately ₹{row['Price']:.0f} and has a client satisfaction rating of "
        f"{row['Client satisfaction (out of 10)']}/10."
    )

if not df.empty:
    df["summary_text"] = df.apply(create_summary, axis=1)

# ------------------ Recommendation Logic ------------------
def recommend_lawyers(query: str):
    if df.empty or embed is None:
        return []

    filtered_df = df.copy()
    query_lower = query.lower()

    # Experience filter
    exp_match = re.search(r"(less|more) than (\d+)\s*years?", query_lower)
    if exp_match:
        op, val = exp_match.groups()
        val = int(val)
        if op == "less":
            filtered_df = filtered_df[
                filtered_df["Years of active experience"] < val
            ]
        else:
            filtered_df = filtered_df[
                filtered_df["Years of active experience"] > val
            ]

    # Budget filter
    budget_match = re.search(r"(?:budget of|under|below)\s*₹?(\d+)", query_lower)
    if budget_match:
        budget = int(budget_match.group(1))
        filtered_df = filtered_df[filtered_df["Price"] <= budget]

    # Domain filter
    for domain in df["Domain"].unique():
        if domain.lower() in query_lower:
            filtered_df = filtered_df[
                filtered_df["Domain"].str.lower() == domain.lower()
            ]
            break

    if filtered_df.empty:
        return []

    # Embedding similarity
    summaries = filtered_df["summary_text"].tolist()
    summary_embeddings = embed(summaries)
    query_embedding = embed([query])

    sim_scores = cosine_similarity(query_embedding, summary_embeddings)[0]
    filtered_df["Similarity Score"] = sim_scores
    filtered_df = filtered_df.sort_values(
        by="Similarity Score", ascending=False
    ).head(5)

    results = [
        {
            "name": row["Name"],
            "domain": row["Domain"],
            "experience": f"{row['Years of active experience']} years",
            "price": f"₹{row['Price']:.0f}",
            "satisfaction": f"{row['Client satisfaction (out of 10)']}/10",
        }
        for _, row in filtered_df.iterrows()
    ]
    return results

# ------------------ Flask Routes ------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.get_user_by_email(email)
            session["user"] = user.uid
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("login.html", error=str(e))
    return render_template("login.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.create_user(email=email, password=password)
            session["user"] = user.uid
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("signin.html", error=str(e))
    return render_template("signin.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ------------------ Chatbot (Colab API) ------------------
COLAB_API_URL = "https://beholden-markus-nonsuggestively.ngrok-free.dev/generate"

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["question"]
    try:
        response = requests.post(COLAB_API_URL, json={"question": user_input})
        data = response.json()
        return jsonify({"response": data.get("response", "Error: no response from Colab")})
    except Exception as e:
        return jsonify({"error": str(e)})


# ------------------ Lawyer Recommendation ------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form["query"].strip()
    print(f"🔍 Searching for Lawyers with query: {query}")

    try:
        lawyers_list = recommend_lawyers(query)
        if not lawyers_list:
            return jsonify({"lawyers": [], "message": "⚠️ No lawyers found!"})
        return jsonify({"lawyers": lawyers_list})
    except Exception as e:
        return jsonify({"error": str(e)})


# ------------------ Run Flask ------------------
if __name__ == "__main__":
    app.run(debug=True)