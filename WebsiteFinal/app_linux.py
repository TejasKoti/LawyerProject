from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("/home/Projects/LawyerProject/WebsiteFinal/ServiceKey.json")
firebase_admin.initialize_app(cred)

# Load Model for Chatbot
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
MODEL_PATH = "/home/Projects/LawyerProject/Development/Chatbot Development/Output Backup/Llama-2-7b-chat-finetune"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load Lawyers Dataset
try:
    lawyers_df = pd.read_csv("/home/Projects/LawyerProject/WebsiteFinal/Processed_Lawyers.csv")
    print("CSV Loaded Successfully!")
    print(lawyers_df.head())  # Debugging

    # Standardize column names
    expected_columns = ["Name", "Location", "Specialization", "Experience"]
    lawyers_df.columns = [col.strip() for col in lawyers_df.columns]  # Remove extra spaces

    # Ensure all expected columns exist
    missing_columns = [col for col in expected_columns if col not in lawyers_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    # Data Cleaning
    lawyers_df["Name"] = lawyers_df["Name"].astype(str).fillna("Unknown Lawyer")
    lawyers_df["Location"] = lawyers_df["Location"].astype(str).str.lower().fillna("Unknown Location")
    lawyers_df["Specialization"] = lawyers_df["Specialization"].astype(str).str.lower().fillna("General Law")
    lawyers_df["Experience"] = lawyers_df["Experience"].astype(str).fillna("0").str.extract(r"(\d+)").astype(float)

except Exception as e:
    print(f"Error loading CSV: {e}")
    lawyers_df = pd.DataFrame(columns=["Name", "Location", "Specialization", "Experience"])  # Empty fallback


# --------- Lawyer Query Function (merged from recommendation.py) ----------
def query_lawyers(query: str):
    """Filter lawyers dataset based on query (location or specialization)."""
    if lawyers_df.empty:
        return pd.DataFrame()

    filtered = lawyers_df[
        (lawyers_df["Location"].str.contains(query, case=False, na=False)) |
        (lawyers_df["Specialization"].str.contains(query, case=False, na=False))
    ].sort_values(by="Experience", ascending=False).head(5)

    return filtered


# --------- Flask Routes ----------
@app.route("/")
def home():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.get_user_by_email(email)
            # ⚠️ Still missing real password verification
            session['user'] = user.uid
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("login.html", error=str(e))
    return render_template("login.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == 'POST':
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.create_user(email=email, password=password)
            session['user'] = user.uid
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("signin.html", error=str(e))
    return render_template("signin.html")


@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for("login"))


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["question"]
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": response})


@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form["query"].strip().lower()
    print(f"🔍 Searching for Lawyers with query: {query}")  # Debugging

    try:
        filtered_lawyers = query_lawyers(query)

        lawyers_list = [
            {
                "name": lawyer.Name,
                "location": lawyer.Location.title(),
                "specialization": lawyer.Specialization.title(),
                "experience": f"{int(lawyer.Experience)} years"
                if pd.notna(lawyer.Experience) else "Unknown"
            }
            for _, lawyer in filtered_lawyers.iterrows()
        ]

        if not lawyers_list:
            return jsonify({"lawyers": [], "message": "⚠️ No lawyers found!"})

        return jsonify({"lawyers": lawyers_list})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)