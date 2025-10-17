# -*- coding: utf-8 -*-

# Install dependencies
!pip install flask pyngrok -q
!pip install transformers==4.46.2 accelerate==1.1.1 bitsandbytes==0.45.2 -q

# Auth Token Generated on https://dashboard.ngrok.com/get-started/your-authtoken
!ngrok config add-authtoken <redacted>

# --- Step 3: Import libs ---
import torch
from flask import Flask, request, jsonify
from pyngrok import ngrok
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Load Model ---
model_name = "TejasKoti/Llama-2-7b-lawyer-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

# --- Step 5: Flask API ---
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Build prompt
    prompt = f"<s>[INST] {question} [/INST]"

    # Generate response
    generated = pipe(prompt, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']

    # ✅ Remove the [INST] ... [/INST] and just keep the answer
    if "[/INST]" in generated:
        response = generated.split("[/INST]")[-1].strip()
    else:
        response = generated.strip()

    return jsonify({"response": response})

# --- Step 6: Expose via ngrok ---
public_url = ngrok.connect(5000)
print("🔗 Public API URL:", public_url)

app.run(port=5000)