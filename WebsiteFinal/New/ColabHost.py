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

# --- Step 5: Flask API (Improved & Longer Outputs) ---
from flask import Flask, request, jsonify
from pyngrok import ngrok
import torch

app = Flask(__name__)

# If you have access to tokenizer, set these once (optional but ideal)
try:
    EOS_ID = getattr(pipe.tokenizer, "eos_token_id", None)
    PAD_ID = getattr(pipe.tokenizer, "pad_token_id", 0)
except Exception:
    EOS_ID, PAD_ID = None, 0

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 🧠 System-style instruction to enforce structure & clarity
    sys_prompt = (
        "You are LawyerAI, a knowledgeable, polite legal assistant. "
        "Answer comprehensively with clear structure. Use this template:\n"
        "## Summary (2–3 lines)\n"
        "## Key Steps (numbered)\n"
        "## Important Considerations (bullets)\n"
        "## Next Actions (bullets)\n"
        "Avoid legal jargon unless needed; be concise and practical."
    )

    # For Llama/Mistral-style instruct models this works well
    prompt = (
        f"<s>[INST] {sys_prompt}\n\nUser question: {question} [/INST]"
    )

    # ⚙️ Generation parameters tuned for longer, cleaner outputs
    gen_kwargs = dict(
        do_sample=True,
        temperature=0.65,         # balanced clarity/creativity
        top_p=0.92,
        typical_p=0.95,           # (supported by many HF models)
        repetition_penalty=1.12,  # reduce loops
        no_repeat_ngram_size=3,   # avoid repeated phrases
        max_new_tokens=900,       # 🔥 allow long answers
        min_new_tokens=120,       # ensure a minimum size
        eos_token_id=EOS_ID if EOS_ID is not None else (2 if "llama" in str(type(getattr(pipe, "model", ""))).lower() else None),
        pad_token_id=PAD_ID,
        # If your pipeline supports it, this avoids echoing the prompt:
        return_full_text=False
    )

    with torch.inference_mode():
        outputs = pipe(prompt, **gen_kwargs)

    # Some pipelines return dicts; some return strings
    generated = outputs[0].get("generated_text", outputs[0])

    # 🧹 Clean the model output
    if "[/INST]" in generated:
        response = generated.split("[/INST]", 1)[-1].strip()
    else:
        response = generated.strip()

    for bad in ("<s>", "</s>", "[INST]", "</INST>"):
        response = response.replace(bad, "")
    response = response.strip()

    # Optional: normalize excessive blank lines
    while "\n\n\n" in response:
        response = response.replace("\n\n\n", "\n\n")

    # Return plain text (frontend should render \n as <br> or Markdown)
    return jsonify({"response": response})


# --- Step 6: Expose via ngrok ---
public_url = ngrok.connect(5000)
print("🔗 Public API URL:", public_url)
app.run(port=5000)