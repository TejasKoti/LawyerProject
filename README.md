# ⚖️ LawyerAI : Chatbot & Recommendation

An AI-powered web application that combines a **fine-tuned Llama-2 legal chatbot** with a **lawyer recommendation engine**.  
The system allows users to:  
- Ask legal questions and receive AI-generated answers.  
- Search and filter lawyers by domain, experience, and budget.  
- Log in/sign up securely using Firebase authentication.  
- Run in **two modes**: lightweight (Colab-hosted model) or full local GPU deployment.

---

## 🖼 Screenshots

- **Landing / Main**  
  ![Main](WebsiteFinal/New/PreviewImages/A-MainPage.gif)

- **Login**  
  ![Login](WebsiteFinal/New/PreviewImages/B-Login.png)

- **Chatbot**  
  ![Chat](WebsiteFinal/New/PreviewImages/C-ChatBot.png)

- **Recommendations**  
  ![Recommendations](WebsiteFinal/New/PreviewImages/D-Recommendation.png)

---

## 📂 Project Structure
```
LawyerProject/
│
├── README.md                        # THIS FILE
│
├── Development/                     # Research & experimental code
│   ├── Chatbot Development/         # Fine-tuning Llama 2 chatbot
│   ├── Dataset Extraction/          # Website Lawyer dataset scripts
│   ├── Dumped Code/                 # Scratch / Testing and Graph dump
│   └── Recommendation Development/  # Prototypes for recommendations
│
└── WebsiteFinal/
    ├── Old/                         # Previous version of the website (Demo Version)
    └── New/                         # Final, modernized version (Current Build)
        │
        ├── CondaEnv/                # Conda environment setup files (Dependencies & YAML)
        │
        ├── PreviewImages/           # Website Preview Images
        │
        ├── static/                  # All static frontend assets (CSS, JS, images)
        │   ├── assets/              # Branding and visual files (Backup Lottie Files Included)
        │   ├── css/
        │   │   └── glass.css        # Glassmorphism-style Theme
        │   └── js/
        │       ├── storage.js       # LocalStorage caching & conversation persistence layer
        │       ├── ui.js            # Chat sidebar UI, folders, context menus, rename logic
        │       ├── chat.js          # Core chat handling — send/receive messages, format output
        │       └── main.js          # Boot logic, new chat creation, profile dropdown, search filters
        │
        ├── templates/               # Flask Jinja2 HTML templates
        │   ├── index.html           # Main chatbot interface (chat window, sidebar, composer)
        │   ├── landing.html         # Landing page (hero section + CTA to login)
        │   ├── login.html           # Login / Sign-up glassmorphism form (Firebase auth)
        │   └── recommendation.html  # Lawyer recommendation search UI (calls /recommend)
        │
        ├── app_main.py              # Flask backend — routes, Firebase auth, /ask + /recommend logic
        ├── ColabHost.py             # Colab-side API server exposing /generate for chatbot
        ├── ColabHost.ipynb          # Notebook version of ColabHost.py (for Google Colab execution)
        ├── ServiceKey.json          # Firebase Admin SDK credentials (Replace with your firebase)
        └── Dummy_Lawyers.csv        # Lawyer dataset used for recommendations
```

---

## 🛠️ Tech Stack

- **Models & AI**:  
  - Hugging Face Transformers, PEFT, Accelerate, BitsAndBytes  
  - Llama-2-7b-chat (fine-tuned on custom made dataset)  
  - TensorFlow Hub (Universal Sentence Encoder)  
  - Scikit-learn (cosine similarity)

- **Backend**: Flask, Firebase Admin SDK  
- **Frontend**: Bootstrap 5, jQuery, HTML5/CSS3  
- **Auth & Hosting**: Firebase, ngrok (Colab mode)  
- **Environment**: Conda / pip (requirements provided)  

---


## 🧰 Prerequisites
- **Python 3.11 or higher**
- (Local) **pip / conda**  
- **Firebase Admin service account** JSON (`ServiceKey.json`)
- (Chatbot) **Google Colab** account + **ngrok** token 

---

## 🔧 Setup & Run

### 1) Backend (Flask)
```bash
cd WebsiteFinal/New
# Create/activate env (example with pip)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
### 2) Create conda environment

- Option A: Using environment.yml (recommended)
```bash
conda env create -f environment.yml
conda activate lawyer
```

- Option B: Manual setup
```bash
conda create -n lawyer python=3.11 -y
conda activate lawyer
pip install -r requirements.txt
```

3. Firebase Setup
- Place your `ServiceKey.json` (Firebase Admin SDK key) inside `WebsiteFinal/New/`.


### 3) Start the Colab Chatbot API
- Open **Colab** and run `ColabHost.ipynb`
- The notebook:
  - Installs `flask`, `pyngrok`, and Hugging Face libs  
  - Loads `TejasKoti/Llama-2-7b-lawyer-chat` (4‑bit)  
  - Starts a Flask server with **`/generate`** and exposes it via **ngrok**; copy the printed public URL.

### 4) Point Flask to Colab
In `app_main.py`, set the Colab URL:
```python
COLAB_API_URL = "https://<your-ngrok-subdomain>/generate"
```
This route is what `/ask` will call when users chat with the chatbot.

### 5) Run Flask
```bash
python app_main.py
```
Open http://127.0.0.1:5000 — you’ll see the **landing page** → Login and Explore

### ~For Linux / High-VRAM (Local GPU)
1. Ensure you have enough GPU VRAM (≥24GB recommended).  

2. Create environment and install requirements like above 

3. Run (in the `WebsiteFinal/Old/` folder):
   ```bash
   python app_linux.py
   ```
4. Open `http://127.0.0.1:5000/`.

---

## 🧪 API Reference (Local Flask)

| Method | Route | Body | Returns | Notes |
|---|---|---|---|---|
| GET | `/` | – | HTML | Landing |
| GET/POST | `/login` | form: `email`, `password` | Redirect/HTML | Firebase auth |
| POST | `/signin` | form: `email`, `password` | Redirect | Creates Firebase user |
| GET | `/logout` | – | Redirect | Clears session |
| GET | `/chat` | – | HTML | Main chat UI |
| GET | `/recommendation` | – | HTML | Recommendation UI |
| POST | `/ask` | JSON/FORM: `question` | JSON `{response}` | Proxies to Colab `/generate` with timeouts & helpful errors |
| POST | `/recommend` | FORM: `query` | JSON `{lawyers:[…]}` | USE similarity + filters; top 5 results |

---

## ⚙️ Configuration & Customization
- **Colab model**: Replace `TejasKoti/Llama-2-7b-lawyer-chat` with your model in the notebook. Generation params (temperature, top_p, penalties, min/max tokens) are already tuned for long, clean answers and headings within my Colab Notebook.
- **Recommendation columns**: CSV must contain `Name`, `Domain`, `Years of active experience`, `Price`, `Client satisfaction (out of 10)`.
- **Currencies**: Responses format price as `${Price}` — adjust in `recommend_lawyers()` if needed.
- **Domains**: Any domain present in the CSV can be matched directly from the user query (case-insensitive).

---

## 🩹 Troubleshooting

- **Colab timeout / no response**: Flask handles long responses (120s) and shows user-friendly messages for timeouts/connection issues. Ensure your ngrok tunnel is active and `COLAB_API_URL` is correct.  
- **Login fails**: Verify your **ServiceKey.json** is valid and not expired; ensure Firebase Email/Password auth is enabled in your Firebase project.  
- **“No lawyers found”**: Your query might filter everything out (e.g., strict budget + domain). Try broader text; the endpoint returns a helpful message when empty.
- **Sign-up not working**: Update the client to post to `/signin` or add a `/signup` Flask route alias or create a login within Firebase itself.

---

### Changelog (New vs Old)
- Consolidated Flask entry into `app_main.py` with explicit `/ask` and `/recommend` endpoints.  
- Modernized UI/UX: **search-in-chat**, **folders**, **export transcript**, fixed dropdown clipping/z-index.  
- Recommendation engine cleaned up with budget/experience/domain filters + top‑5 cosine-similarity results.

---

## ⚠️ Notes

- Running Llama-2 locally requires **large VRAM (24GB+)**.  
- For lightweight devices, use **ColabHost + ngrok**.
