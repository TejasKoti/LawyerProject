# ⚖️ Lawyer Chatbot & Recommendation Website

An AI-powered web application that combines a **fine-tuned Llama-2 legal chatbot** with a **lawyer recommendation engine**.  
The system allows users to:  
- Ask legal questions and receive AI-generated answers.  
- Search and filter lawyers by domain, experience, and budget.  
- Log in/sign up securely using Firebase authentication.  
- Run in **two modes**: lightweight (Colab-hosted model) or full local GPU deployment.

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

## 📂 Project Structure

```
LawyerProject/
│
├── Development/                # Research & experimental code
│   ├── Chatbot Development/    # Fine-tuning Llama 2 chatbot
│   ├── Dataset Extraction/     # Website Lawyer dataset scripts
│   ├── Dumped Code/            # Scratch / Testing and Graph dump
│   └── Recommendation Development/ # Prototypes for recommendations
│
├── WebsiteFinal/               # Production-ready website
│   ├── templates/              # Frontend templates (login.html, index.html)
│   ├── app_windows.py          # Flask app (Windows + Colab backend)
│   ├── app_linux.py            # Flask app (Linux, local GPU inference)
│   ├── ColabHost.py            # Flask+ngrok API for chatbot on Colab
│   ├── ColabHost.ipynb         # Notebook version of above
│   ├── Dummy_Lawyers.csv       # Sample dataset for lawyers
│   ├── ServiceKey.json         # Firebase credentials (Replace with your firebase)
│   ├── requirements.txt        # Python deps backup (pip)
│   └── environment.yml         # Conda environment setup
│
└── README.md                   # This file
```

---

## ⚙️ Installation

### A) Windows Setup (with Google Colab Offload)
1. Clone the repository
```bash
git clone https://github.com/yourusername/LawyerProject.git
cd LawyerProject/WebsiteFinal
```

2. Create conda environment

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
- Place your `ServiceKey.json` (Firebase Admin SDK key) inside `WebsiteFinal/`.

4. Launch `ColabHost.ipynb` on Google Colab (starts chatbot API and copy the API URL).  

5. Start Flask app on VSCode (Change Colab API URL):
   ```bash
   python app_windows.py
   ```
6. Open `http://127.0.0.1:5000/`.

### B) Linux / High-VRAM (Local GPU)
1. Ensure you have enough GPU VRAM (≥24GB recommended).  

2. Create environment and install requirements like above 

3. Run:
   ```bash
   python app_linux.py
   ```
4. Open `http://127.0.0.1:5000/`.

---

## 📊 Dataset

- **`Dummy_Lawyers.csv`** → demo dataset with:  
  - Name  
  - Domain/Specialization  
  - Years of Experience  
  - Price (fees)  
  - Client Satisfaction Score  

Used for recommendation testing.

---

## 📸 Screenshots

- **Login Page**  
  ![Login](templates/login.png)  

- **Dashboard (Chatbot + Lawyer Recs)**  
  ![Dashboard](templates/index.png)  

---

## ⚠️ Notes

- Running Llama-2 locally requires **large VRAM (24GB+)**.  
- For lightweight devices, use **ColabHost + ngrok**.  