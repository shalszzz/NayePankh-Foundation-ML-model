# 🕊️ NayePankh ML Intelligence Suite — Setup Guide

## What's in this project

```
nayepankh_ml/
├── Home.py                          ← Main app homepage
├── requirements.txt                 ← All Python packages needed
├── data/
│   ├── generate_data.py             ← Generates all 3 datasets (run once)
│   ├── donors.csv                   ← 500 donor records (auto-generated)
│   ├── campaigns.csv                ← 120 campaign records (auto-generated)
│   └── districts.csv                ← 75 UP districts (auto-generated)
└── pages/
    ├── 1_💰_Donor_Predictor.py      ← Module 1: Donor retention ML
    ├── 2_📣_Campaign_Classifier.py  ← Module 2: Campaign impact classifier
    ├── 3_🗺️_Expansion_Heatmap.py   ← Module 3: District clustering map
    └── 4_💡_Insights_&_Recommendations.py ← Analysis & website audit
```

---

## Step-by-Step Setup

### Step 1 — Install Python (if you don't have it)
Download Python 3.11 from https://python.org/downloads
✅ Check "Add Python to PATH" during install.

### Step 2 — Open the project in VS Code
1. Unzip the downloaded folder
2. Open VS Code → File → Open Folder → select `nayepankh_ml`

### Step 3 — Open the Terminal in VS Code
Press **Ctrl + `** (backtick) to open the terminal.

### Step 4 — Create a virtual environment (recommended)
```bash
python -m venv venv
```
Then activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll see `(venv)` appear in your terminal. Good.

### Step 5 — Install all dependencies
```bash
pip install -r requirements.txt
```
This installs: streamlit, scikit-learn, xgboost, plotly, pandas, numpy, folium.
Takes ~2–3 minutes. Wait for it to finish.

### Step 6 — Generate the datasets
```bash
python data/generate_data.py
```
You should see:
```
✅ donors.csv — 500 rows
✅ campaigns.csv — 120 rows
✅ districts.csv — 75 rows
🎉 All datasets ready in /data/
```

### Step 7 — Run the app!
```bash
streamlit run Home.py
```
Your browser will automatically open to `http://localhost:8501`
You'll see the full 4-page ML dashboard.

---

## Step 8 — Deploy to Streamlit Cloud (get your demo link)

1. Create a free account at https://streamlit.io/cloud (sign in with GitHub)
2. Push this folder to a new GitHub repo:
   ```bash
   git init
   git add .
   git commit -m "NayePankh ML Intelligence Suite"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/nayepankh-ml.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io → New App
4. Select your repo, branch: `main`, file: `Home.py`
5. Click **Deploy** — live link in ~2 minutes!

That link is your submission demo URL. 🎉

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: donors.csv` | Run `python data/generate_data.py` first |
| Map not showing | Make sure `plotly` is installed; try `pip install plotly==5.22.0` |
| Port already in use | Run `streamlit run Home.py --server.port 8502` |

---

## What to mention in your submission email

> "I built a 3-module ML Intelligence Suite specifically for NayePankh, covering donor retention prediction (XGBoost/Random Forest), campaign impact classification before launch, and district-level expansion prioritisation using K-Means clustering on all 75 UP districts. I also included a full website audit and 90-day ML roadmap. Live demo: [your link]"
