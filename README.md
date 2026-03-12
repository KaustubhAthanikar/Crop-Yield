# 🌾 Pune Rice Yield Intelligence — Streamlit App

## Deploy to Streamlit Cloud (FREE) in 5 minutes

### Step 1 — Push to GitHub
```bash
cd pune-streamlit
git init
git add .
git commit -m "Pune yield app"
```
Create a new repo at https://github.com/new, then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/pune-yield-app.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → Branch: `main` → Main file: `app.py`
5. Click **Deploy** ✅

Your app will be live at:
`https://YOUR_USERNAME-pune-yield-app-app-XXXX.streamlit.app`

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
```
pune-streamlit/
├── app.py                  ← Main Streamlit app
├── yield_model.pkl         ← Trained GBM model
├── district_yields.csv     ← District yield data (2019–2024)
├── requirements.txt        ← Python dependencies
└── .streamlit/
    └── config.toml         ← Dark theme config
```

## Features
- 📍 Interactive Mapbox map of all 14 Pune districts
- 📈 Year-wise trend lines per district
- 📊 Ranked bar chart + heatmap + full data table
- 🔮 Live yield predictor — input lat/lon + GEE spectral indices
- Fully dark-themed, mobile-friendly
