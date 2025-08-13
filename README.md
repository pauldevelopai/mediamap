## DataSafe (HF-powered) Quickstart

Run Streamlit Dashboard:

```bash
streamlit run datasafe/app_streamlit/app.py
```

Run FastAPI API:

```bash
uvicorn datasafe.app_api.main:app --reload
```

Seed data:

```bash
python -m datasafe.scripts.ingest_media
```

Environment example: see `datasafe/secrets.example.env`.

# DataSafe Hook PLUS

Why was the previous zip small? Because it contains **code only** — models auto‑download from Hugging Face at first run (hundreds of MB). This pack adds **more collectors** and a **config switch** to keep non‑finance items while you test.

## Install
pip install -r requirements.txt

## Ingest (seed your DB)
# Keep everything while testing:
export DS_KEEP_NON_FINANCE=true
python scripts/ingest_finance.py

# Later (finance‑only):
unset DS_KEEP_NON_FINANCE
python scripts/ingest_finance.py

Optional env:
- PHISHTANK_CSV_URL=https://data.phishtank.com/data/<key>/online-valid.csv
- NVD_API_KEY=<your_key> (optional)
- NVD_QUERY="bank OR payment OR visa OR mastercard"
- DS_RSS=https://www.itweb.co.za/rss/categories/security

Models download on first run — expect a big download.
