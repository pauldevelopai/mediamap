import os
HF_ZERO_SHOT_MODEL = os.getenv("DS_HF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
HF_SUMMARY_MODEL   = os.getenv("DS_HF_SUMMARY_MODEL", "facebook/bart-large-cnn")
HF_EMBED_MODEL     = os.getenv("DS_HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ZERO_SHOT_CONFIDENCE = float(os.getenv("DS_HF_ZERO_SHOT_CONF", "0.5")) if os.getenv("DS_HF_ZERO_SHOT_CONF") else float(os.getenv("DS_ZERO_SHOT_CONF", "0.5"))
DEDUP_SIM_THRESHOLD  = float(os.getenv("DS_DEDUP_SIM", "0.88"))
MAX_SUMMARY_INPUT_CHARS = int(os.getenv("DS_MAX_SUMMARY_INPUT", "1800"))
DB_PATH = os.getenv("DATASAFE_DB", "datasafe.db")
FINANCE_KEYWORDS = [k.lower() for k in [
    "absa","standard bank","fnb","first national bank","nedbank","capitec",
    "tymebank","discovery bank","investec","old mutual","sanlam","momentum",
    "bank","atm","swift","visa","mastercard","payment","fintech","interbank","core banking"
]]
KEEP_NON_FINANCE = os.getenv("DS_KEEP_NON_FINANCE", "false").lower() in {"1","true","yes","y"}
