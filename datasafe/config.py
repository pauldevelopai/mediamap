import os

DB_PATH = os.getenv("DATASAFE_DB", "datasafe.db")
TZ = os.getenv("DATASAFE_TZ", "Africa/Johannesburg")

HF_ZERO_SHOT_MODEL = os.getenv("DS_HF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
HF_SUMMARY_MODEL   = os.getenv("DS_HF_SUMMARY_MODEL", "facebook/bart-large-cnn")
HF_EMBED_MODEL     = os.getenv("DS_HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ZERO_SHOT_CONFIDENCE = float(os.getenv("DS_HF_ZERO_SHOT_CONF", os.getenv("DS_ZERO_SHOT_CONF", "0.55")))
DEDUP_SIM_THRESHOLD  = float(os.getenv("DS_DEDUP_SIM", "0.90"))
MAX_SUMMARY_INPUT_CHARS = int(os.getenv("DS_MAX_SUMMARY_INPUT", "1800"))

MEDIA_KEYWORDS = [k.lower() for k in [
  "newsroom","publisher","journalist","editor","investigative","cms","wordpress","drupal","joomla",
  "cloudflare","ddos","source protection","whistleblower","press freedom"
]]

ACTION_PLAYBOOK = {
  "Phishing": [
      "Block URLs/domains in email/web filter",
      "Send staff warning w/ screenshots",
      "Add domains to brand‑impersonation watchlist"
  ],
  "Ransomware": [
      "Verify offline backups (tested restores)",
      "Patch exposed RDP/VPN; disable unused remote access",
      "Harden endpoint policies; EDR in monitor+block"
  ],
  "Credential leak": [
      "Force reset on affected accounts; enforce MFA",
      "Invalidate tokens/sessions; search reuse",
      "Run targeted phishing awareness for team"
  ],
  "DDoS": [
      "Enable CDN/DDoS shield (eg Cloudflare Under Attack)",
      "Rate-limit origins; warm static fallback",
      "Coordinate w/ hosting for traffic scrubbing"
  ],
  "Harassment": [
      "Escalate to security point person",
      "Document evidence; brief staff on doxxing responses",
      "If threats credible: notify legal & law enforcement"
  ]
}
import os

# Database and runtime
DB_PATH = (
    os.getenv("DATASAFE_DB")
    or os.getenv("DATASAFE_DB_PATH")
    or "datasafe.db"
)
TZ = os.getenv("DATASAFE_TZ", "Africa/Johannesburg")

# Hugging Face models
HF_ZERO_SHOT_MODEL = os.getenv("DS_HF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
HF_SUMMARY_MODEL   = os.getenv("DS_HF_SUMMARY_MODEL", "facebook/bart-large-cnn")
HF_EMBED_MODEL     = os.getenv("DS_HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ZERO_SHOT_CONFIDENCE = (
    float(os.getenv("DS_HF_ZERO_SHOT_CONF", "0.5")) if os.getenv("DS_HF_ZERO_SHOT_CONF")
    else float(os.getenv("DS_ZERO_SHOT_CONF", "0.5"))
)
DEDUP_SIM_THRESHOLD  = float(os.getenv("DS_DEDUP_SIM", "0.90"))
MAX_SUMMARY_INPUT_CHARS = int(os.getenv("DS_MAX_SUMMARY_INPUT", "1800"))

# Sector keyword seeds (expandable)
FINANCE_KEYWORDS = [k.lower() for k in [
    "absa","standard bank","fnb","first national bank","nedbank","capitec",
    "tymebank","discovery bank","investec","old mutual","sanlam","momentum",
    "bank","atm","swift","visa","mastercard","payment","fintech","interbank","core banking"
]]

MEDIA_KEYWORDS = [k.lower() for k in [
    "newsroom", "editor", "journalist", "publisher", "broadcast", "investigative",
    "wordpress", "drupal", "joomla", "cms", "cloudflare", "ddos"
]]

# Filtering toggles
KEEP_NON_FINANCE = os.getenv("DS_KEEP_NON_FINANCE", "false").lower() in {"1","true","yes","y"}
KEEP_NON_MEDIA = os.getenv("DS_KEEP_NON_MEDIA", "false").lower() in {"1","true","yes","y"}

# Risk knobs
DIGITAL_RISK_WEIGHTS = {"phishing":1, "malware":1.5, "doxxing":2, "credential_leak":2.5, "ddos":1.5, "cve_critical":2}
PHYSICAL_RISK_WEIGHTS = {"targeted_harassment":2, "threat_of_violence":3, "geotagged_tracking":3, "doxxing":2.5}

# Collectors defaults
OPENPHISH_URL = os.getenv("OPENPHISH_URL", "https://openphish.com/feed.txt")
URLHAUS_RECENT = os.getenv("URLHAUS_RECENT", "https://urlhaus.abuse.ch/downloads/csv_recent/")
MEDIA_RSS = os.getenv("MEDIA_RSS", "https://www.itweb.co.za/rss/categories/security")

# Alerts
ALERT_EMAIL_TO = os.getenv("DS_ALERT_EMAIL_TO", "")
SIGNAL_NUMBER  = os.getenv("DS_SIGNAL_NUMBER", "")
