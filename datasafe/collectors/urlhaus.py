import requests, csv, io
from typing import List, Dict
URLHAUS_CSV = "https://urlhaus.abuse.ch/downloads/csv_recent/"
def collect(limit: int = 200) -> List[Dict]:
    r = requests.get(URLHAUS_CSV, timeout=30)
    r.raise_for_status()
    text = r.content.decode("utf-8", errors="ignore")
    buff = io.StringIO(text)
    rows = []
    for line in buff:
        if line.startswith("#"): continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6: continue
        url = parts[2]; threat = parts[4]
        rows.append({
            "source": "URLhaus",
            "title": f"URLhaus: {threat}",
            "body": f"URL: {url} | Threat: {threat}",
            "url": url,
            "published_at": parts[0] if parts else None
        })
        if len(rows) >= limit: break
    return rows
