import os, csv, io, requests
from typing import List, Dict
PHISHTANK_URL = os.getenv("PHISHTANK_CSV_URL", "https://data.phishtank.com/data/online-valid.csv")
def collect(limit: int = 200) -> List[Dict]:
    r = requests.get(PHISHTANK_URL, timeout=30)
    r.raise_for_status()
    data = r.content.decode("utf-8", errors="ignore")
    rows = []
    for i, row in enumerate(csv.DictReader(io.StringIO(data))):
        if i >= limit: break
        rows.append({
            "source": "PhishTank",
            "title": row.get("phish_detail_url") or "PhishTank entry",
            "body": f"URL: {row.get('url')} | Target: {row.get('target') or ''}",
            "url": row.get("url"),
            "published_at": row.get("submission_time"),
        })
    return rows
