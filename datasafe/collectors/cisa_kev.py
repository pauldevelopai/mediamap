import requests
from typing import List, Dict
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
def collect(limit:int=200) -> List[Dict]:
    r = requests.get(CISA_KEV_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = []
    for v in data.get("vulnerabilities", [])[:limit]:
        cve = v.get("cveID")
        name = (v.get("vendorProject","") + " " + (v.get("product","") or "")).strip()
        desc = v.get("vulnerabilityName","") or v.get("shortDescription","")
        items.append({
            "source":"CISA KEV",
            "title": f"{cve} — {name}".strip(),
            "body": f"{desc} | Known exploited vulnerability. Due date: {v.get('dueDate','N/A')}",
            "url": v.get("notes") or "",
            "published_at": v.get("dateAdded")
        })
    return items
