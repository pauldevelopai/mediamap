import requests, os
from typing import List, Dict
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
def collect(query:str="bank OR payment OR visa OR mastercard", limit:int=100) -> List[Dict]:
    params = {"keywordSearch": query, "resultsPerPage": min(limit, 2000)}
    api_key = os.getenv("NVD_API_KEY")
    headers = {"apiKey": api_key} if api_key else {}
    r = requests.get(NVD_API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = []
    for v in data.get("vulnerabilities", [])[:limit]:
        cve = v.get("cve",{}).get("id")
        desc = ""
        for d in v.get("cve",{}).get("descriptions",[]):
            if d.get("lang")=="en":
                desc = d.get("value",""); break
        items.append({
            "source":"NVD",
            "title": cve or "NVD CVE",
            "body": desc,
            "url": f"https://nvd.nist.gov/vuln/detail/{cve}" if cve else "",
            "published_at": v.get("cve",{}).get("published")
        })
    return items
