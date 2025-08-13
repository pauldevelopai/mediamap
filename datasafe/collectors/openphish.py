import os, requests

def collect(limit=200):
    url = os.getenv("OPENPHISH_URL","https://openphish.com/feed.txt")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    out = []
    for u in lines[:limit]:
        out.append({"source":"OpenPhish","title":"Phishing URL","body":f"URL: {u}","url":u,"published_at":""})
    return out
import requests
from typing import List, Dict
from ..config import OPENPHISH_URL

def fetch(max_items: int = 200) -> List[Dict]:
    resp = requests.get(OPENPHISH_URL, timeout=15)
    resp.raise_for_status()
    lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
    items: List[Dict] = []
    for url in lines[:max_items]:
        items.append({
            "source": "openphish",
            "title": f"OpenPhish URL",
            "body": f"URL: {url}",
            "url": url,
            "published_at": None,
        })
    return items


