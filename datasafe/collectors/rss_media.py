import os, feedparser

def collect(limit=100):
    rss = os.getenv("MEDIA_RSS","https://www.itweb.co.za/rss/categories/security")
    feed = feedparser.parse(rss)
    items = []
    for e in feed.entries[:limit]:
        items.append({
          "source":"MediaRSS",
          "title":e.get("title",""),
          "body":(e.get("summary") or "") + "\n" + (e.get("title") or ""),
          "url":e.get("link"),
          "published_at":e.get("published") or e.get("updated") or ""
        })
    return items
import feedparser
import time
from typing import List, Dict, Optional
from ..config import MEDIA_RSS

def fetch(max_items: int = 200, rss_url: Optional[str] = None) -> List[Dict]:
    url = rss_url or MEDIA_RSS
    feed = feedparser.parse(url)
    items: List[Dict] = []
    for entry in feed.entries[:max_items]:
        items.append({
            "source": url,
            "title": entry.get("title", ""),
            "body": entry.get("summary", ""),
            "url": entry.get("link"),
            "published_at": entry.get("published", entry.get("updated"))
        })
    return items


