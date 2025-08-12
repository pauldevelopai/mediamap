import feedparser
from typing import List, Dict
def collect(feed_url: str, limit:int=50) -> List[Dict]:
    d = feedparser.parse(feed_url)
    out = []
    for e in d.entries[:limit]:
        out.append({
            "source": f"RSS:{d.feed.get('title','rss')}",
            "title": e.get("title",""),
            "body": e.get("summary","") or e.get("description","") or "",
            "url": e.get("link"),
            "published_at": e.get("published") or e.get("updated")
        })
    return out
