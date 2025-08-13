from typing import List, Dict, Optional
import json
from pathlib import Path

def fetch_from_jsonl(path: str, max_items: int = 200) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    items: List[Dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("content") or ""
            items.append({
                "source": "social_x_local",
                "title": (text[:80] + "...") if len(text) > 80 else text,
                "body": text,
                "url": obj.get("url"),
                "published_at": obj.get("created_at"),
            })
            if len(items) >= max_items:
                break
    return items


