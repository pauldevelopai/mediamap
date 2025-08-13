from typing import List, Dict
from pathlib import Path

def fetch_from_folder(folder: str, max_items: int = 200) -> List[Dict]:
    items: List[Dict] = []
    p = Path(folder)
    if not p.exists():
        return items
    for file in sorted(p.glob("*.txt"))[:max_items]:
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        items.append({
            "source": "abuse_folder",
            "title": file.stem,
            "body": text,
            "url": None,
            "published_at": None,
        })
    return items


