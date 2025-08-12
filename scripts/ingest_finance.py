from datasafe.collectors import urlhaus, phishtank, cisa_kev, nvd_cve, rss_generic
from datasafe.pipeline.normalize import RawItem, normalize
from datasafe.storage.sqlite_store import ensure_schema, insert_record
import os

def run_ingest(limit=200):
    ensure_schema()
    items = []
    for collector, args in [
        (urlhaus.collect, {"limit": limit//4}),
        (phishtank.collect, {"limit": limit//4}),
        (cisa_kev.collect, {"limit": limit//4}),
        (lambda **_: nvd_cve.collect(query=os.getenv("NVD_QUERY","bank OR payment OR visa OR mastercard"), limit=limit//4), {}),
    ]:
        try:
            items += collector(**args) if args else collector()
        except Exception as e:
            print(f"[WARN] Collector failed: {collector.__name__} —", e)

    feed = os.getenv("DS_RSS")
    if feed:
        try:
            from datasafe.collectors import rss_generic
            items += rss_generic.collect(feed, limit=50)
        except Exception as e:
            print("[WARN] RSS:", e)

    saved = 0; seen = 0
    for d in items:
        seen += 1
        raw = RawItem(source=d["source"], title=d.get("title") or "", body=d.get("body") or "", url=d.get("url"), published_at=d.get("published_at"))
        n = normalize(raw)
        if n:
            insert_record(n.__dict__); saved += 1
    print(f"[ingest] items_seen={seen} saved_finance_related={saved}")

if __name__ == "__main__":
    run_ingest()
