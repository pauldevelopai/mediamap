from ..agents.worker import ingest

if __name__ == "__main__":
    results = ingest()
    print(f"Backfill inserted {len(results)} records")


