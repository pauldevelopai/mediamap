from ..agents.worker import ingest

if __name__ == "__main__":
    results = ingest()
    print(f"Inserted {len(results)} records")


