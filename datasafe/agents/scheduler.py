from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from .worker import ingest
from ..config import TZ

def main():
    sch=BlockingScheduler(timezone=TZ)
    sch.add_job(lambda: ingest(200), CronTrigger(minute="*/30"))
    print(f"[scheduler] running in {TZ}. Ctrl+C to stop.")
    sch.start()

if __name__=="__main__":
    main()
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from .worker import ingest

def job(name: str):
    print(f"[{datetime.utcnow().isoformat()}] Running job: {name}")
    try:
        results = ingest()
        print(f"[{name}] Ingested {len(results)} records")
    except Exception as e:
        print(f"[{name}] Error: {e}")

if __name__ == "__main__":
    sched = BlockingScheduler()
    # every 15 min
    sched.add_job(lambda: job("openphish/urlhaus/rss"), "interval", minutes=15)
    # every 30 min
    sched.add_job(lambda: job("rss_media"), "interval", minutes=30)
    # hourly
    sched.add_job(lambda: job("social_x"), "interval", hours=1)
    print("DataSafe scheduler started")
    sched.start()


