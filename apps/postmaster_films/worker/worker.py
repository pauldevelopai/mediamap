"""RQ Worker for background job processing"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import redis
    from rq import Worker, Queue, Connection
    from apps.postmaster_films.backend.settings import get_settings
    
    settings = get_settings()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Redis connection
    redis_conn = redis.from_url(settings.REDIS_URL)
    
    # Queue configuration
    listen = ['default', 'video_generation', 'audio_processing']
    
    def main():
        """Main worker function"""
        logger.info("Starting Postmaster Films worker...")
        logger.info(f"Listening on queues: {listen}")
        logger.info(f"Redis URL: {settings.REDIS_URL}")
        
        with Connection(redis_conn):
            worker = Worker(map(Queue, listen))
            try:
                worker.work()
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
            except Exception as e:
                logger.error(f"Worker error: {e}")
                raise
    
    if __name__ == '__main__':
        main()

except ImportError as e:
    print(f"Redis/RQ not available: {e}")
    print("Install with: pip install redis rq")
    print("Or run jobs synchronously via FastAPI background tasks")
    sys.exit(1)
except Exception as e:
    print(f"Worker startup failed: {e}")
    sys.exit(1)

