import time
import subprocess
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_scheduler(interval_minutes: int = 10):
    logger.info(f"🚀 Scheduler started. Running auto_predict.py every {interval_minutes} minutes.")
    
    script_path = "src/scripts/auto_predict.py"
    
    while True:
        try:
            now = datetime.now()
            logger.info(f"⏰ Triggering auto_predict.py at {now.strftime('%H:%M:%S')}...")
            
            # Run the command inside Docker
            # Assuming this script is also run inside Docker container 'app'
            # If running from host, command should be "docker compose exec app python ..."
            
            # Since user runs manual commands via `docker compose exec app python ...`,
            # this script is likely intended to run INSIDE the container for simplicity?
            # Or OUTSIDE on the host?
            
            # If running INSIDE container:
            cmd = ["python", script_path]
            
            # If running OUTSIDE (Host):
            # cmd = ["docker", "compose", "exec", "app", "python", script_path]
            
            # Let's detect environment or provide robust method.
            # Assuming INSIDE container execution for daemon mode described below.
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ auto_predict.py finished successfully.")
                # Show simpler output (last few lines)
                lines = result.stdout.strip().split('\n')
                if lines:
                    logger.info(f"Output tail: {lines[-1]}")
            else:
                logger.error(f"❌ auto_predict.py failed with code {result.returncode}")
                logger.error(f"Error output:\n{result.stderr}")
                
        except Exception as e:
            logger.error(f"⚠️ Scheduler error: {e}")
        
        # Sleep
        logger.info(f"💤 Sleeping for {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    try:
        run_scheduler(10) # 10 minutes interval
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user.")
