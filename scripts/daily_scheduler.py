
import time
import subprocess
import schedule
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SCHEDULER - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reports/scheduler.log")
    ]
)
logger = logging.getLogger(__name__)

def run_job():
    today_str = datetime.now().strftime("%Y%m%d")
    logger.info(f"⏰ Starting daily job for {today_str}...")
    
    # Check if DISCORD_WEBHOOK_URL is set
    if not os.environ.get("DISCORD_WEBHOOK_URL"):
        logger.warning("DISCORD_WEBHOOK_URL not set! Notification will fail.")

    try:
        # Run predict_combined_formation.py (V13/V14 Logic)
        # We assume this is running inside the container where python is available
        cmd = ["python", "scripts/predict_combined_formation.py", "--discord"]
        
        # Determine if we should force regenerate (optional, maybe on weekends?)
        # cmd.append("--force_regenerate") 
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Job finished successfully.")
            logger.info(result.stdout)
        else:
            logger.error("❌ Job failed.")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"An error occurred during job execution: {e}")

def main():
    logger.info("🚀 Keiiba-AI Daily Scheduler Started")
    logger.info("Waiting for trigger time (00:00 UTC = 09:00 JST)...")
    
    # Schedule the job every day at 00:00 UTC (matches 09:00 JST)
    schedule.every().day.at("00:00").do(run_job)
    
    # Also run immediately if launch argument is provided for testing
    if "--run-now" in sys.argv:
        run_job()

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Ensure reports dir exists
    os.makedirs("reports", exist_ok=True)
    main()
