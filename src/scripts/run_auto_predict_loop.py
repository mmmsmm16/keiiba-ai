import time
import subprocess
import sys
import os
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [LOOP] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_auto_predict():
    """auto_predict.py を実行する"""
    script_path = os.path.join(os.path.dirname(__file__), 'auto_predict.py')
    cmd = [sys.executable, script_path]
    
    logger.info("Executing auto_predict.py...")
    try:
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        if result.returncode == 0:
            logger.info("auto_predict.py finished successfully.")
        else:
            logger.error(f"auto_predict.py failed with exit code {result.returncode}")
            
    except Exception as e:
        logger.error(f"Failed to execute script: {e}")

def main():
    logger.info("Starting Auto Predict Loop (Interval: 10 mins)")
    
    try:
        while True:
            # Check time: Only run between 8:00 and 17:00 (JRA race hours generally 9:30 - 16:30)
            # But let's keep it simple for now and run always or maybe loosely filtered.
            # JRA races are usually 9:00 - 17:00.
            
            now = datetime.now()
            if 8 <= now.hour <= 17:
                run_auto_predict()
            else:
                logger.info("Outside of race hours (08:00-17:00). Sleeping...")
            
            # Sleep 10 minutes
            time.sleep(600)
            
    except KeyboardInterrupt:
        logger.info("Loop stopped by user.")

if __name__ == "__main__":
    main()
