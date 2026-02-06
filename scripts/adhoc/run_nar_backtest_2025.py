
import os
import subprocess
import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 12, 17) # Today in simulation context

def run_command(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd}")
        return False

def main():
    logger.info(f"Starting Backtest from {START_DATE} to {END_DATE}")
    
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime('%Y-%m-%d')
        logger.info(f"Processing {date_str}...")
        
        # 1. Run Pipeline
        cmd_pipeline = f"python src/nar/run_production_pipeline.py {date_str}"
        if not run_command(cmd_pipeline):
            # Might fail if no races, continue
            pass
            
        # 2. Run Settlement
        # Only run if ledger exists? Settle script handles empty gracefully?
        # Check if ledger was created to save time
        ledger_path = f"reports/nar/ledgers/{current_date.strftime('%Y%m%d')}_ledger.parquet"
        if os.path.exists(ledger_path):
             cmd_settle = f"python src/nar/settle_paper_trades.py --date {date_str}"
             run_command(cmd_settle)
        else:
             logger.info(f"No ledger for {date_str}, skipping settlement.")
        
        current_date += datetime.timedelta(days=1)
        
    logger.info("Backtest Complete. Check reports/nar/daily/ for details.")
    
    # Optional: Aggregate
    # Could load all daily reports or re-calc from some log using settle script capability if it supported range.
    # For now, just run user through the daily flow.

if __name__ == "__main__":
    main()
