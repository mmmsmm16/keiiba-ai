"""
JIT Scheduler
=============
Monitors today's races and triggers prediction 10 minutes before start time.
Design:
1. On startup (and periodically, e.g., every 1 hour):
   - Load today's race list from DB (JraVanDataLoader).
2. Schedule jobs:
   - For each race, calculate trigger_time = start_time - 10 min.
   - If trigger_time > now, schedule the job.
3. Execution loop:
   - Uses `schedule` library to check pending jobs every 10 seconds.
   - Job executes `production_run_t2_jit.py --race_id X --discord`.
"""

import time
import subprocess
import schedule
import logging
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.preprocessing.loader import JraVanDataLoader

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - JIT_SCHEDULER - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reports/jit_scheduler.log")
    ]
)
logger = logging.getLogger(__name__)

SCHEDULED_RACES = set()

def run_prediction(race_id, venue_name, race_number, date_str=None):
    """Execute prediction for a single race"""
    logger.info(f"⚡ Triggering Prediction for {venue_name} {race_number}R (ID: {race_id})")
    
    cmd = ["python", "-u", "scripts/predict_combined_formation.py", "--race_id", str(race_id), "--discord"]
    if date_str:
        cmd.extend(["--date", date_str])
    
    # Ensure environment variables are passed (especially DISCORD_WEBHOOK_URL)
    env = os.environ.copy()
    
    try:
        # Run in subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            logger.info(f"✅ Prediction success for {race_id}")
            # Log the output (prediction table)
            logger.info("--- Prediction Output ---")
            if result.stdout.strip():
                logger.info("\n" + result.stdout)
            else:
                logger.warning("No output captured from prediction script (stdout is empty).")
                if result.stderr:
                    logger.info("Stderr content:\n" + result.stderr)
            logger.info("-------------------------")
        else:
            logger.error(f"❌ Prediction failed for {race_id}")
            logger.error(result.stderr)
            logger.error(result.stdout) # Log stdout too, might contain error details

            
    except Exception as e:
        logger.error(f"Error executing prediction: {e}")

def run_cache_update():
    """Execute daily incremental feature update"""
    logger.info("🔄 Starting Daily Feature Update (update_daily_features.py)...")
    cmd = ["python", "scripts/update_daily_features.py"]
    try:
        # 20 minute timeout (full feature pipeline can take time)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode == 0:
            logger.info("✅ Feature Update Completed Successfully.")
        else:
            logger.error("❌ Feature Update Failed.")
            logger.error(result.stderr)
    except Exception as e:
        logger.error(f"Error executing feature update: {e}")

def get_jst_now():
    return datetime.utcnow() + timedelta(hours=9)

def update_schedule():
    """Fetch today's races and update schedule"""
    # Use JST to determine 'today'
    now_jst = get_jst_now()
    date_cond = now_jst.strftime("%Y%m%d")
    
    logger.info(f"🔄 Updating race schedule for date (JST): {date_cond}...")
    
    try:
        loader = JraVanDataLoader()
        query = f"""
        SELECT 
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
            keibajo_code,
            race_bango,
            hasso_jikoku
        FROM jvd_ra
        WHERE kaisai_nen = '{date_cond[:4]}' 
          AND kaisai_tsukihi = '{date_cond[4:]}'
        """
        
        df = pd.read_sql(query, loader.engine)
        
        if df.empty:
            logger.info("No races found for today (or yet).")
            return

        VENUE_MAP = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
            '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
        }
        
        now_utc = datetime.utcnow()
        count = 0
        
        for _, row in df.iterrows():
            rid = row['race_id']
            if rid in SCHEDULED_RACES: continue
            
            # Parse start time (JST)
            t_str = str(row['hasso_jikoku']).zfill(4)
            h_jst, m_jst = int(t_str[:2]), int(t_str[2:])
            
            # Race time in JST (Correctly constructed from date_cond)
            # date_cond is YYYYMMDD string of the query target
            race_dt_jst = datetime.strptime(f"{date_cond} {h_jst:02}{m_jst:02}", "%Y%m%d %H%M")
            
            # Convert to UTC
            race_dt_utc = race_dt_jst - timedelta(hours=9)
            
            # Trigger: 10 mins before
            trigger_dt_utc = race_dt_utc - timedelta(minutes=10)
            
            if trigger_dt_utc > now_utc:
                trigger_str = trigger_dt_utc.strftime("%H:%M")
                logger.info(f"📅 Scheduled (UTC): {VENUE_MAP.get(row['keibajo_code'])} {row['race_bango']}R at {trigger_str} (JST Start: {h_jst:02}:{m_jst:02})")
                
                def job(r_id=rid, v_name=VENUE_MAP.get(row['keibajo_code']), r_num=row['race_bango']):
                    run_prediction(r_id, v_name, r_num, date_cond)
                    return schedule.CancelJob
                
                schedule.every().day.at(trigger_str).do(job)
                SCHEDULED_RACES.add(rid)
                count += 1
                
            elif race_dt_utc > now_utc:
                # Late trigger allowed window
                delay = (now_utc - trigger_dt_utc).total_seconds()
                if delay < 300: 
                    if rid not in SCHEDULED_RACES:
                        # Fix: Add to set BEFORE running to prevent duplicate triggers if update_schedule runs again
                        SCHEDULED_RACES.add(rid) 
                        logger.info(f"⚠️ Late Trigger (Immediate): {VENUE_MAP.get(row['keibajo_code'])} {row['race_bango']}R")
                        run_prediction(rid, VENUE_MAP.get(row['keibajo_code']), row['race_bango'], date_cond)

        if count > 0:
            logger.info(f"Added {count} new jobs.")
            
    except Exception as e:
        logger.error(f"Update schedule failed: {e}")

def reset_daily_state():
    """Clear processed races state for the new day"""
    logger.info("🧹 Performing daily reset of scheduled races...")
    SCHEDULED_RACES.clear()
    # Note: Pending jobs from 'schedule' library are cleared automatically when they run (return CancelJob)
    # But if there are stale jobs, we might want to clear schedule?
    # schedule.clear() would verify that. But update_schedule will re-add if valid.
    # Safer to just clear Set.

def main():
    logger.info("🚀 JIT Scheduler Started (Continuous Mode)")
    
    # 1. Update schedule immediately
    update_schedule()
    
    # 2. Refresh schedule every 30 mins (to catch late-added races or date changes)
    schedule.every(30).minutes.do(update_schedule)
    
    # 3. Daily Reset at 08:00 JST (23:00 UTC previous day) - Before first race
    # 08:00 JST = 23:00 UTC. 
    # schedule uses system time (UTC). So we schedule at 23:00.
    schedule.every().day.at("23:00").do(reset_daily_state)

    # 4. Daily Cache Update at 08:15 JST (23:15 UTC previous day)
    schedule.every().day.at("23:15").do(run_cache_update)
    
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    main()
