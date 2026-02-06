
import sys
import os
import logging
import pandas as pd
import numpy as np
import datetime
import yaml
import json
import requests
import warnings

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from runtime.strategy_engine import StrategyEngine

# Setup Logging
LOG_DIR = "logs/production"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL") # User should set this
CONFIG_PATH = "config/prod_strategy_v24.yaml"
HISTORY_LOG_PATH = os.path.join(LOG_DIR, "bets_v24_m5.parquet")

def send_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord Webhook URL not set. Skipping notification.")
        print(message)
        return
    
    try:
        data = {"content": message}
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        logger.error(f"Failed to send Discord: {e}")

def load_realtime_data(race_id):
    # Mock for Phase J (Stub)
    # in real deployment, this calls JVD API or scraper
    # For now, we assume this script is called WITH data or fetches it via existing loader
    # TODO: Connect to DataProvider
    pass

def run_cycle():
    logger.info("🚀 Starting Phase J Live Cycle (v24_m5_final)")
    
    # 1. Initialize Engine
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found: {CONFIG_PATH}")
        return
    engine = StrategyEngine(CONFIG_PATH)
    
    # 2. Get Today's Races (Stub)
    # In real world, this loops through getting races
    logger.info("Connecting to Data Stream... (Mock Mode for Deployment Setup)")
    
    # 3. Simulate Logic Check
    # We will verify the config loaded correctly and 'max_odds' is present
    win_cfg = engine.config['strategies']['win_core']
    max_odds = win_cfg.get('max_odds')
    
    status_msg = f"""**[Phase J Start] v24_m5_final**
Strat: Win Only (Odds < {max_odds})
EV Type: Ad-hoc (No Shrinkage)
"""
    send_discord(status_msg)
    logger.info("Deployment script initialized. Ready for cron execution.")

if __name__ == "__main__":
    run_cycle()
