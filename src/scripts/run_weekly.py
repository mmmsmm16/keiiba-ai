
import os
import sys
import yaml
import datetime
import subprocess
import logging
import pandas as pd
import numpy as np
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reporting.html_generator import HTMLReportGenerator
from model.betting_strategy import BettingStrategy, BettingOptimizer, TicketType
from model.evaluate import load_payout_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(cmd):
    logger.info(f"Executing: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logger.error(f"Command failed: {cmd}")
        # Phase 16 requirement: Log error but maybe continue? Or stop?
        # Critical failure if model training fails.
        if "train.py" in cmd or "regenerate" in cmd:
             raise RuntimeError("Critical pipeline step failed.")

def get_target_dates():
    # Find upcoming Saturday and Sunday
    today = datetime.date.today()
    dataset = []
    # Look ahead 7 days
    for i in range(7):
        d = today + datetime.timedelta(days=i)
        # 5=Sat, 6=Sun
        if d.weekday() in [5, 6]:
            dataset.append(d.strftime('%Y-%m-%d'))
    return sorted(list(set(dataset)))

def main():
    config = load_config()
    bet_conf = config.get('betting', {})
    pipe_conf = config.get('pipeline', {})
    
    current_year = datetime.date.today().year
    valid_year = pipe_conf.get('valid_year', current_year)
    
    logger.info("=== STARTING WEEKLY PIPELINE ===")
    
    # 1. Update Data (Optional if assumed manual db sync)
    # user usually runs this manually? But automating preprocessing is key.
    if pipe_conf.get('update_data', True):
        cmd = "python src/preprocessing/run_preprocessing.py"
        run_command(cmd)
        
    # 2. Retrain Model
    if pipe_conf.get('retrain_model', True):
        # Regenerate Datasets
        cmd_regen = f"python src/scripts/regenerate_datasets.py --valid_year {valid_year}"
        run_command(cmd_regen)
        
        # Train
        model_type = bet_conf.get('model_type', 'lgbm')
        model_version = "weekly_update" # Use a fixed name for weekly model
        cmd_train = f"python src/model/train.py --model {model_type} --version {model_version}"
        run_command(cmd_train)
    else:
        model_type = bet_conf.get('model_type', 'lgbm')
        model_version = bet_conf.get('model_version', 'v4_emb')

    # 3. Inference & Betting
    target_dates = get_target_dates()
    logger.info(f"Target Dates: {target_dates}")
    
    all_bets = []
    
    for date_str in target_dates:
        # Run Inference
        cmd_pred = f"python src/inference/predict.py --date {date_str} --model {model_type} --version {model_version}"
        run_command(cmd_pred)
        
        # Load Prediction
        pred_path = f"experiments/predictions/{date_str.replace('-','')}_{model_type}.csv"
        if not os.path.exists(pred_path):
            logger.warning(f"Prediction file not found: {pred_path}")
            continue
            
        df_pred = pd.read_csv(pred_path)
        if df_pred.empty: continue
        
        
        # Betting Logic using Shared Strategy
        from inference.strategy import WeeklyBettingStrategy
        strategy = WeeklyBettingStrategy(config)
        
        # Apply Strategy
        df_bets = strategy.apply(df_pred)
        
        if not df_bets.empty:
            all_bets.extend(df_bets.to_dict('records'))
            
    # 4. Generate Report
    if all_bets:
        genes = HTMLReportGenerator()
        df_all = pd.DataFrame(all_bets)
        genes.generate(df_all, bankroll=bet_conf.get('initial_bankroll'))
        logger.info("Report generated successfully.")
    else:
        logger.info("No bets generated.")


        
if __name__ == "__main__":
    main()
