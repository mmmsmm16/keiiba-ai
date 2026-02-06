
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from datetime import datetime, timedelta
import requests
import json
import time
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CACHE_DIR = "data/features_t2"

# --- Discord Notification Helper ---
def send_discord_notification(webhook_url, content):
    if not webhook_url:
        return
    try:
        MAX_LEN = 1900
        lines = content.split('\n')
        chunks = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 > MAX_LEN:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk: chunks.append(current_chunk)
        for chunk in chunks:
            requests.post(webhook_url, json={"content": f"```\n{chunk}\n```"})
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"Discord error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date (YYYYMMDD)")
    parser.add_argument("--exp", type=str, default="exp_lambdarank", help="Experiment Name")
    parser.add_argument("--discord", action='store_true', help="Send results to Discord")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Webhook")
    parser.add_argument("--race_id", type=str, help="Target Race ID (for JIT execution)")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    exp_name = args.exp
    
    logger.info(f"🚀 Starting Production Run (LambdaRank) for: {target_date} (Exp: {exp_name})")
    
    # Paths
    base_dir = f"models/experiments/{exp_name}"
    model_path_pkl = os.path.join(base_dir, "model.pkl")
    config_path = os.path.join(base_dir, "config.yaml")
    calib_path = os.path.join(base_dir, "calibrator_win.pkl") # Use Win Calibrator
    
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return
        
    # Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    feature_blocks = config.get('features', [])
    
    logger.info(f"Loaded config. Feature Blocks: {len(feature_blocks)}")

    # Load Model
    model = joblib.load(model_path_pkl)
    logger.info(f"Loaded model: {model_path_pkl}")

    # Load Calibrator
    calibrator = None
    if os.path.exists(calib_path):
        calibrator = joblib.load(calib_path)
        logger.info(f"Loaded Calibrator: {calib_path}")
    else:
        logger.warning(f"Calibrator NOT found at {calib_path}. Using raw scores (Miscalibrated Risk!).")

    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Load Data (Fast Mode: Read existing parquet)
    # The JIT scheduler ensures 'update_daily_features.py' runs daily.
    # So we assume 'preprocessed_data_v11.parquet' contains today's features.
    DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        return
        
    logger.info(f"Loading features from {DATA_PATH}...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all['race_id'] = df_all['race_id'].astype(str)
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Filter only today
    today_dt = pd.to_datetime(target_date)
    
    if args.race_id:
        logger.info(f"Filtering for Race ID: {args.race_id}")
        df_today_features = df_all[df_all['race_id'] == str(args.race_id)].copy()
    else:
        df_today_features = df_all[df_all['date'] == today_dt].copy()
        
    if df_today_features.empty:
        logger.error(f"No features found for {target_date} (or race_id) in {DATA_PATH}.")
        logger.info("Please run 'python scripts/update_daily_features.py' first.")
        return
        
    logger.info(f"Found {len(df_today_features)} records.")
    
    # === Fetch live odds ===
    def fetch_jvd_o1_odds(loader, target_year, target_mmdd):
        try:
            q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
            df_odds = pd.read_sql(q, loader.engine)
            if df_odds.empty: return pd.DataFrame()
            
            rows = []
            for _, row in df_odds.iterrows():
                race_id = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
                odds_str = str(row['odds_tansho'])
                for i in range(0, len(odds_str), 8):
                    chunk = odds_str[i:i+8]
                    if len(chunk) < 6: continue
                    try:
                        horse_num_str = chunk[0:2]
                        odds_part = chunk[2:6]
                        if horse_num_str.isdigit():
                            horse_num = int(horse_num_str)
                            if odds_part.isdigit():
                                odds_val = int(odds_part) / 10.0
                                rows.append({'race_id': race_id, 'horse_number': horse_num, 'odds_live': odds_val})
                            else:
                                rows.append({'race_id': race_id, 'horse_number': horse_num, 'odds_live': np.nan})
                    except:
                        continue
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return pd.DataFrame()
    
    # Parse target date
    target_year = target_date[:4]
    target_mmdd = target_date[4:]
    
    loader = JraVanDataLoader()
    df_live_odds = fetch_jvd_o1_odds(loader, target_year, target_mmdd)
    
    if not df_live_odds.empty:
        logger.info(f"Fetched {len(df_live_odds)} live odds records.")
        df_today_features['horse_number'] = df_today_features['horse_number'].astype(int)
        df_live_odds['horse_number'] = df_live_odds['horse_number'].astype(int)
        
        # Merge odds into features
        if 'odds_10min' in df_today_features.columns:
             df_today_features.drop(columns=['odds_10min'], inplace=True)
             
        df_today_features = pd.merge(
            df_today_features, 
            df_live_odds[['race_id', 'horse_number', 'odds_live']].rename(columns={'odds_live': 'odds_10min'}), 
            on=['race_id', 'horse_number'], 
            how='left'
        )
    elif 'odds' in df_today_features.columns:
        df_today_features['odds_10min'] = pd.to_numeric(df_today_features['odds'], errors='coerce') / 10.0
    else:
        df_today_features['odds_10min'] = np.nan
        
    # Skip raw feature loading/processing as we loaded processed features directly
    

    # 2. Metadata & Final Setup
    df_today = df_today_features.copy()
    
    # Fetch Horse Names (jvd_se) for display
    def fetch_horse_names(loader, target_date):
        try:
            # target_date is YYYYMMDD
            # jvd_se keys: kaisai_nen, kaisai_tsukihi, ...
            # Actually easier to fetch by race_id list if we have many?
            # Or just fetch for the day.
            year = target_date[:4]
            mmdd = target_date[4:]
            
            q = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, umaban, bamei FROM jvd_se WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{mmdd}'"
            df_names = pd.read_sql(q, loader.engine)
            
            if df_names.empty: return pd.DataFrame()
            
            rows = []
            for _, row in df_names.iterrows():
                rid = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
                rows.append({'race_id': rid, 'horse_number': int(row['umaban']), 'horse_name': row['bamei']})
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error fetching names: {e}")
            return pd.DataFrame()

    logger.info("Fetching horse names from DB...")
    df_names = fetch_horse_names(loader, target_date)
    
    if not df_names.empty:
        df_today = pd.merge(df_today, df_names, on=['race_id', 'horse_number'], how='left')
    else:
        df_today['horse_name'] = 'Unknown'

    # Ensure race_info works for venue
    # We need 'keibajo_code' or 'venue' column in df_today for the output loop
    if 'keibajo_code' not in df_today.columns:
        df_today['keibajo_code'] = df_today['race_id'].str.slice(4, 6)
    
    # 3. Model Prediction
    logger.info("Predicting...")
    
    # Feature alignment
    try:
        if hasattr(model, "feature_name"):
             model_features = model.feature_name()
        elif hasattr(model, "booster_"):
             model_features = model.booster_.feature_name()
        else:
             model_features = list(model.feature_name_)
    except:
        logger.warning("Could not get feature names from model. Using DataFrame columns.")
        model_features = df_today_features.columns.tolist()

    X = df_today_features.copy()
    
    # Drop irrelevant
    for c in ['race_id', 'date', 'rank', 'target', 'year', 'time_diff', 'odds', 'horse_id', 'odds_10min', 'odds_live']:
        if c in X.columns: X.drop(columns=[c], inplace=True)
        
    # Align
    for c in model_features:
        if c not in X.columns: X[c] = np.nan
    X = X[model_features]
    
    # Categorical handling (Codes)
    for c in X.columns:
        if X[c].dtype.name == 'category' or X[c].dtype == 'object':
            X[c] = X[c].astype('category').cat.codes
        else:
            X[c] = X[c].fillna(-999.0)
            
    # Numpy Bypass
    X_val = X.values.astype(np.float32)
    
    try:
        raw_preds = model.predict(X_val)
        
        # Apply Calibration
        if calibrator:
             preds = calibrator.transform(raw_preds)
             logger.info("Applied Isotonic Calibration.")
        else:
             preds = raw_preds
             logger.warning("Using RAW SCORES.")
             
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return

    df_today_features['pred_prob'] = preds
    
    # NOTE: Do NOT normalize LambdaRank scores (Probabilities) to sum to 1.
    # While Win Probs should sum to 1, calibration happens horse-by-horse.
    # Normalizing invalidates the absolute probability value (ECE).
    # We trust the calibrated absolute probability.
    
    # EV Calculation
    df_today_features['ev'] = df_today_features['pred_prob'] * df_today_features['odds_10min'].fillna(0)
    
    # 4. Output
    log_print(f"\n{'='*60}")
    log_print(f" 🏇 KEIIBA-AI PRODUCTION (LambdaRank) PREDICTIONS ({target_date}) ")
    log_print(f"{'='*60}")
    
    VENUE_MAP = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }

    df_today_features = df_today_features.sort_values(['race_id', 'pred_prob'], ascending=[True, False])
    today_ids = sorted(df_today_features['race_id'].unique())
    
    for rid in today_ids:
        sub = df_today_features[df_today_features['race_id'] == rid]
        race_info = df_today[df_today['race_id'] == rid].iloc[0]
        
        venue_code = race_info.get('keibajo_code')
        if not venue_code and 'venue' in race_info: venue_code = str(race_info['venue']).zfill(2)
        if not venue_code: venue_code = rid[4:6]
        venue_name = VENUE_MAP.get(venue_code, venue_code)
        
        race_num = int(race_info.get('race_bango', rid[-2:]))
        
        log_print(f"\n[{venue_name} {race_num}R] (ID: {rid})")
        log_print(f"{'馬番':<4} {'馬名':<16} {'勝率%':<8} {'期待値':<6} {'オッズ':<6}")
        log_print("-" * 50)
        
        for _, row in sub.iterrows():
            h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == row['horse_number'])]
            name = str(h_sub.iloc[0].get('horse_name', 'Unknown'))[:12] if not h_sub.empty else "Unknown"
            
            ev_str = f"{row['ev']:.2f}" if not pd.isna(row['ev']) else "---"
            odds_str = f"{row.get('odds_10min', np.nan):.1f}" if not pd.isna(row.get('odds_10min', np.nan)) else "---"
            
            # Betting Logic
            rec_mark = ""
            stats_prob = row['pred_prob']
            stats_ev = row.get('ev', 0)
            
            is_top1 = (sub['pred_prob'].idxmax() == row.name)
            
            # Strategy: LambdaRank Calibrated Win
            # EV > 1.6 (Aggressive) or 1.8 (Conservative)
            if is_top1 and stats_ev >= 1.6:
                rec_mark = "🎯[推奨] Win EV={:.2f}".format(stats_ev)
            elif is_top1 and stats_ev >= 1.2:
                rec_mark = "📊[監視] EV={:.2f}".format(stats_ev)
                
            log_print(f"{int(row['horse_number']):02}   {name:<16} {stats_prob:>6.1%}   {ev_str:<6} {odds_str:<6} {rec_mark}")
            
    log_print(f"\n{'='*60}")
    log_print("🚀 Production Run Completed.")
    
    if args.discord:
        send_discord_notification(args.webhook_url, "".join(output_log))

if __name__ == "__main__":
    main()
