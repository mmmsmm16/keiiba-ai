
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from datetime import datetime
import requests
import json
import time
from dotenv import load_dotenv
from scipy.special import softmax

# Load .env file explicitly
load_dotenv()

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CACHE_DIR = "data/features_v13"

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
    parser.add_argument("--exp", type=str, default="exp_lambdarank_hard_weighted", help="Experiment Name")
    parser.add_argument("--discord", action='store_true', help="Send results to Discord")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Webhook")
    parser.add_argument("--race_id", type=str, help="Target Race ID (for JIT execution)")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    exp_name = args.exp
    
    logger.info(f"🚀 Starting Deep Value Prediction (V13) for: {target_date} (Exp: {exp_name})")
    
    # Paths
    base_dir = f"models/experiments/{exp_name}"
    model_path_pkl = os.path.join(base_dir, "model.pkl")
    config_path = os.path.join(base_dir, "config.yaml")

    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return
        
    # Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    feature_blocks = config.get('features', [])
    model_params = config.get('model_params', {})
    prediction_type = model_params.get('prediction_type', 'sigmoid') # Default to sigmoid if not specified
    
    logger.info(f"Loaded config. Prediction Type: {prediction_type}")

    # Load Model
    if not os.path.exists(model_path_pkl):
        logger.error("Model file not found.")
        return
    
    model = joblib.load(model_path_pkl)
    logger.info(f"Loaded model: {model_path_pkl}")

    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Load Data
    loader = JraVanDataLoader()
    start_year = str(int(target_date[:4]) - 1) + "0101"
    df_raw = loader.load(history_start_date=start_year, end_date=target_date)
    
    # Filter only today for prediction
    today_dt = pd.to_datetime(target_date)
    
    # Numeric Conversion
    numeric_cols = ['time', 'last_3f', 'rank', 'weight', 'weight_diff', 'impost', 'honshokin', 'fukashokin']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    if 'time' in df_raw.columns:
        min_times = df_raw.groupby('race_id')['time'].transform('min')
        df_raw['time_diff'] = (df_raw['time'] - min_times).fillna(0)
    else:
        df_raw['time_diff'] = 0
    
    # === Fetch live odds from jvd_o1 ===
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
                    except: continue
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return pd.DataFrame()
    
    target_year = target_date[:4]
    target_mmdd = target_date[4:]
    df_live_odds = fetch_jvd_o1_odds(loader, target_year, target_mmdd)
    
    if not df_live_odds.empty:
        logger.info(f"Fetched {len(df_live_odds)} live odds records.")
        df_raw['horse_number'] = df_raw['horse_number'].astype(int)
        df_live_odds['horse_number'] = df_live_odds['horse_number'].astype(int)
        df_raw = pd.merge(df_raw, df_live_odds, on=['race_id', 'horse_number'], how='left')
        df_raw['odds_10min'] = df_raw['odds_live'].combine_first(pd.to_numeric(df_raw.get('odds', pd.Series()), errors='coerce') / 10.0)
    elif 'odds' in df_raw.columns:
        df_raw['odds_10min'] = pd.to_numeric(df_raw['odds'], errors='coerce') / 10.0
    else:
        df_raw['odds_10min'] = np.nan
    
    df_today = df_raw[df_raw['date'] == today_dt].copy()
    
    if args.race_id:
        logger.info(f"Filtering for Race ID: {args.race_id}")
        df_today = df_today[df_today['race_id'] == str(args.race_id)]
    
    if df_today.empty:
        logger.error(f"No races found for {target_date}.")
        return

    # 2. Feature Generation
    exp_cache_dir = os.path.join(CACHE_DIR, exp_name)
    pipeline = FeaturePipeline(cache_dir=exp_cache_dir)
    df_features = pipeline.load_features(df_raw, feature_blocks)
    
    today_ids = df_today['race_id'].unique()
    df_today_features = df_features[df_features['race_id'].isin(today_ids)].copy()
    
    # Merge odds back if missing
    if not df_live_odds.empty and 'odds_10min' not in df_today_features.columns:
        df_today_features = pd.merge(
            df_today_features, 
            df_live_odds[['race_id', 'horse_number', 'odds_live']].rename(columns={'odds_live': 'odds_10min'}), 
            on=['race_id', 'horse_number'], 
            how='left'
        )
    # Ensure odds_10min is filled (for calc)
    if 'odds_10min' not in df_today_features.columns:
        df_today_features['odds_10min'] = np.nan
        
    # Merge yoso_juni from df_raw if available (needed for undervalued features)
    if 'yoso_juni' not in df_today_features.columns and 'yoso_juni' in df_raw.columns:
        # Filter df_raw for today first to speed up
        today_raw = df_raw[df_raw['race_id'].isin(today_ids)][['race_id', 'horse_number', 'yoso_juni']]
        df_today_features = pd.merge(df_today_features, today_raw, on=['race_id', 'horse_number'], how='left')
    
    # --- Ad-hoc Feature Engineering (Undervalued Features) ---
    logger.info("Injecting Undervalued Features...")
    
    # 1. Fill Odds
    # Use 10min odds for calculation (Simulating pre-race state)
    # If NaN, fill with something neutral (e.g. median or 50.0)
    df_today_features['odds_calc'] = df_today_features['odds_10min'].fillna(10.0)
    
    # 2. Yoso Juni check
    if 'yoso_juni' in df_today_features.columns:
        df_today_features['yoso_juni_num'] = pd.to_numeric(df_today_features['yoso_juni'], errors='coerce').fillna(8)
    else:
        logger.warning("yoso_juni not found! Filling with default.")
        df_today_features['yoso_juni_num'] = 8
        
    # 3. Popularity check (Mining Feature Generator should provide it if available, else load from somewhere?)
    # Usually popularity is in 'pass_1' or similar in jvd_o1? No, popularity is rank of odds.
    # We can calculate popularity from odds_10min!
    df_today_features['popularity_calc'] = df_today_features.groupby('race_id')['odds_calc'].rank(ascending=True)
    df_today_features['popularity_num'] = df_today_features['popularity_calc']
    
    # 4. Features
    df_today_features['popularity_vs_yoso'] = df_today_features['popularity_num'] - df_today_features['yoso_juni_num']
    df_today_features['odds_rank'] = df_today_features['popularity_calc'] # Same as odds rank
    df_today_features['odds_rank_vs_yoso'] = df_today_features['odds_rank'] - df_today_features['yoso_juni_num']
    
    if 'relative_horse_elo_z' in df_today_features.columns:
        df_today_features['elo_rank'] = df_today_features.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
        df_today_features['odds_rank_vs_elo'] = df_today_features['odds_rank'] - df_today_features['elo_rank']
    else:
        df_today_features['odds_rank_vs_elo'] = 0
        
    df_today_features['is_high_odds'] = (df_today_features['odds_calc'] >= 10).astype(int)
    df_today_features['is_mid_odds'] = ((df_today_features['odds_calc'] >= 5) & (df_today_features['odds_calc'] < 10)).astype(int)
    
    # Mapping for alignment
    # Model uses 'odds', so map 'odds_calc' to 'odds' if model requests it.
    # Actually model features list will contain 'is_high_odds' etc.
    # 'odds' raw column might be in model? If so, map it.
    df_today_features['odds'] = df_today_features['odds_calc']
    
    # 3. Model Prediction
    # ------------------
    # Align Columns
    if hasattr(model, 'booster_'):
        expected_features = model.booster_.feature_name()
    else:
        expected_features = model.feature_name()
    X = df_today_features.copy()
    
    # Fill missing
    for c in expected_features:
        if c not in X.columns:
            X[c] = np.nan
            
    # Select cols
    X = X[expected_features]
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999.0)
        
    X_np = X.values.astype(np.float64)
    
    logger.info(f"Predicting w/ type: {prediction_type}")
    
    raw_preds = model.predict(X_np)
    
    if prediction_type == 'softmax':
        df_today_features['pred_score'] = raw_preds
        # Apply Softmax per race
        def softmax_func(x):
            try:
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0) 
            except:
                return x
        df_today_features['pred_prob'] = df_today_features.groupby('race_id')['pred_score'].transform(softmax_func)
    else:
        # Sigmoid
        df_today_features['pred_prob'] = 1 / (1 + np.exp(-raw_preds))
        # Norm
        df_today_features['pred_prob'] = df_today_features.groupby('race_id')['pred_prob'].transform(lambda x: x / x.sum())

    # EV Calc
    df_today_features['ev'] = df_today_features['pred_prob'] * df_today_features['odds_10min'].fillna(0)
    
    # 4. Output
    log_print(f"\n{'='*60}")
    log_print(f" 🏇 DEEP VALUE JAL (V13) PREDICTIONS ({target_date}) ")
    log_print(f" Model: {exp_name}")
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
        
        venue_code = rid[4:6]
        venue_name = VENUE_MAP.get(venue_code, venue_code)
        race_num = int(rid[-2:])
        
        log_print(f"\n[{venue_name} {race_num}R] (ID: {rid})")
        log_print(f"{'No':<3} {'Name':<12} {'Prob':<6} {'EV':<5} {'Odds':<5} {'Action'}")
        log_print("-" * 55)
        
        # Determine Top 1
        top_horse_idx = sub['pred_prob'].idxmax()
        
        for idx, row in sub.iterrows():
            h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == row['horse_number'])]
            name = str(h_sub.iloc[0].get('horse_name', 'Unknown'))[:10] if not h_sub.empty else "Unknown"
            
            ev_val = row.get('ev', 0)
            odds_val = row.get('odds_10min', np.nan)
            prob_val = row.get('pred_prob', 0)
            
            is_top1 = (idx == top_horse_idx)
            
            # Recommendation Logic (Deep Value Strategy)
            # Rank 1 & Odds 1.0-2.0 -> Place Axis (ROI 91.3%)
            # Rank 1 & Odds 2.0-3.0 -> High Value Axis (ROI 86.9%)
            
            action = ""
            if is_top1:
                if 1.0 <= odds_val <= 2.0:
                    action = "🔥[AXIS/PLACE] ROI91%"
                elif 2.0 < odds_val <= 3.0:
                    action = "💰[VALUE AXIS] ROI86%"
                elif odds_val > 3.0:
                    action = "⚠️[Risk Axis]"
                else:
                    action = "[Top1]"
            
            # Optional: Wide Partner Logic? (No, ROI<100, so keep empty or simple mark)
            # Mark High EV horses (>1.5) as Hole candidates
            elif ev_val >= 1.5:
                action = "★[Hole]"
                
            ev_str = f"{ev_val:.2f}"
            odds_str = f"{odds_val:.1f}" if not pd.isna(odds_val) else "-"
            
            log_print(f"{int(row['horse_number']):02}  {name:<12} {prob_val:.1%}  {ev_str:<5} {odds_str:<5} {action}")
            
    log_print(f"\n{'='*60}")
    
    if args.discord:
        send_discord_notification(args.webhook_url, "".join(output_log))

if __name__ == "__main__":
    main()
