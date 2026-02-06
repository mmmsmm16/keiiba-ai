
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

# Load .env file explicitly
load_dotenv()

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Note: odds_fluctuation temporarily disabled (apd_sokuho_o1 has no 2026 data)
FEATURE_BLOCKS = [
    'base_attributes', 'history_stats', 'jockey_stats', 
    'pace_stats', 'bloodline_stats', 'burden_stats', 
    'changes_stats', 'aptitude_stats', 'speed_index_stats', 
    'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats', 
    'temporal_jockey_stats', 'temporal_trainer_stats', 'class_stats', 
    'segment_stats', 'risk_stats', 'course_aptitude', 
    'extended_aptitude', 'runstyle_fit', 'jockey_trainer_compatibility', 
    'interval_aptitude', 'physique_training', 'jockey_strategy', 
    'race_dynamics', 'sire_aptitude', 'track_bias', 'frame_bias', 
    'weight_pattern', 'rest_pattern', 'corner_dynamics', 
    'head_to_head', 'training_detail', 'bloodline_detail', 
    'strategy_pattern', 'horse_gear'
]

BASE_MODEL_PATH = "models/experiments/exp_t2_refined/model.pkl"
META_MODEL_PATH = "models/experiments/exp_t2_meta/meta_lgbm.txt"
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
    parser.add_argument("--discord", action='store_true', help="Send results to Discord")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Webhook")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    logger.info(f"🚀 Starting T2 Production Run for: {target_date}")
    
    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Load Data
    loader = JraVanDataLoader()
    # History data for context (last 30 days)
    # Plus today's data from DB
    df_raw = loader.load(history_start_date='2024-01-01', end_date=target_date)
    
    # Filter only today for prediction
    today_dt = pd.to_datetime(target_date)
    
    # Pre-calculate time_diff for history stats (needed by FeaturePipeline)
    # And Ensure numeric types for critical calculation columns
    numeric_cols = ['time', 'last_3f', 'rank', 'weight', 'weight_diff', 'impost', 'honshokin', 'fukashokin']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    if 'time' in df_raw.columns:
        min_times = df_raw.groupby('race_id')['time'].transform('min')
        df_raw['time_diff'] = (df_raw['time'] - min_times).fillna(0)
    else:
        df_raw['time_diff'] = 0
    
    # === Fetch live odds from jvd_o1 for today's races ===
    def fetch_jvd_o1_odds(loader, target_year, target_mmdd):
        """Fetch and parse odds from jvd_o1 (concatenated 4-digit per horse)."""
        try:
            q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
            df_odds = pd.read_sql(q, loader.engine)
            
            if df_odds.empty:
                logger.warning("No jvd_o1 data found for today.")
                return pd.DataFrame()
            
            rows = []
            for _, row in df_odds.iterrows():
                race_id = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
                odds_str = str(row['odds_tansho'])
                
                # Each horse's odds is 4 digits (e.g., 1188 = 118.8x)
                # Each horse's odds is 4 digits (e.g., 0150 = 1.5x, stored as odds * 100)
                for i in range(0, len(odds_str), 4):
                    horse_num = i // 4 + 1
                    odds_raw = odds_str[i:i+4]
                    if odds_raw.isdigit():
                        odds_val = int(odds_raw) / 100.0  # Divide by 100, not 10
                        rows.append({'race_id': race_id, 'horse_number': horse_num, 'odds_live': odds_val})
            
            return pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"Failed to fetch jvd_o1: {e}")
            return pd.DataFrame()
    
    # Parse target date
    target_year = target_date[:4]
    target_mmdd = target_date[4:]
    
    logger.info(f"Fetching jvd_o1 odds for year={target_year}, mmdd={target_mmdd}")
    df_live_odds = fetch_jvd_o1_odds(loader, target_year, target_mmdd)
    logger.info(f"df_live_odds size: {len(df_live_odds)}")
    
    # Debug: save to file
    with open('/workspace/debug_odds.txt', 'w') as f:
        f.write(f"target_year={target_year}, target_mmdd={target_mmdd}\n")
        f.write(f"df_live_odds size: {len(df_live_odds)}\n")
        if not df_live_odds.empty:
            f.write(f"Sample:\n{df_live_odds.head(10).to_string()}\n")
    
    if not df_live_odds.empty:
        logger.info(f"Fetched {len(df_live_odds)} live odds records from jvd_o1")
        # Ensure type consistency for merge
        df_raw['horse_number'] = df_raw['horse_number'].astype(int)
        df_live_odds['horse_number'] = df_live_odds['horse_number'].astype(int)
        df_raw = pd.merge(df_raw, df_live_odds, on=['race_id', 'horse_number'], how='left')
        df_raw['odds_10min'] = df_raw['odds_live'].combine_first(
            pd.to_numeric(df_raw.get('odds', pd.Series()), errors='coerce') / 10.0
        )
        logger.info(f"After merge, odds_10min non-null: {df_raw['odds_10min'].notnull().sum()}")
    elif 'odds' in df_raw.columns:
        df_raw['odds_10min'] = pd.to_numeric(df_raw['odds'], errors='coerce') / 10.0
    else:
        df_raw['odds_10min'] = np.nan
    
    df_today = df_raw[df_raw['date'] == today_dt].copy()
    
    if df_today.empty:
        logger.error(f"No races found for {target_date} in DB.")
        return

    # 2. Feature Generation
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    df_features = pipeline.load_features(df_raw, FEATURE_BLOCKS)
    
    # Filter features for today
    today_ids = df_today['race_id'].unique()
    df_today_features = df_features[df_features['race_id'].isin(today_ids)].copy()
    
    # Merge odds from df_live_odds directly to df_today_features (since FeaturePipeline may not include odds_10min)
    if not df_live_odds.empty:
        df_today_features = pd.merge(
            df_today_features, 
            df_live_odds[['race_id', 'horse_number', 'odds_live']].rename(columns={'odds_live': 'odds_10min'}), 
            on=['race_id', 'horse_number'], 
            how='left'
        )
        logger.info(f"Merged odds_10min to df_today_features: non-null={df_today_features['odds_10min'].notnull().sum()}")
    else:
        df_today_features['odds_10min'] = np.nan
    
    # 3. Base Model Prediction
    logger.info("Predicting with Base Model (T2)...")
    base_model = joblib.load(BASE_MODEL_PATH)
    
    if hasattr(base_model, 'booster_'):
        booster = base_model.booster_
    else:
        booster = base_model
        
    base_feats = booster.feature_name()
    
    # Identify potential categorical features from dtype
    # and model expectation
    # If the error persists, it's often because LightGBM stores category names.
    
    # Ensure all features exist
    for f in base_feats:
        if f not in df_today_features.columns:
            df_today_features[f] = 0
            
    # Try converting to float and then predict using numpy to bypass pandas categorical check
    # Many categorical features in this pipeline are strings which need label encoding.
    # FeaturePipeline should have handled encoding, but let's check.
    X_base = df_today_features[base_feats].copy()
    for col in X_base.columns:
        if X_base[col].dtype == 'object':
            X_base[col] = pd.to_numeric(X_base[col], errors='coerce').fillna(0)
            
    # Predicting with numpy array bypasses several pandas-specific checks in LightGBM
    try:
        df_today_features['pred_prob'] = booster.predict(X_base.values.astype(float))
    except Exception as e:
        logger.warning(f"Numpy prediction failed, trying with pandas: {e}")
        df_today_features['pred_prob'] = booster.predict(X_base)
    
    # 4. Meta Model Prediction
    logger.info("Predicting with Meta Model...")
    meta_model = lgb.Booster(model_file=META_MODEL_PATH)
    
    # Meta Features: pred_prob, odds_ratio_60_10, prob_market, prob_diff, bias_adversity_score_mean_5
    # Calculate additional Meta features
    df_today_features['prob_market'] = 0.8 / (df_today_features['odds_10min'].fillna(100) + 1e-9)
    df_today_features['prob_diff'] = df_today_features['pred_prob'] - df_today_features['prob_market']
    
    # Ensure meta features exist (with default values if missing)
    meta_feats = ['pred_prob', 'odds_ratio_60_10', 'prob_market', 'prob_diff', 'bias_adversity_score_mean_5']
    for mf in meta_feats:
        if mf not in df_today_features.columns:
            df_today_features[mf] = 1.0 if 'ratio' in mf else 0.0  # ratio=1 is neutral, others=0
            logger.warning(f"Meta feature '{mf}' missing, set to default.")
    
    # Ensure no NaNs in meta features
    X_meta = df_today_features[meta_feats].fillna(0)
    df_today_features['meta_prob'] = meta_model.predict(X_meta)
    
    # Calculate EV
    df_today_features['ev'] = df_today_features['meta_prob'] * df_today_features['odds_10min'].fillna(0)
    
    # 5. Output Recommendations
    log_print(f"\n{'='*60}")
    log_print(f" 🏇 KEIIBA-AI T2 PRODUCTION PREDICTIONS ({target_date}) ")
    log_print(f"{'='*60}")
    
    # Sort by race_id and meta_prob
    df_today_features = df_today_features.sort_values(['race_id', 'meta_prob'], ascending=[True, False])
    
    for rid in today_ids:
        sub = df_today_features[df_today_features['race_id'] == rid]
        race_info = df_today[df_today['race_id'] == rid].iloc[0]
        
        venue_name = race_info.get('venue', 'Unknown')
        race_num = race_info.get('race_number', 0)
        
        log_print(f"\n[{venue_name} {race_num}R] (ID: {rid})")
        log_print(f"{'No':<3} {'Horse':<16} {'MetaProb':<8} {'EV':<6} {'Odds':<6}")
        log_print("-" * 50)
        
        for _, row in sub.iterrows():
            # Get horse name from df_today
            h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == row['horse_number'])]
            if h_sub.empty:
                 name = "Unknown"
            else:
                 name = str(h_sub.iloc[0].get('horse_name', 'Unknown'))[:12]
            
            ev_str = f"{row['ev']:.2f}" if not pd.isna(row['ev']) else "---"
            odds_str = f"{row['odds_10min']:.1f}" if not pd.isna(row['odds_10min']) else "---"
            log_print(f"{int(row['horse_number']):02} {name:<16} {row['meta_prob']:.2%} {ev_str:<6} {odds_str:<6}")
            
        # Strategy Alerts
        # 1. High-Confidence EV
        targets = sub[sub['ev'] >= 1.05]
        if not targets.empty:
            log_print("\n💡 推奨馬 (EV > 1.05):")
            for _, t in targets.iterrows():
                h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == t['horse_number'])]
                h_name = h_sub.iloc[0].get('horse_name', 'Unknown') if not h_sub.empty else "Unknown"
                log_print(f"  - {int(t['horse_number']):02} {h_name} (EV: {t['ev']:.2f})")
                
        # 2. Trio Nagashi (if top EV horse is good)
        top1 = sub.iloc[0]
        if top1['ev'] >= 1.1:
            followers = sub.iloc[1:8]['horse_number'].astype(int).tolist()
            log_print(f"\n🎯 三連複 1頭軸流し:")
            log_print(f"  軸: {int(top1['horse_number']):02} ({top1['meta_prob']:.1%} )")
            log_print(f"  相手: {', '.join([f'{f:02}' for f in followers])}")

    log_print(f"\n{'='*60}")
    log_print("🚀 Production Run Completed.")
    
    if args.discord:
        send_discord_notification(args.webhook_url, "".join(output_log))

if __name__ == "__main__":
    main()
