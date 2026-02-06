"""
Gap Prediction Script (V13) - Place Betting
============================================
Uses the improved Gap Model (Top 5 Only) for Place betting recommendations.
- Target: Horses that finish Top 5 AND beat their popularity
- Focus: オッズ 10-50倍 の中穴 (Place ROI 117-172%)

Usage:
  python scripts/predict_v13_gap.py --date 20240101 --discord
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
import requests
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_DIR = "models/experiments/exp_gap_prediction_reg"
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
    parser.add_argument("--discord", action='store_true', help="Send results to Discord")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Webhook")
    parser.add_argument("--race_id", type=str, help="Target Race ID (for JIT execution)")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    
    logger.info(f"🚀 Starting Gap Prediction (V13 Place) for: {target_date}")
    
    # Load Model and Features
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    features_path = os.path.join(MODEL_DIR, "features.csv")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
        
    model = joblib.load(model_path)
    expected_features = pd.read_csv(features_path)['0'].tolist()
    logger.info(f"Loaded model with {len(expected_features)} features.")
    
    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Load Data
    loader = JraVanDataLoader()
    start_year = str(int(target_date[:4]) - 1) + "0101"
    df_raw = loader.load(history_start_date=start_year, end_date=target_date)
    
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
    
    # Fetch live odds
    def fetch_jvd_o1_odds(loader, target_year, target_mmdd):
        import psycopg2
        try:
            # Use direct connection as verified in check_jvd_o1.py
            # This ensures we hit the correct DB with 2026 data
            conn_str = "host='host.docker.internal' port=5433 dbname='pckeiba' user='postgres' password='postgres'"
            conn = psycopg2.connect(conn_str)
            
            q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
            logger.info(f"Odds Query: {q}")
            
            df_odds = pd.read_sql(q, conn)
            conn.close()
            
            logger.info(f"Odds Query Result: {len(df_odds)} rows")
            
            if df_odds.empty: 
                logger.warning(f"No records in jvd_o1 for {target_year}-{target_mmdd}")
                return pd.DataFrame()
            
            rows = []
            for _, row in df_odds.iterrows():
                try:
                    # Fix ID construction: Ensure zero-padding for 2-digit fields
                    race_id = str(row['kaisai_nen']) + \
                              str(row['keibajo_code']).zfill(2) + \
                              str(row['kaisai_kai']).zfill(2) + \
                              str(row['kaisai_nichime']).zfill(2) + \
                              str(row['race_bango']).zfill(2)
                    
                    val = row['odds_tansho']
                    if isinstance(val, bytes):
                        odds_str = val.decode('utf-8')
                    else:
                        odds_str = str(val)
                    
                    # Debug Log for first row
                    if len(rows) == 0:
                        logger.info(f"DEBUG: Sample ID: {race_id}")
                        logger.info(f"DEBUG: Sample Odds Str: {odds_str[:20]}...")
                        logger.info(f"DEBUG: Type of val: {type(val)}")
                    
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
                except Exception as ex:
                    logger.error(f"Row Parse Error: {ex}")
                    continue
            
            logger.info(f"Parsed {len(rows)} individual odds records.")
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
    try:
        pipeline = FeaturePipeline(cache_dir=os.path.join(CACHE_DIR, "gap"))
        
        # Use ALL available blocks to ensure compatibility with processed_data_v12
        feature_blocks = list(pipeline.registry.keys())
        logger.info(f"Using feature blocks: {feature_blocks}")
        
        df_features = pipeline.load_features(df_raw, feature_blocks)
        
        today_ids = df_today['race_id'].unique()
        df_today_features = df_features[df_features['race_id'].isin(today_ids)].copy()
    except Exception as e:
        import traceback
        logger.error(f"Pipeline Error: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Merge odds
    if not df_live_odds.empty and 'odds_10min' not in df_today_features.columns:
        df_today_features = pd.merge(
            df_today_features, 
            df_live_odds[['race_id', 'horse_number', 'odds_live']].rename(columns={'odds_live': 'odds_10min'}), 
            on=['race_id', 'horse_number'], 
            how='left'
        )
    if 'odds_10min' not in df_today_features.columns:
        df_today_features['odds_10min'] = np.nan
    
    # Ad-hoc Feature Engineering (Same as training)
    df_today_features['odds_calc'] = df_today_features['odds_10min'].fillna(10.0)
    df_today_features['odds'] = df_today_features['odds_calc']
    df_today_features['odds_rank'] = df_today_features.groupby('race_id')['odds_calc'].rank(ascending=True)
    
    if 'relative_horse_elo_z' in df_today_features.columns:
        df_today_features['elo_rank'] = df_today_features.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
        df_today_features['odds_rank_vs_elo'] = df_today_features['odds_rank'] - df_today_features['elo_rank']
    else:
        df_today_features['odds_rank_vs_elo'] = 0
        
    df_today_features['is_high_odds'] = (df_today_features['odds_calc'] >= 10).astype(int)
    df_today_features['is_mid_odds'] = ((df_today_features['odds_calc'] >= 5) & (df_today_features['odds_calc'] < 10)).astype(int)
    
    # 3. Prediction
    X = df_today_features.copy()
    for c in expected_features:
        if c not in X.columns:
            X[c] = np.nan
    X = X[expected_features]
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999.0)
    
    X_np = X.values.astype(np.float64)
    
    raw_preds = model.predict(X_np)
    df_today_features['gap_score'] = raw_preds
    df_today_features['gap_rank'] = df_today_features.groupby('race_id')['gap_score'].rank(ascending=False)
    
    # 4. Output
    log_print(f"\n{'='*60}")
    log_print(f" 🎯 GAP MODEL PLACE PREDICTIONS ({target_date}) ")
    log_print(f" Strategy: 中穴の複勝 (10-50倍, Place ROI 117-172%)")
    log_print(f"{'='*60}")
    
    VENUE_MAP = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }

    df_today_features = df_today_features.sort_values(['race_id', 'gap_rank'], ascending=[True, True])
    today_ids = sorted(df_today_features['race_id'].unique())
    
    for rid in today_ids:
        sub = df_today_features[df_today_features['race_id'] == rid]
        
        venue_code = rid[4:6]
        venue_name = VENUE_MAP.get(venue_code, venue_code)
        race_num = int(rid[-2:])
        
        log_print(f"\n[{venue_name} {race_num}R] (ID: {rid})")
        log_print(f"{'No':<3} {'Name':<12} {'Odds':<6} {'GapRank':<8} {'Action'}")
        log_print("-" * 55)
        
        for idx, row in sub.iterrows():
            h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == row['horse_number'])]
            name = str(h_sub.iloc[0].get('horse_name', 'Unknown'))[:10] if not h_sub.empty else "Unknown"
            
            odds_val = row.get('odds_10min', np.nan)
            gap_rank = int(row['gap_rank'])
            
            # Recommendation Logic
            # Gap Top 1-3 + Odds 10-50 -> Place Bet
            action = ""
            if gap_rank <= 3:
                if 10 <= odds_val <= 20:
                    action = "🔥[PLACE BET] ROI117%"
                elif 20 < odds_val <= 50:
                    action = "💰[HIGH VALUE] ROI171%"
                elif odds_val < 10:
                    action = "⚠️[Low Odds]"
                elif odds_val > 50:
                    action = "⚠️[High Risk]"
                else:
                    action = f"[Gap Top {gap_rank}]"
            elif gap_rank <= 5:
                if 10 <= odds_val <= 50:
                    action = "★[Hole Candidate]"
            
            odds_str = f"{odds_val:.1f}" if not pd.isna(odds_val) else "-"
            
            log_print(f"{int(row['horse_number']):02}  {name:<12} {odds_str:<6} {gap_rank:<8} {action}")
            
    log_print(f"\n{'='*60}")
    log_print("Strategy: Gap Model Top 1-3 + Odds 10-50倍 で複勝!")
    log_print(f"{'='*60}")
    
    if args.discord:
        send_discord_notification(args.webhook_url, "".join(output_log))

if __name__ == "__main__":
    main()
