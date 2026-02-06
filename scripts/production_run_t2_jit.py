"""
Production Run T2 - Just-In-Time (Hybrid) Version
=================================================
Hybrid Approach:
1. Horse Features: Computed Just-In-Time from full history of today's horses.
   - Ensures 100% accurate Lag, Rolling, Interval features.
   - Avoids "mean of means" approximation issues.
2. Human/Sire Features: Loaded from pre-computed Cache.
   - Ensures global statistics are accurate (not limited to loaded horse history).
   - Fast and existing mechanism.

Execution Time estimation: ~10-20 seconds.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import shutil
import time
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.utils.discord import NotificationManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# 2026-01-05: Switched to No-IDs model for better generalization
# Previous: models/experiments/exp_t2_track_bias/model.pkl (with horse_id, mare_id, etc.)
BASE_MODEL_PATH = "models/production/model.pkl"
META_MODEL_PATH = "models/experiments/exp_t2_meta/meta_lgbm.txt"
CACHE_DIR = "data/aggregates"
TEMP_FEATURE_DIR = "data/jit_features_temp"

# JITで計算するブロック
# 馬の履歴だけで完結するもの + レース内相対統計
JIT_BLOCKS = [
    'base_attributes',
    'history_stats',       # Crucial: Lag, Rolling, Interval (importance: lag1_rank=90, mean_rank_5=52)
    'jockey_stats',        # Crucial: jockey_top3_rate (importance=31), jockey_avg_rank (26)
    'pace_stats',          # avg_first_corner_norm, avg_pci
    'pace_pressure_stats', # nige_pressure_interaction
    'relative_stats',      # Crucial: relative_speed_index_pct (importance=91), relative_last_3f_pct (25)
    'jockey_trainer_stats', # jt_top3_rate_smoothed (importance=8)
    'bloodline_stats',     # sire_win_rate, etc.
    'training_stats',      # If available
    'burden_stats',
    'changes_stats',
    'aptitude_stats',      # Course aptitude (basic)
    'speed_index_stats',   # Absolute speed ratings
    'class_stats',         # Class progression
    'runstyle_fit',        # Runstyle consistency
    'risk_stats',          # Returning from break etc.
    'training_detail',     # Priority 2: Training Detail
    'strategy_pattern',    # Priority 6: Interactions
    'horse_gear'           # New: Blinker
]


def send_discord_notification(webhook_url, content):
    if not webhook_url: return
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

def fetch_jvd_o1_odds(loader, target_year, target_mmdd):
    try:
        q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
        df_odds = pd.read_sql(q, loader.engine)
        if df_odds.empty: return pd.DataFrame()
        
        rows = []
        for _, row in df_odds.iterrows():
            race_id = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
            odds_str = str(row['odds_tansho'])
            # Format: [HorseNum(2)][Odds(4)][Rank(2)]
            for i in range(0, len(odds_str), 8):
                if i + 6 > len(odds_str): break
                hn_raw = odds_str[i:i+2]
                od_raw = odds_str[i+2:i+6]
                if hn_raw.isdigit() and od_raw.isdigit():
                    rows.append({'race_id': race_id, 'horse_number': int(hn_raw), 'odds_live': int(od_raw) / 10.0})
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"Failed to fetch odds: {e}")
        return pd.DataFrame()

def load_human_cache():
    """Load pre-computed Use/Trainer/Sire stats"""
    cache = {}
    files = {
        'jockey': 'jockey_stats.parquet',
        'trainer': 'trainer_stats.parquet',
        'sire': 'sire_stats.parquet',
        'jockey_trainer': 'jockey_trainer_combos.parquet',
        'nicks': 'nicks_stats.parquet'
    }
    for key, fname in files.items():
        path = os.path.join(CACHE_DIR, fname)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Ensure IDs are string
            for c in ['jockey_id', 'trainer_id', 'sire_id']:
                if c in df.columns: df[c] = df[c].astype(str).str.strip()
            cache[key] = df
            logger.info(f"Loaded Cached {key}: {len(df)}")
        else:
            cache[key] = pd.DataFrame()
    return cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYYMMDD")
    parser.add_argument("--race_id", type=str, help="Specific Race ID to predict (optional)")
    parser.add_argument("--discord", action='store_true')
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    race_id_filter = args.race_id
    
    label = f"Race {race_id_filter}" if race_id_filter else f"Date {target_date}"
    logger.info(f"🚀 Starting T2 Production Run (JIT Hybrid) for: {label}")
    
    start_time = time.time()
    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
        
    loader = JraVanDataLoader()
    
    # 1. Load Today's Races
    logger.info("Loading today's races...")
    t_year = target_date[:4]
    t_mmdd = target_date[4:]
    date_str = f"{t_year}-{t_mmdd[:2]}-{t_mmdd[2:]}"
    
    df_today = loader.load(history_start_date=date_str, end_date=date_str)
    if df_today.empty:
        logger.error("No races found.")
        return
        
    if race_id_filter:
        df_today = df_today[df_today['race_id'] == race_id_filter].copy()
        if df_today.empty:
            logger.error(f"Race ID {race_id_filter} not found in today's data.")
            return
        
    horse_ids = df_today['horse_id'].astype(str).unique().tolist()
    logger.info(f"Found {len(horse_ids)} horses today (Filter: {race_id_filter}).")
    
    # 2. JIT History Loading
    logger.info("Loading JIT History...")
    # Load past 12 years to be safe for old horses
    # Use loader to filter by horse_ids
    df_history = loader.load(
        history_start_date="2014-01-01", 
        end_date=(pd.to_datetime(date_str) - timedelta(days=1)).strftime('%Y-%m-%d'),
        horse_ids=horse_ids,
        skip_odds=True
    )
    logger.info(f"Loaded {len(df_history)} history records.")
    
    # 3. Combine
    df_total = pd.concat([df_history, df_today], ignore_index=True)
    df_total.drop_duplicates(subset=['race_id', 'horse_number'], inplace=True)
    df_total.sort_values(['horse_id', 'date'], inplace=True)
    
    # Ensure datatypes
    for c in ['horse_id', 'jockey_id', 'trainer_id', 'sire_id']:
        if c in df_total.columns: df_total[c] = df_total[c].astype(str).str.strip()
    if 'date' in df_total.columns:
        df_total['date'] = pd.to_datetime(df_total['date'])

    # 4. Computed Horse Features (JIT)
    logger.info("Running JIT Feature Pipeline...")
    # Use temp cache dir to avoid polluting main cache
    if os.path.exists(TEMP_FEATURE_DIR):
        shutil.rmtree(TEMP_FEATURE_DIR)
        
    pipeline = FeaturePipeline(cache_dir=TEMP_FEATURE_DIR)
    
    # We only run selected blocks
    try:
        df_jit_feats = pipeline.load_features(df_total, JIT_BLOCKS)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return
    finally:
        if os.path.exists(TEMP_FEATURE_DIR):
            shutil.rmtree(TEMP_FEATURE_DIR)

    # 5. Filter for Today
    today_race_ids = df_today['race_id'].unique()
    df_features = df_jit_feats[df_jit_feats['race_id'].isin(today_race_ids)].copy()
    logger.info(f"Generated features for {len(df_features)} records.")
    
    # 6. Merge Cached Human Stats
    human_cache = load_human_cache()
    
    # Jockey
    if not human_cache['jockey'].empty:
        # Select columns to merge (avoid duplicates if any)
        # Assuming cache has 'jockey_win_rate' etc.
        cols = [c for c in human_cache['jockey'].columns if c != 'jockey_id' and 'jockey_' in c]
        df_features = pd.merge(df_features, human_cache['jockey'][['jockey_id'] + cols], on='jockey_id', how='left')
        
    # Trainer
    if not human_cache['trainer'].empty:
        cols = [c for c in human_cache['trainer'].columns if c != 'trainer_id' and 'trainer_' in c]
        df_features = pd.merge(df_features, human_cache['trainer'][['trainer_id'] + cols], on='trainer_id', how='left')

    # Sire
    if not human_cache['sire'].empty:
        cols = [c for c in human_cache['sire'].columns if c != 'sire_id' and 'sire_' in c]
        df_features = pd.merge(df_features, human_cache['sire'][['sire_id'] + cols], on='sire_id', how='left')
        
    # Jockey-Trainer
    if not human_cache['jockey_trainer'].empty:
         cols = [c for c in human_cache['jockey_trainer'].columns if c not in ['jockey_id', 'trainer_id']]
         df_features = pd.merge(df_features, human_cache['jockey_trainer'][['jockey_id', 'trainer_id'] + cols], on=['jockey_id', 'trainer_id'], how='left')

    # Nicks (Sire x BMS)
    if 'nicks' in human_cache and not human_cache['nicks'].empty:
        cols = [c for c in human_cache['nicks'].columns if c not in ['sire_id', 'bms_id']]
        if 'sire_id' in df_features.columns and 'bms_id' in df_features.columns:
             df_features = pd.merge(df_features, human_cache['nicks'][['sire_id', 'bms_id'] + cols], on=['sire_id', 'bms_id'], how='left')

    # 7. Odds & Prediction
    df_odds = fetch_jvd_o1_odds(loader, t_year, t_mmdd)
    if not df_odds.empty:
        df_features = pd.merge(df_features, df_odds, on=['race_id', 'horse_number'], how='left')
        df_features['odds_10min'] = df_features['odds_live']
    else:
        df_features['odds_10min'] = np.nan
        
    # Load Models
    base_model = joblib.load(BASE_MODEL_PATH)
    booster = base_model.booster_ if hasattr(base_model, 'booster_') else base_model
    base_feats = booster.feature_name()
    
    # Fill missing columns (e.g. relative stats we skipped)
    for f in base_feats:
        if f not in df_features.columns:
            df_features[f] = 0
            
    # Convert object to numeric
    X_base = df_features[base_feats].copy()
    for c in X_base.columns:
        if X_base[c].dtype == 'object':
            X_base[c] = pd.to_numeric(X_base[c], errors='coerce').fillna(0)
            
    df_features['pred_prob'] = booster.predict(X_base.values.astype(float))
    
    # Meta Model
    # Meta Model (Disabled for No-IDs Phase)
    # meta_model = lgb.Booster(model_file=META_MODEL_PATH)
    meta_model = None
    df_features['prob_market'] = 0.8 / (df_features['odds_10min'].fillna(100) + 1e-9)
    df_features['prob_diff'] = df_features['pred_prob'] - df_features['prob_market']
    
    meta_cols = ['pred_prob', 'odds_ratio_60_10', 'prob_market', 'prob_diff', 'bias_adversity_score_mean_5']
    for c in meta_cols:
        if c not in df_features.columns:
            df_features[c] = 1.0 if 'ratio' in c else 0.0
            
    X_meta = df_features[meta_cols].fillna(0)
    if meta_model:
        df_features['meta_prob'] = meta_model.predict(X_meta)
    else:
        df_features['meta_prob'] = df_features['pred_prob'] # Fallback
    
    # Normalize Base Probabilities per race to sum to 1.0 (100%)
    df_features['base_score'] = df_features['pred_prob']
    df_features['base_prob_norm'] = df_features.groupby('race_id')['pred_prob'].transform(lambda x: x / (x.sum() + 1e-9))
    
    # Use Normalized Base Prob for EV Calculation (Experimental)
    df_features['ev'] = df_features['base_prob_norm'] * df_features['odds_10min'].fillna(0)
    
    elapsed = time.time() - start_time
    
    # Output
    log_print(f"\n{'='*60}")
    log_print(f" 🏇 KEIIBA-AI T2 PRODUCTION (JIT Hybrid) ({target_date}) ")
    log_print(f"⏱️  Time: {elapsed:.1f}s")
    
    # Sort Logic: Venue -> Race Number
    # Get distinct races with metadata
    race_meta = df_today[['race_id', 'venue', 'race_number']].drop_duplicates()
    # Sort by venue code (as int) and race number
    race_meta['venue_int'] = pd.to_numeric(race_meta['venue'], errors='coerce')
    race_meta = race_meta.sort_values(['venue_int', 'race_number'])
    
    VENUE_MAP = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }

    df_features = df_features.sort_values(['race_id', 'base_prob_norm'], ascending=[True, False])
    
    for rid in race_meta['race_id']:
        sub = df_features[df_features['race_id'] == rid]
        if sub.empty: continue
        
        info = race_meta[race_meta['race_id'] == rid].iloc[0]
        venue_code = info.get('venue', '00')
        venue_name = VENUE_MAP.get(venue_code, f"Code{venue_code}")
        rnum = info.get('race_number', 0)
        
        log_print(f"\n[{venue_name} {rnum}R] (ID: {rid})")
        log_print(f"{'No':<3} {'Horse':<16} {'BaseScore':<9} {'NormProb':<8} {'EV(Base)':<8} {'Odds':<6}")
        log_print("-" * 65)
        
        for _, row in sub.iterrows():
            if 'horse_name' in df_today.columns:
                h_name = df_today[(df_today['race_id'] == rid) & (df_today['horse_id'] == row['horse_id'])]['horse_name'].iloc[0]
            else:
                h_name = "Unknown"
            
            ev_str = f"{row['ev']:.2f}" if not pd.isna(row['ev']) else "-"
            # Check odds_live or odds_10min
            odds_val = row.get('odds_10min', np.nan)
            odds_str = f"{odds_val:.1f}" if not pd.isna(odds_val) else "-"
            
            log_print(f"{int(row['horse_number']):02} {str(h_name)[:12]:<16} {row['base_score']:.4f}    {row['base_prob_norm']:.2%}   {ev_str:<8} {odds_str:<6}")

    if args.discord:
        nm = NotificationManager(args.webhook_url)
        for rid in race_meta['race_id']:
            sub = df_features[df_features['race_id'] == rid].copy()
            if sub.empty: continue
            
            # Prepare meta
            info = race_meta[race_meta['race_id'] == rid].iloc[0]
            venue_code = info.get('venue', '00')
            venue_name = VENUE_MAP.get(venue_code, f"Code{venue_code}")
            rnum = info.get('race_number', 0)
            
            # Map horse names
            if 'horse_name' in df_today.columns:
                # Merge horse_name
                race_names = df_today[df_today['race_id'] == rid][['horse_id', 'horse_name']]
                sub = pd.merge(sub, race_names, on='horse_id', how='left')
            else:
                sub['horse_name'] = "Unknown"
            
            meta = {
                'race_id': rid,
                'venue_name': venue_name,
                'race_number': rnum,
                'date': target_date
            }
            
            nm.send_jit_report(meta, sub)
            time.sleep(1) # Prevent rate limit

if __name__ == "__main__":
    main()
