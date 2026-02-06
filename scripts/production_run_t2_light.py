"""
Production Run T2 - Lightweight Version with Cached Aggregates
==============================================================
事前計算済みキャッシュを使用して、当日レースのみを高速に予測する。
過去データの再読み込み・再計算を避け、1分以内で実行可能。
"""

import os
import sys
import json
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_MODEL_PATH = "models/experiments/exp_t2_track_bias/model.pkl"
META_MODEL_PATH = "models/experiments/exp_t2_meta/meta_lgbm.txt"
CACHE_DIR = "data/aggregates"

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

def fetch_jvd_o1_odds(loader, target_year, target_mmdd):
    """jvd_o1 からオッズを取得"""
    try:
        q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
        df_odds = pd.read_sql(q, loader.engine)
        
        if df_odds.empty:
            return pd.DataFrame()
        
        rows = []
        for _, row in df_odds.iterrows():
            race_id = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
            odds_str = str(row['odds_tansho'])
            
            # Format: [HorseNum(2)][Odds(4)][Rank(2)] = 8 chars per horse
            for i in range(0, len(odds_str), 8):
                if i + 6 > len(odds_str): break
                
                horse_num_raw = odds_str[i:i+2]
                odds_raw = odds_str[i+2:i+6]
                
                if horse_num_raw.isdigit() and odds_raw.isdigit():
                    horse_num = int(horse_num_raw)
                    odds_val = int(odds_raw) / 10.0
                    rows.append({'race_id': race_id, 'horse_number': horse_num, 'odds_live': odds_val})
        
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"Failed to fetch jvd_o1: {e}")
        return pd.DataFrame()

def load_cached_aggregates():
    """事前計算済みキャッシュを読み込み"""
    cache = {}
    
    cache_files = {
        'jockey': 'jockey_stats.parquet',
        'trainer': 'trainer_stats.parquet',
        'sire': 'sire_stats.parquet',
        'horse': 'horse_stats.parquet',
        'jockey_trainer': 'jockey_trainer_combos.parquet',
        'course_aptitude': 'course_aptitude.parquet'
    }
    
    for key, filename in cache_files.items():
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.exists(filepath):
            cache[key] = pd.read_parquet(filepath)
            logger.info(f"Loaded cache: {key} ({len(cache[key])} records)")
        else:
            cache[key] = pd.DataFrame()
            logger.warning(f"Cache not found: {filepath}")
    
    return cache

def compute_features_with_cache(df_today, cache):
    """キャッシュから統計をマージして特徴量を生成"""
    features = df_today[['race_id', 'horse_number', 'horse_id']].copy()
    
    # Copy basic features if available
    basic_cols = ['sex', 'age', 'weight', 'weight_diff', 'impost', 'wakuban', 'umaban', 
                  'jockey_id', 'trainer_id', 'sire_id', 'mare_id', 'venue', 'surface', 
                  'distance', 'track_condition_code', 'weather_code']
    
    for col in basic_cols:
        if col in df_today.columns:
            features[col] = df_today[col]
            
    if 'sex' in df_today.columns:
        features['sex_code'] = df_today['sex'].map({'1': 0, '2': 1, '3': 2}).fillna(0)
    if 'distance' in df_today.columns:
        features['distance'] = pd.to_numeric(df_today['distance'], errors='coerce').fillna(1800)
    if 'surface' in df_today.columns:
        features['surface_code'] = df_today['surface'].apply(lambda x: 0 if str(x).startswith('1') else 1)
    
    features['horse_number'] = df_today['horse_number']
    if 'frame_number' in df_today.columns:
        features['frame_number'] = pd.to_numeric(df_today['frame_number'], errors='coerce').fillna(4)
    
    # 2. 馬統計（キャッシュから）
    if not cache['horse'].empty:
        # Columns in new cache schema:
        # horse_id, horse_win_rate, horse_top3_rate, horse_mean_rank, run_count, 
        # last_race_date, lag1_rank, lag1_time_diff, mean_rank_5, mean_time_diff_5
        
        # We need to map these to what the model expects or use new names.
        target_cols = ['horse_id', 'horse_mean_rank', 'horse_win_rate', 'run_count', 'last_race_date',
                       'lag1_rank', 'lag1_time_diff', 'mean_rank_5', 'mean_time_diff_5']
        
        valid_cols = [c for c in target_cols if c in cache['horse'].columns]
        
        features = pd.merge(features, cache['horse'][valid_cols], on='horse_id', how='left')
        
        # Calculate Interval
        if 'last_race_date' in features.columns and 'date' in df_today.columns:
            features['date_dt'] = pd.to_datetime(df_today['date'])
            features['last_race_date_dt'] = pd.to_datetime(features['last_race_date'], errors='coerce')
            features['interval'] = (features['date_dt'] - features['last_race_date_dt']).dt.days
            features['interval'] = features['interval'].fillna(0)
            features.drop(columns=['date_dt', 'last_race_date_dt', 'last_race_date'], inplace=True)
    
    # Alias for backward compatibility
    if 'run_count' in features.columns and 'horse_races' not in features.columns:
        features['horse_races'] = features['run_count']
    if 'mean_rank_5' in features.columns:
        features['mean_rank_5'] = features['mean_rank_5'].fillna(features['horse_mean_rank'])

    # Default filling for critical features
    for col in ['horse_mean_rank', 'horse_win_rate', 'horse_races']:
        if col not in features.columns:
            features[col] = 8 if 'rank' in col else 0
        features[col] = features[col].fillna(8 if 'rank' in col else 0)
    
    # 3. 騎手統計
    if not cache['jockey'].empty and 'jockey_id' in df_today.columns:
        jockey_info = df_today[['race_id', 'horse_number', 'jockey_id']].copy()
        jockey_merged = pd.merge(jockey_info, cache['jockey'], on='jockey_id', how='left')
        # Select relevant stats
        stat_cols = ['jockey_win_rate', 'jockey_top3_rate', 'jockey_avg_rank']
        found_cols = [c for c in stat_cols if c in jockey_merged.columns]
        if found_cols:
            features = pd.merge(features, jockey_merged[['race_id', 'horse_number'] + found_cols], 
                               on=['race_id', 'horse_number'], how='left')

    # 4. 調教師統計
    if not cache['trainer'].empty and 'trainer_id' in df_today.columns:
        trainer_info = df_today[['race_id', 'horse_number', 'trainer_id']].copy()
        trainer_merged = pd.merge(trainer_info, cache['trainer'], on='trainer_id', how='left')
        stat_cols = ['trainer_win_rate', 'trainer_top3_rate']
        found_cols = [c for c in stat_cols if c in trainer_merged.columns]
        if found_cols:
            features = pd.merge(features, trainer_merged[['race_id', 'horse_number'] + found_cols], 
                               on=['race_id', 'horse_number'], how='left')

    # 5. 種牡馬統計
    if not cache['sire'].empty and 'sire_id' in df_today.columns:
        sire_info = df_today[['race_id', 'horse_number', 'sire_id']].copy()
        sire_merged = pd.merge(sire_info, cache['sire'], on='sire_id', how='left')
        stat_cols = ['sire_win_rate', 'sire_top3_rate']
        found_cols = [c for c in stat_cols if c in sire_merged.columns]
        if found_cols:
            features = pd.merge(features, sire_merged[['race_id', 'horse_number'] + found_cols], 
                               on=['race_id', 'horse_number'], how='left')
                               
    # 6. コース適性 (Course Aptitude)
    # df_today needs 'keibajo_code', 'surface', 'distance' to compute course_key
    if not cache['course_aptitude'].empty and 'keibajo_code' in df_today.columns:
        # Re-derive course key logic
        # Assuming df_today has 'distance_cat' derived? No, we need to derive it.
        # This is getting complex for lightweight. 
        # Alternatively, we can try to merge by horse_id and rely on the fact that cache might have average course aptitude?
        # No, course aptitude is (horse, course) pair.
        # We need to compute course_key for today's race.
        
        # Simplified: We skip course aptitude implementation in light script for now unless critical.
        # It requires same logic as cache builder.
        pass

    # 7. Track Bias (Defaults)
    features['bias_adversity_score_mean_5'] = 0.0
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date (YYYYMMDD)")
    parser.add_argument("--discord", action='store_true', help="Send results to Discord")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Webhook")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    logger.info(f"🚀 Starting T2 Production Run (Cached) for: {target_date}")
    
    start_time = time.time()
    output_log = []
    
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    logger.info("Loading cached aggregates...")
    cache = load_cached_aggregates()
    
    loader = JraVanDataLoader()
    logger.info("Loading today's data only...")
    
    target_year = target_date[:4]
    target_mmdd = target_date[4:]
    
    df_today = loader.load(
        history_start_date=f"{target_year}-{target_mmdd[:2]}-{target_mmdd[2:]}", 
        end_date=f"{target_year}-{target_mmdd[:2]}-{target_mmdd[2:]}"
    )
    
    logger.info(f"Today: {len(df_today)} records")
    
    if df_today.empty:
        logger.error(f"No races found for {target_date}")
        return
    
    # Odds
    df_live_odds = fetch_jvd_o1_odds(loader, target_year, target_mmdd)
    if not df_live_odds.empty:
        df_today = pd.merge(df_today, df_live_odds, on=['race_id', 'horse_number'], how='left')
        df_today['odds_10min'] = df_today['odds_live']
    else:
        df_today['odds_10min'] = np.nan
    
    # Feature Generation
    for col in ['horse_id', 'jockey_id', 'trainer_id', 'sire_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.strip()
            
    for key in cache:
        if not cache[key].empty:
            for id_col in ['horse_id', 'jockey_id', 'trainer_id', 'sire_id']:
                if id_col in cache[key].columns:
                    cache[key][id_col] = cache[key][id_col].astype(str).str.strip()

    logger.info("Generating features...")
    df_features = compute_features_with_cache(df_today, cache)
    
    # Merge odds back
    df_features = pd.merge(df_features, df_today[['race_id', 'horse_number', 'odds_10min']], on=['race_id', 'horse_number'], how='left')
    
    # Load Model
    logger.info("Predicting with Base Model (T2)...")
    base_model = joblib.load(BASE_MODEL_PATH)
    booster = base_model.booster_ if hasattr(base_model, 'booster_') else base_model
    
    base_feats = booster.feature_name()
    
    # Fill missing
    for f in base_feats:
        if f not in df_features.columns:
            df_features[f] = 0
            
    X_base = df_features[base_feats].copy()
    for col in X_base.columns:
        # Handle object columns
        if X_base[col].dtype == 'object':
            X_base[col] = pd.to_numeric(X_base[col], errors='coerce').fillna(0)
            
    df_features['pred_prob'] = booster.predict(X_base.values.astype(float))
    
    # Debug Stats
    logger.info(f"Pred Prob Stats: Min={df_features['pred_prob'].min():.4f}, Max={df_features['pred_prob'].max():.4f}")
    if len(df_features['pred_prob'].unique()) < 10:
        logger.warning(f"Low variance in pred_prob! Unique: {len(df_features['pred_prob'].unique())}")
        
    # Meta Model
    logger.info("Predicting with Meta Model...")
    meta_model = lgb.Booster(model_file=META_MODEL_PATH)
    
    df_features['prob_market'] = 0.8 / (df_features['odds_10min'].fillna(100) + 1e-9)
    df_features['prob_diff'] = df_features['pred_prob'] - df_features['prob_market']
    
    meta_feats = ['pred_prob', 'odds_ratio_60_10', 'prob_market', 'prob_diff', 'bias_adversity_score_mean_5']
    for mf in meta_feats:
        if mf not in df_features.columns:
            # odds_ratio should default to 1.0 (no change), others to 0.0
            if 'ratio' in mf:
                df_features[mf] = 1.0
            else:
                df_features[mf] = 0.0
            
    X_meta = df_features[meta_feats].fillna(0)
    df_features['meta_prob'] = meta_model.predict(X_meta)
    
    # EV
    df_features['ev'] = df_features['meta_prob'] * df_features['odds_10min'].fillna(0)
    
    # Output
    elapsed = time.time() - start_time
    log_print(f"\n{'='*60}")
    log_print(f" 🏇 KEIIBA-AI T2 PRODUCTION PREDICTIONS ({target_date}) ")
    log_print(f"⏱️  処理時間: {elapsed:.1f}秒")
    
    df_features = df_features.sort_values(['race_id', 'meta_prob'], ascending=[True, False])
    
    for rid in df_today['race_id'].unique():
        sub = df_features[df_features['race_id'] == rid]
        race_info = df_today[df_today['race_id'] == rid].iloc[0]
        venue = race_info.get('venue', 'Unk')
        rnum = race_info.get('race_number', 0)
        
        log_print(f"\n[{venue} {rnum}R] (ID: {rid})")
        log_print(f"{'No':<3} {'Horse':<16} {'MetaProb':<8} {'EV':<6} {'Odds':<6}")
        log_print("-" * 50)
        
        for _, row in sub.iterrows():
            h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == row['horse_number'])]
            name = str(h_sub.iloc[0].get('horse_name', 'Unknown'))[:12] if not h_sub.empty else "Unknown"
            
            ev_val = row['ev']
            odds_val = row['odds_10min']
            ev_str = f"{ev_val:.2f}" if not pd.isna(ev_val) else "---"
            odds_str = f"{odds_val:.1f}" if not pd.isna(odds_val) else "---"
            log_print(f"{int(row['horse_number']):02} {name:<16} {row['meta_prob']:.2%} {ev_str:<6} {odds_str:<6}")
            
    if args.discord:
        send_discord_notification(args.webhook_url, "".join(output_log))

if __name__ == "__main__":
    main()
