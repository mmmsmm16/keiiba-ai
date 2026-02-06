
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import joblib
import yaml
import gc
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from itertools import combinations
from pytorch_tabnet.tab_model import TabNetClassifier

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.models.nn_baseline import SimpleMLP

import requests
import json
import logging
import time
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Discord Notification Helper ---
def send_discord_notification(webhook_url, content):
    if not webhook_url:
        logger.warning("Discord Webhook URL not provided. Skipping notification.")
        return
        
    try:
        # Split content if it exceeds Discord's 2000 char limit
        # Simple splitting by lines
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
        if current_chunk:
            chunks.append(current_chunk)
            
        for chunk in chunks:
            data = {"content": f"```\n{chunk}\n```"}
            
            # Rate Limit Handling / Retry Logic
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(webhook_url, json=data)
                
                if response.status_code == 204:
                    logger.info("Discord notification sent successfully.")
                    time.sleep(0.5) # Gentle delay between chunks
                    break
                elif response.status_code == 429:
                    try:
                        retry_after = response.json().get("retry_after", 1.0)
                    except:
                        retry_after = 1.0
                    logger.warning(f"Rate limited by Discord. Sleeping for {retry_after}s...")
                    time.sleep(float(retry_after) + 0.5)
                    continue
                else:
                    logger.error(f"Failed to send Discord notification: {response.status_code} {response.text}")
                    break
                
    except Exception as e:
        logger.error(f"Error sending Discord notification: {e}")

# --- Configuration ---
FEATURE_BLOCKS = [
    'base_attributes', 'history_stats', 'jockey_stats', 
    'pace_stats', 'bloodline_stats', 'training_stats',
    'burden_stats', 'changes_stats', 'aptitude_stats', 
    'speed_index_stats', 'pace_pressure_stats',
    'relative_stats', 'jockey_trainer_stats',
    'class_stats',
    'risk_stats',
    'course_aptitude',
    'extended_aptitude',
    'runstyle_fit',
    'jockey_trainer_compatibility',
    'interval_aptitude'
]

MODEL_DIR_MLP = "models/experiments/exp_r1_mlp"
MODEL_DIR_TABNET = "models/experiments/exp_r2_tabnet"
CACHE_DIR = "data/features_q8"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_mlp_feature_names(model_dir):
    """Reconstruct exact numerical feature list to match Q8 scaler"""
    exclude_cols = [
        'race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff',
        'interval_type_code', 'frame_number',
        'horse_id', # horse_id is in cat list usually, but if not, exclude unique id
        'apt_int_win', 'apt_int_top3', 'is_first_int_type'
    ]
    
    # Use config from Iteration 8
    mlp_cfg = load_config("config/experiments/exp_r1_mlp.yaml")
    cat_feat_names = mlp_cfg['dataset'].get('categorical_features', [])
    
    q8_path = "data/temp_q8/Q8_features.parquet"
    if os.path.exists(q8_path):
        q8_df = pd.read_parquet(q8_path).head(1)
        
        all_cols_q8 = [c for c in q8_df.columns if c not in exclude_cols]
        num_cols_q8 = [c for c in all_cols_q8 if c not in cat_feat_names]
        
        final_num_feats = []
        for c in num_cols_q8:
            # Replicate type checking from ensemble script
            val = q8_df[c].iloc[0]
            if isinstance(val, str):
                 try: val = float(val)
                 except: pass
            if isinstance(val, (int, float, np.number)):
                final_num_feats.append(c)
        
        return final_num_feats, cat_feat_names
    else:
        return [], []

def predict_mlp(df, model_dir):
    logger.info("Predicting MLP...")
    
    # Load Preprocessors & Config
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    embedding_dims = joblib.load(os.path.join(model_dir, "embedding_dims.pkl"))
    mlp_cfg = load_config("config/experiments/exp_r1_mlp.yaml")
    
    num_feat_names, cat_feat_names = get_mlp_feature_names(model_dir)
    if not num_feat_names:
        raise ValueError("MLP numerical feature names could not be reconstructed.")
        
    # Preprocess Numerical
    # Ensure all numerical cols exist
    for c in num_feat_names:
        if c not in df.columns:
            df[c] = 0
            
    # Preprocess Numerical
    X_num = df[num_feat_names].copy()
    
    # Handle object-to-numeric/float conversion issues explicitly
    for col in X_num.columns:
         if X_num[col].dtype == 'object':
             X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)
    
    X_num = X_num.fillna(0).values
    X_num = scaler.transform(X_num)
    
    # Preprocess Categorical
    X_cat_list = []
    df_proc = df.copy() # Avoid SettingWithCopy
    for cat_col in cat_feat_names:
        if cat_col not in df_proc.columns:
            df_proc[cat_col] = 'unknown'
        df_proc[cat_col] = df_proc[cat_col].astype(str).fillna('unknown')
        
        if cat_col in encoders:
            oe = encoders[cat_col]
            enc = oe.transform(df_proc[[cat_col]].values)
            enc = enc + 1
            X_cat_list.append(enc)
    X_cat = np.hstack(X_cat_list).astype(int) if X_cat_list else np.zeros((len(df), 0), dtype=int)
    
    # Model
    device = torch.device('cpu')
    model = SimpleMLP(
        num_numerical_features=len(num_feat_names),
        embedding_dims=embedding_dims,
        hidden_dims=mlp_cfg['model_params']['hidden_dims'],
        dropout_rate=mlp_cfg['model_params']['dropout_rate']
    ).to(device)
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        xn = torch.tensor(X_num, dtype=torch.float32).to(device)
        xc = torch.tensor(X_cat, dtype=torch.long).to(device)
        preds = model(xn, xc).squeeze().numpy()
    return preds

def predict_tabnet(df, model_dir):
    logger.info("Predicting TabNet...")
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
    
    df_proc = df.copy()
    for c in feature_columns:
        if c not in df_proc.columns: df_proc[c] = 0
        if df_proc[c].dtype == 'object':
             try: df_proc[c] = pd.to_numeric(df_proc[c])
             except: pass
        if pd.api.types.is_numeric_dtype(df_proc[c]):
             df_proc[c] = df_proc[c].fillna(0)
             
    X_eval = df_proc[feature_columns].copy()
    for col, mapping in encoders.items():
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype(str).map(mapping).fillna(0).astype(int)
            
    clf = TabNetClassifier()
    model_path = os.path.join(model_dir, "tabnet_model.zip")
    # TabNet save/load often appends .zip automatically, but we have tabnet_model.zip.zip in some cases.
    if not os.path.exists(model_path) and os.path.exists(model_path + ".zip"):
         model_path = model_path + ".zip"
    elif not os.path.exists(model_path):
         # Try base name without extension as TabNet load_model adds .zip
         model_path = os.path.join(model_dir, "tabnet_model")
         
    clf.load_model(model_path)
    return clf.predict_proba(X_eval.values.astype(float))[:, 1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date (YYYYMMDD)")
    parser.add_argument("--force_regenerate", action='store_true', help="Force clear cache and re-compute features")
    # Discord Notification Arguments
    parser.add_argument("--discord", action='store_true', help="Send results to Discord Webhook")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Discord Webhook URL")
    
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    logger.info(f"🚀 Starting Production Run for: {target_date}")
    
    # Capture output for Discord
    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Prepare Full Raw Data including Today
    logger.info("Loading Historical Raw Data...")
    years = range(2015, int(target_date[:4]) + 1)
    raw_dfs = []
    for y in years:
        fpath = f"data/temp_q1/year_{y}.parquet"
        if os.path.exists(fpath):
            raw_dfs.append(pd.read_parquet(fpath))
    
    if not raw_dfs:
        logger.error("No raw data found in data/temp_q1/")
        return
        
    df_full = pd.concat(raw_dfs, ignore_index=True)
    
    # Identify today's race IDs (if they are in DB but not yet in parquet)
    loader = JraVanDataLoader()
    # In PC-KEIBA jvd_se, metadata like bamei, kishu etc. are included in the same table.
    # Corrected column names: bamei, kishumei_ryakusho, tansho_odds
    # Join with jvd_ra to get hasso_jikoku
    query_se = f"""
        SELECT se.*, bamei, kishumei_ryakusho as kishu_mei, tansho_odds as odds_tansho, ra.hasso_jikoku
        FROM jvd_se se
        LEFT JOIN jvd_ra ra ON 
            se.kaisai_nen = ra.kaisai_nen AND
            se.keibajo_code = ra.keibajo_code AND
            se.kaisai_kai = ra.kaisai_kai AND
            se.kaisai_nichime = ra.kaisai_nichime AND
            se.race_bango = ra.race_bango
        WHERE se.kaisai_nen = '{target_date[:4]}' 
        AND se.kaisai_tsukihi = '{target_date[4:]}'
        AND se.keibajo_code BETWEEN '01' AND '10'
    """
    df_today = pd.read_sql(query_se, loader.engine)
    # Deduplicate columns (bamei, etc might be in se.* and explicit select)
    df_today = df_today.loc[:, ~df_today.columns.duplicated()]
    
    if not df_today.empty:
        df_today['race_id'] = (
            df_today['kaisai_nen'].astype(str).str.strip() + 
            df_today['keibajo_code'].astype(str).str.strip() + 
            df_today['kaisai_kai'].astype(str).str.strip().str.zfill(2) + 
            df_today['kaisai_nichime'].astype(str).str.strip().str.zfill(2) + 
            df_today['race_bango'].astype(str).str.strip().str.zfill(2)
        )
        df_today = df_today.rename(columns={'umaban': 'horse_number', 'ketto_toroku_bango': 'horse_id'})
        # Ensure horse_number is int for merging
        df_today['horse_number'] = pd.to_numeric(df_today['horse_number'], errors='coerce').fillna(0).astype(int)
        df_today['date'] = pd.to_datetime(target_date)
        
        # We already have odds_tansho in df_today
        df_odds = df_today[['race_id', 'horse_number', 'odds_tansho']].copy()
        df_odds['horse_number'] = pd.to_numeric(df_odds['horse_number']).astype(int)
        df_odds['odds_tansho'] = pd.to_numeric(df_odds['odds_tansho'], errors='coerce') / 10.0
        
        # Check if today's races are already in df_full. if not, append.
        existing_ids = set(df_full['race_id'].unique())
        today_ids = set(df_today['race_id'].unique())
        new_races = today_ids - existing_ids
        
        if new_races:
            logger.info(f"Adding {len(new_races)} new races from DB to context.")
            # Standard pandas concat handles missing columns by filling with NaN
            df_new = df_today[df_today['race_id'].isin(new_races)].copy()
            df_full = pd.concat([df_full, df_new], ignore_index=True, sort=False)
    else:
        df_odds = pd.DataFrame()
        today_ids = set()
        logger.info("No races found for today in DB.") # Only info, might be intended
            
    # Pre-calculate time_diff and ensure numeric types
    if 'time' in df_full.columns:
        df_full['time'] = pd.to_numeric(df_full['time'], errors='coerce')
        df_full['last_3f'] = pd.to_numeric(df_full.get('last_3f'), errors='coerce') # Fix for PCI calc
        min_time = df_full.groupby('race_id')['time'].transform('min')
        df_full['time_diff'] = (df_full['time'] - min_time).fillna(99.9)
    if 'rank' not in df_full.columns:
        df_full['rank'] = np.nan
        
    # 2. Feature Generation (Full Pipe for context)
    if args.force_regenerate:
        logger.info("Clearing Feature Cache...")
        import shutil
        if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    else:
        # We MUST clear the cache for blocks that are history-dependent to include today
        # simplest: clear all or clear for today's races? 
        # FeaturePipeline doesn't support incremental. Let's clear to be safe.
        logger.info("Clearing Cache for re-computation...")
        for b in FEATURE_BLOCKS:
            cp = os.path.join(CACHE_DIR, f"{b}.parquet")
            if os.path.exists(cp): os.remove(cp)

    logger.info("Generating Features (This may take a few minutes)...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    df_features_full = pipeline.load_features(df_full, FEATURE_BLOCKS)
    
    # 3. Filter Today and Predict
    today_ids_str = [str(rid) for rid in today_ids]
    # Keep original metadata for reporting (Get from df_today directly to ensure columns exist)
    # Included hasso_jikoku in meta_cols
    meta_cols = ['race_id', 'horse_number', 'bamei', 'kishu_mei', 'hasso_jikoku']
    if not df_today.empty:
        # Ensure hasso_jikoku exists, fill if missing
        if 'hasso_jikoku' not in df_today.columns:
             df_today['hasso_jikoku'] = '0000'
        df_meta = df_today[meta_cols].drop_duplicates()
    else:
        df_meta = pd.DataFrame(columns=meta_cols)
    
    df_today_features = df_features_full[df_features_full['race_id'].astype(str).isin(today_ids_str)].copy()
    
    if df_today_features.empty:
        logger.warning("No features generated for today.")
        return # Exit if no features

    # 3. Predict
    # ensemble logic
    try:
        preds_mlp = predict_mlp(df_today_features, MODEL_DIR_MLP)
        preds_tabnet = predict_tabnet(df_today_features, MODEL_DIR_TABNET)
        
        df_today_features['pred_prob'] = (preds_mlp + preds_tabnet) / 2
        df_today_features['pred_rank'] = df_today_features.groupby('race_id')['pred_prob'].rank(ascending=False)
        
        # 4. Recommendation
        logger.info("Generating Recommendations...")
        
        # Merge metadata back for display
        df_today_features = pd.merge(df_today_features, df_meta, on=['race_id', 'horse_number'], how='left')
        df_today_features = pd.merge(df_today_features, df_odds[['race_id', 'horse_number', 'odds_tansho']], on=['race_id', 'horse_number'], how='left')
        
        THRESHOLD = 0.10
        ROI_PROB_THRESHOLD = 0.30
        ROI_ODDS_MIN = 2.0
        ROI_ODDS_MAX = 20.0
        
        # JRA Venue Code Map
        JRA_VENUES = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
            '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
        }
        
        log_print("\n" + "="*80)
        log_print(" 🏇 KEIIBA-AI 2025年 本番予測実行 ")
        log_print(" アンサンブルモデル (MLP + TabNet) | 戦略: 三連複1頭軸流し (相手7頭)")
        log_print("="*80)
        
        # Sort by Hasso Jikoku
        # Create a race-level dataframe to sort
        race_list = df_today_features[['race_id', 'hasso_jikoku']].drop_duplicates()
        # Convert hasso_jikoku to something sortable (it's string HHMM, so string sort works)
        race_list['hasso_jikoku'] = race_list['hasso_jikoku'].fillna('9999').astype(str)
        race_list = race_list.sort_values(['hasso_jikoku', 'race_id'])
        
        for rid in race_list['race_id']:
            # Parse Race ID for display
            try:
                r_year = rid[:4]
                r_venue_code = rid[4:6]
                r_race_num = int(rid[10:12])
                
                venue_name = JRA_VENUES.get(r_venue_code, f"Unknown({r_venue_code})")
                display_date = f"{target_date[:4]}年{target_date[4:6]}月{target_date[6:]}日"
                
                # Get start time for display
                st_time = race_list[race_list['race_id'] == rid]['hasso_jikoku'].iloc[0]
                st_str = f"{st_time[:2]}:{st_time[2:]}" if len(st_time) == 4 else st_time
                
                header_str = f"{display_date} {venue_name} {r_race_num}R ({st_str}発走)"
            except:
                header_str = f"Race ID: {rid}"

            sub = df_today_features[df_today_features['race_id'] == rid].sort_values('pred_prob', ascending=False)
            
            log_print(f"\n[{header_str}] (ID: {rid})")
            log_print(f"{'No':<3} {'馬名':<16} {'騎手':<12} {'予測値':<8} {'オッズ':<6}")
            log_print("-" * 55)
            
            for _, row in sub.iterrows():
                bamei = str(row['bamei'])[:12]
                kishu = str(row['kishu_mei'])[:10]
                prob = f"{row['pred_prob']:.2%}"
                odds = f"{row['odds_tansho']:.1f}" if not pd.isna(row['odds_tansho']) else "---"
                log_print(f"{int(row['horse_number']):02} {bamei:<16} {kishu:<12} {prob:<8} {odds:<6}")
                
            top1 = sub.iloc[0]
            if top1['pred_prob'] >= THRESHOLD:
                axis = int(top1['horse_number'])
                followers = sub.iloc[1:8]['horse_number'].astype(int).tolist()
                
                log_print(f"\n💡 推奨買い目:")
                log_print(f"  - 軸馬: {axis:02} ({top1['bamei']})")
                log_print(f"  - 相手: {', '.join([f'{f:02}' for f in followers])}")
                
                # ROI Condition Check
                is_high_roi = (top1['pred_prob'] >= ROI_PROB_THRESHOLD) and \
                             (not pd.isna(top1['odds_tansho'])) and \
                             (ROI_ODDS_MIN <= top1['odds_tansho'] <= ROI_ODDS_MAX)
                
                if is_high_roi:
                    log_print(f"\n  ★高回収率アラート★")
                    log_print(f"  条件合致: 確率 {top1['pred_prob']:.2%} >= 30% かつ オッズ 2.0-20.0倍")
                    log_print(f"  勝負レース推奨: 単勝 / ワイド流し / 三連複1頭軸")
                
            log_print("\n" + "." * 80)
            
        log_print("\n" + "="*80)
        
        # Send Discord Notification
        if args.discord:
            msg = "".join(output_log)
            logger.info("Sending Discord Notification...")
            send_discord_notification(args.webhook_url, msg)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
