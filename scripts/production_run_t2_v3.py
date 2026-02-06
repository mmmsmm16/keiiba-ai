
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
    parser.add_argument("--exp", type=str, default="exp_t2_refined_v3", help="Experiment Name")
    parser.add_argument("--discord", action='store_true', help="Send results to Discord")
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Webhook")
    parser.add_argument("--race_id", type=str, help="Target Race ID (for JIT execution)")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    exp_name = args.exp
    
    logger.info(f"🚀 Starting T2 Production Run (V3) for: {target_date} (Exp: {exp_name})")
    
    # Paths
    base_dir = f"models/experiments/{exp_name}"
    model_path_pkl = os.path.join(base_dir, "model.pkl")
    model_path_cbm = os.path.join(base_dir, "model.cbm")
    config_path = os.path.join(base_dir, "config.yaml")
    calib_path = os.path.join(base_dir, "calibrator.pkl") # If exists
    
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return
        
    # Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    feature_blocks = config.get('features', [])
    cat_features = config.get('dataset', {}).get('categorical_features', [])
    model_type = config.get('model_params', {}).get('model_type', 'lightgbm')
    
    logger.info(f"Loaded config. Feature Blocks: {len(feature_blocks)}")

    # Load Model
    model = None
    is_cb = False
    if model_type == 'catboost' or os.path.exists(model_path_cbm):
        import catboost as cb
        if os.path.exists(model_path_cbm):
            model = cb.CatBoostClassifier()
            model.load_model(model_path_cbm)
            is_cb = True
            logger.info(f"Loaded CatBoost model: {model_path_cbm}")
    
    if model is None and os.path.exists(model_path_pkl):
        model = joblib.load(model_path_pkl)
        logger.info(f"Loaded pickle model: {model_path_pkl}")
        
    if model is None:
        logger.error("Model file not found.")
        return

    # Load Calibrator
    calibrator = None
    if os.path.exists(calib_path):
        calibrator = joblib.load(calib_path)
        logger.info(f"Loaded Calibrator: {calib_path}")

    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Load Data
    loader = JraVanDataLoader()
    # Load history (e.g. 1 year context) + Today
    # For feature calculation like 5-race-avg, we need history.
    # Production loader usually handles this smartly or we request a range.
    # Assuming we need some history. Let's load 2024-01-01 to Target Date.
    start_year = str(int(target_date[:4]) - 1) + "0101"
    df_raw = loader.load(history_start_date=start_year, end_date=target_date)
    
    # Filter only today for prediction
    today_dt = pd.to_datetime(target_date)
    
    # Numeric Conversion
    numeric_cols = ['time', 'last_3f', 'rank', 'weight', 'weight_diff', 'impost', 'honshokin', 'fukashokin']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    # Time Diff helper
    if 'time' in df_raw.columns:
        min_times = df_raw.groupby('race_id')['time'].transform('min')
        df_raw['time_diff'] = (df_raw['time'] - min_times).fillna(0)
    else:
        df_raw['time_diff'] = 0
    
    # === Fetch live odds from jvd_o1 for today's races ===
    def fetch_jvd_o1_odds(loader, target_year, target_mmdd):
        try:
            q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
            df_odds = pd.read_sql(q, loader.engine)
            if df_odds.empty: return pd.DataFrame()
            
            rows = []
            for _, row in df_odds.iterrows():
                race_id = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
                odds_str = str(row['odds_tansho'])
                # Format: Horse(2) + Odds(4) + Pop(2) = 8 bytes
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
                                # "----" or invalid
                                start_odds_if_avail = pd.to_numeric(odds_part, errors='coerce') # Unlikely to work
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
        logger.error(f"No races found for {target_date} (or specific race_id) in DB.")
        return

    # 2. Feature Generation
    # Use exp-specific cache dir to avoid stale cache from other versions
    exp_cache_dir = os.path.join(CACHE_DIR, exp_name)
    pipeline = FeaturePipeline(cache_dir=exp_cache_dir)
    df_features = pipeline.load_features(df_raw, feature_blocks)
    
    # Filter features for today
    today_ids = df_today['race_id'].unique()
    df_today_features = df_features[df_features['race_id'].isin(today_ids)].copy()
    
    # Merge odds back if missing
    if 'odds_10min' not in df_today_features.columns:
        if not df_live_odds.empty:
            df_today_features = pd.merge(
                df_today_features, 
                df_live_odds[['race_id', 'horse_number', 'odds_live']].rename(columns={'odds_live': 'odds_10min'}), 
                on=['race_id', 'horse_number'], 
                how='left'
            )
        elif 'odds_10min' in df_today.columns:
            df_today_features = pd.merge(
                df_today_features,
                df_today[['race_id', 'horse_number', 'odds_10min']],
                on=['race_id', 'horse_number'],
                how='left'
            )
        else:
            df_today_features['odds_10min'] = np.nan
    
    # 3. Model Prediction
    logger.info("Predicting...")
    
    # Prepare X
    drop_cols = ['race_id', 'date', 'rank', 'target', 'year', 'time_diff', 'odds', 'horse_id', 'odds_10min', 'odds_live']
    X_cols = [c for c in df_today_features.columns if c not in drop_cols and not pd.api.types.is_datetime64_any_dtype(df_today_features[c])]
    
    # Define X first (ensure copy)
    X = df_today_features[X_cols].copy()

    # Apply exclude_features from config (Crucial for No-IDs)
    dataset_cfg = config.get('dataset', {})
    exclude_cols = dataset_cfg.get('exclude_features', [])
    if exclude_cols:
        logger.info(f"Dropping excluded features per config: {len(exclude_cols)} cols")
        X = X.drop(columns=[c for c in exclude_cols if c in X.columns])

    # Determine Expected Categorical Features from Model
    final_cat_features = []
    try:
        # Check if model is Booster or sklearn wrapper
        booster = None
        if hasattr(model, 'booster_'):
            booster = model.booster_
        else:
            booster = model # Assume Booster
            
        if booster:
            cat_indices = []
            # Method 1: Check params (Robust for Booster)
            if hasattr(booster, 'params'):
                cat_indices = booster.params.get('categorical_column', [])
            
            # Method 2: Check dump_model (Fallback)
            if not cat_indices and hasattr(booster, 'dump_model'):
                model_json = booster.dump_model()
                cat_indices = model_json.get('categorical_feature', [])
                if not cat_indices:
                    cat_indices = model_json.get('categorical_column', [])
            
            if cat_indices:
                all_features = booster.feature_name()
                cat_indices = [int(i) for i in cat_indices]
                final_cat_features = [all_features[i] for i in cat_indices]
                logger.info(f"Using {len(final_cat_features)} categorical features extracted from model.")
            else:
                logger.warning("No categorical features found in model params/dump.")

    except Exception as e:
        logger.warning(f"Failed to extract categorical features from model: {e}")
        auto_cat = [c for c in X.columns if X[c].dtype == 'object']
        final_cat_features = list(set(auto_cat))

    # Force Cast
    for c in final_cat_features:
        if c in X.columns:
            X[c] = X[c].fillna("missing").astype('category')

    # Feature Alignment & Type Enforcement
    def get_expected_feature_names(model):
        if hasattr(model, "booster_") and model.booster_ is not None:
            return model.booster_.feature_name()
        if hasattr(model, "feature_name") and callable(model.feature_name):
            return model.feature_name()
        if hasattr(model, "feature_name_"):
            return list(model.feature_name_)
        return None

    expected_features = get_expected_feature_names(model)
    if expected_features is None:
        logger.warning("Could not extract expected feature names from model. Using current X columns (risk).")
    else:
        # 1. Align Columns
        current_cols = set(X.columns)
        expected_set = set(expected_features)
        
        missing = [c for c in expected_features if c not in current_cols]
        extra = [c for c in current_cols if c not in expected_set]

        logger.info(f"[Feature Align] expected={len(expected_features)} current={X.shape[1]} missing={len(missing)} extra={len(extra)}")
        
        if missing:
            logger.error(f"[Feature Align] missing columns: {missing[:10]}... (Total: {len(missing)})")
            for c in missing:
                X[c] = np.nan
        if extra:
            X = X.drop(columns=extra)

        X = X.reindex(columns=expected_features)
        
        # Debugging: Check for all-NaN columns or constant columns
        nan_counts = X.isna().sum()
        all_nan_cols = nan_counts[nan_counts == len(X)].index.tolist()
        if all_nan_cols:
            logger.warning(f"All-NaN columns: {len(all_nan_cols)}/{len(expected_features)}. Head: {all_nan_cols[:10]}")
        
        # 2. Convert ALL features to numeric (avoid categorical mismatch)
        # LightGBM Booster can work with numeric codes instead of categorical
        logger.info("Converting all features to numeric...")
        for col in expected_features:
            if X[col].dtype.name == 'category':
                X[col] = X[col].cat.codes
            elif X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999.0)


    # Predict
    try:
        logger.info(f"Predicting with shape: {X.shape}")
        
        # Convert to numpy array to avoid categorical feature mismatch
        X_np = X.values.astype(np.float64)
        
        # Check model type and predict appropriately
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(X_np)[:, 1]
        else:
            raw_preds = model.predict(X_np)
            logger.info(f"Raw Score Sample: {raw_preds[:5]}")
            preds = 1 / (1 + np.exp(-raw_preds))  # Sigmoid
        
        logger.info(f"Prediction range: min={preds.min():.4f}, max={preds.max():.4f}")
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e


    df_today_features['pred_prob'] = preds
    
    # Normalize probabilities to sum to 1.0 per race (User Request)
    # This ensures all horses in a race sum to 100%
    df_today_features['pred_prob'] = df_today_features.groupby('race_id')['pred_prob'].transform(lambda x: x / x.sum())
    logger.info(f"Normalized predictions. Sample: {df_today_features['pred_prob'].head(5).tolist()}")
    
    # EV Calculation
    df_today_features['ev'] = df_today_features['pred_prob'] * df_today_features['odds_10min'].fillna(0)
    
    # 4. Output
    log_print(f"\n{'='*60}")
    log_print(f" 🏇 KEIIBA-AI PRODUCTION (V3) PREDICTIONS ({target_date}) ")
    log_print(f"{'='*60}")
    
    VENUE_MAP = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }

    df_today_features = df_today_features.sort_values(['race_id', 'pred_prob'], ascending=[True, False])
    
    # Sort race_ids to ensure order (Race 1 -> 12)
    # race_id format: YYYY(4) + Place(2) + Kai(2) + Day(2) + Race(2)
    today_ids = sorted(df_today_features['race_id'].unique())
    
    for rid in today_ids:
        sub = df_today_features[df_today_features['race_id'] == rid]
        race_info = df_today[df_today['race_id'] == rid].iloc[0]
        
        venue_code = race_info.get('keibajo_code')
        if not venue_code and 'venue' in race_info:
            venue_code = str(race_info['venue']).zfill(2)
        elif not venue_code:
             # Extract from ID
             venue_code = rid[4:6]
             
        venue_name = VENUE_MAP.get(venue_code, venue_code)
        
        race_num_val = race_info.get('race_bango')
        if not race_num_val:
            race_num_val = rid[-2:]
        race_num = int(race_num_val)
        
        log_print(f"\n[{venue_name} {race_num}R] (ID: {rid})")
        log_print(f"{'馬番':<4} {'馬名':<16} {'予測確率':<8} {'期待値':<6} {'オッズ':<6}")
        log_print("-" * 50)
        
        for _, row in sub.iterrows():
            h_sub = df_today[(df_today['race_id'] == rid) & (df_today['horse_number'] == row['horse_number'])]
            name = str(h_sub.iloc[0].get('horse_name', 'Unknown'))[:12] if not h_sub.empty else "Unknown"
            
            ev_str = f"{row['ev']:.2f}" if not pd.isna(row['ev']) else "---"
            odds_str = f"{row.get('odds_10min', np.nan):.1f}" if not pd.isna(row.get('odds_10min', np.nan)) else "---"
            
            # Betting Recommendation (Win-Top1 EV Strategy from Backtest)
            # Best: EV >= 1.5, Prob >= 0.10 -> ROI ~103%
            rec_mark = ""
            stats_prob = row['pred_prob']
            stats_odds = row.get('odds_10min', 0)
            stats_ev = row.get('ev', 0)
            
            # Is this the Top1 predicted horse?
            is_top1 = (sub['pred_prob'].idxmax() == row.name)
            
            if is_top1 and stats_ev >= 1.5 and stats_prob >= 0.10:
                rec_mark = "🎯[単勝推奨] EV={:.2f}".format(stats_ev)
            elif is_top1 and stats_ev >= 1.0:
                rec_mark = "📊[検討] EV={:.2f}".format(stats_ev)
            elif stats_prob >= 0.36 and stats_odds >= 1.5:
                rec_mark = "🔥[PLACE]"
                
            log_print(f"{int(row['horse_number']):02}   {name:<16} {row['pred_prob']:.2%}   {ev_str:<6} {odds_str:<6} {rec_mark}")
            
    log_print(f"\n{'='*60}")
    log_print("🚀 Production Run Completed.")
    
    if args.discord:
        send_discord_notification(args.webhook_url, "".join(output_log))

if __name__ == "__main__":
    main()
