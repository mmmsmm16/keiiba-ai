"""
Production Prediction Script V14
================================
Gap Model V14 (Leak Free) Inference
Targets: Place Betting (Odds 10-50, Gap Rank 1-3)
Features:
- Handles Shift-JIS/UTF-8 output encoding.
- Robust Time-Series Odds fetching.
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
from datetime import datetime, timedelta
import unicodedata

def get_visual_width(s):
    width = 0
    for c in str(s):
        if unicodedata.east_asian_width(c) in 'WF':
            width += 2
        else:
            width += 1
    return width

def pad_japanese(s, width, align='left'):
    s = str(s)
    cur_width = get_visual_width(s)
    pad_len = max(0, width - cur_width)
    if align == 'left':
        return s + ' ' * pad_len
    else:
        return ' ' * pad_len + s

# Force UTF-8 Output for Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
# from src.preprocessing.pipeline import FeaturePipeline # Old
from src.preprocessing.feature_pipeline import FeaturePipeline

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
MODEL_PATH = 'models/experiments/exp_gap_v14_production/model_v14.pkl'
FEATURES_PATH = 'models/experiments/exp_gap_v14_production/features.csv'
CACHE_DIR = 'data/features_v14/prod_cache' # Dedicated cache for production

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='YYYYMMDD')
    parser.add_argument('--place_cutoff', type=float, default=3, help='Rank Diff threshold to highlight')
    parser.add_argument('--force', action='store_true', help='Force recompute features')
    args = parser.parse_args()
    
    date_str = args.date
    target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
        
    logger.info("Loading Model V14...")
    model = joblib.load(MODEL_PATH)
    features_list = pd.read_csv(FEATURES_PATH)['feature'].tolist()
    
    # 2. Load Raw Data
    logger.info(f"Loading Raw Data for {target_date}...")
    loader = JraVanDataLoader()
    # history_start_date: context needed for features (e.g. 5 races history)
    # Load past 1 year to be safe
    start_history = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    
    df_raw = loader.load(history_start_date=start_history, end_date=target_date, skip_odds=False)
    
    # Filter for target date races
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_target = df_raw[df_raw['date'] == pd.to_datetime(target_date)].copy()
    logger.info(f"Raw data columns: {df_raw.columns.tolist()}")
    if not df_target.empty:
        logger.info(f"Sample odds/popularity if available: {df_target.filter(like='odds').columns.tolist()} {df_target.filter(like='ninki').columns.tolist()}")
    
    if df_target.empty:
        logger.warning("No races found for target date.")
        return
        
    logger.info(f"Found {len(df_target)} horses for {target_date}")
    
    # 3. Feature Generation
    # Ensure 'odds_fluctuation' runs to get 'odds_10min'
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # We might need to clear cache for this day to ensure fresh odds if running "Real-time"?
    # For now assume cache handles it or timestamp check.
    
    # We need to process THE WHOLE loaded chunk to get history features correctly?
    # Yes.
    
    logger.info("Generating Features (this might take a moment)...")
    blocks = list(pipeline.registry.keys())
    df_features = pipeline.load_features(df_raw, blocks, force=args.force)
    
    # Merge back core info from df_raw that might have been dropped by pipeline
    # (Pipeline only preserves race_id, horse_number, horse_id)
    core_cols = ['race_id', 'horse_number', 'date', 'odds', 'popularity', 'horse_name', 'start_time_str']
    available_core = [c for c in core_cols if c in df_raw.columns]
    df_features = pd.merge(df_features, df_raw[available_core].drop_duplicates(['race_id', 'horse_number']), 
                           on=['race_id', 'horse_number'], how='left', suffixes=('', '_raw'))
    
    # Resolve conflicting columns if any (prefer raw for ground truth keys like date)
    if 'date_raw' in df_features.columns:
        df_features['date'] = df_features['date_raw']
        df_features.drop(columns=['date_raw'], inplace=True)

    # Filter target again
    df_features['date'] = pd.to_datetime(df_features['date'])
    df_today = df_features[df_features['date'] == pd.to_datetime(target_date)].copy()
    
    # Row count validation
    n_raw = len(df_target)
    n_feat = len(df_today)
    if n_raw != n_feat:
        logger.warning(f"Row count mismatch! Raw: {n_raw}, Features: {n_feat}. Some horses might have been dropped during processing.")
    else:
        logger.info(f"Row count consistent: {n_feat} horses.")
    
    # 4. Check Odds Availability
    if 'odds_10min' not in df_today.columns or df_today['odds_10min'].isna().all():
        logger.warning("odds_10min column missing or empty! Attempting manual computation...")
        from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
        
        # Ensure start_time_str fallback if needed (though df_raw usually has it from loader)
        if 'start_time_str' not in df_today.columns and 'start_time_str' in df_raw.columns:
             # Merge it from raw
             df_today = pd.merge(df_today, df_raw[['race_id', 'horse_number', 'start_time_str']], on=['race_id', 'horse_number'], how='left')
             
        df_odds = compute_odds_fluctuation(df_today)
        
        if not df_odds.empty:
            logger.info(f"Computed {len(df_odds)} odds records. Merging...")
            # Drop old columns
            for c in ['odds_10min', 'odds_final', 'odds_60min', 'odds_ratio_10min', 'rank_diff_10min', 'odds_log_ratio_10min', 'odds_ratio_60_10']:
                if c in df_today.columns:
                    df_today = df_today.drop(columns=[c])
            
            # Key types
            df_today['race_id'] = df_today['race_id'].astype(str)
            df_today['horse_number'] = df_today['horse_number'].astype(int)
            df_odds['race_id'] = df_odds['race_id'].astype(str)
            df_odds['horse_number'] = df_odds['horse_number'].astype(int)
            
            df_odds = df_odds.drop_duplicates(subset=['race_id', 'horse_number'])
            df_today = pd.merge(df_today, df_odds.drop(columns=['horse_id'], errors='ignore'), 
                          on=['race_id', 'horse_number'], how='left')
        else:
            logger.warning("Manual odds computation failed. Falling back to current 'odds' as 'odds_10min'.")
            # If DB (apd_sokuho_o1) is missing data for 2026, fallback to existing 'odds' column
            if 'odds' in df_today.columns:
                df_today['odds_10min'] = df_today['odds']
                if 'odds_final' not in df_today.columns: df_today['odds_final'] = df_today['odds']
                # Ratio features default to 1.0 (no change) or 0.0
                if 'odds_ratio_10min' not in df_today.columns: df_today['odds_ratio_10min'] = 1.0
                if 'odds_ratio_60_10' not in df_today.columns: df_today['odds_ratio_60_10'] = 1.0
                if 'rank_diff_10min' not in df_today.columns: df_today['rank_diff_10min'] = 0.0
                if 'odds_log_ratio_10min' not in df_today.columns: df_today['odds_log_ratio_10min'] = 0.0
            else:
                logger.warning("No odds available in 'odds' column. Trying jvd_o1...")
                # Last resort: Try jvd_o1 (Standard Odds string)
                try:
                    unique_years = df_today['race_id'].str[:4].unique()
                    year_str = ",".join([f"'{y}'" for y in unique_years])
                    q_o1 = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen IN ({year_str})"
                    df_o1_raw = pd.read_sql(q_o1, loader.engine)
                    print(f"DEBUG: jvd_o1 Rows={len(df_o1_raw)}", flush=True)
                    if not df_o1_raw.empty:
                        def build_rid(row):
                            try:
                                nen = str(int(float(row['kaisai_nen'])))
                                place = str(int(float(row['keibajo_code']))).zfill(2)
                                kai = str(int(float(row['kaisai_kai']))).zfill(2)
                                nichi = str(int(float(row['kaisai_nichime']))).zfill(2)
                                race = str(int(float(row['race_bango']))).zfill(2)
                                return f"{nen}{place}{kai}{nichi}{race}"
                            except: return None
                        
                        df_o1_raw['race_id'] = df_o1_raw.apply(build_rid, axis=1)
                        target_rids = df_today['race_id'].unique()
                        
                        print(f"DEBUG: Target RID Count={len(target_rids)}. Sample={target_rids[0]}", flush=True)
                        print(f"DEBUG: Constructed RID Sample={df_o1_raw['race_id'].iloc[0]}", flush=True)
                        
                        df_o1_raw = df_o1_raw[df_o1_raw['race_id'].isin(target_rids)].copy()
                        print(f"DEBUG: Matched Rows={len(df_o1_raw)}", flush=True)
                        
                        parsed = []
                        for _, row in df_o1_raw.iterrows():
                            s = row['odds_tansho']
                            rid = row['race_id']
                            if not isinstance(s, str): continue
                            for i in range(0, len(s), 8):
                                chunk = s[i:i+8]
                                if len(chunk) < 8: break
                                try:
                                    parsed.append({
                                        'race_id': rid, 'horse_number': int(chunk[0:2]),
                                        'odds_10min': int(chunk[2:6]) / 10.0,
                                        'popularity': int(chunk[6:8])
                                    })
                                except: continue
                        if parsed:
                            df_o1_parsed = pd.DataFrame(parsed)
                            print(f"DEBUG: Parsed Records={len(df_o1_parsed)}. First={df_o1_parsed.iloc[0].to_dict()}", flush=True)
                            # Update df_today
                            df_today = df_today.drop(columns=['odds_10min', 'popularity'], errors='ignore')
                            df_today = pd.merge(df_today, df_o1_parsed, on=['race_id', 'horse_number'], how='left')
                            print(f"DEBUG: df_today Rows after merge={len(df_today)}. Non-zero odds count={(df_today['odds_10min'] > 0).sum()}", flush=True)
                except Exception as e:
                    print(f"DEBUG: EXCEPTION in o1 fallback: {e}", flush=True)
                    return

    # Check again
    n_valid_odds = df_today['odds_10min'].notna().sum()
    logger.info(f"Valid Odds (10min): {n_valid_odds} / {len(df_today)}")
    
    if n_valid_odds == 0:
        logger.warning("No valid odds found. Cannot predict.")
        return
        
    # Feature Engineering (Derived)
    # Model expects 'odds_rank_10min'. Compute it from 'odds_10min'.
    if 'odds_rank_10min' not in df_today.columns and 'odds_10min' in df_today.columns:
        logger.info("Computing derived feature: odds_rank_10min")
        df_today['odds_rank_10min'] = df_today.groupby('race_id')['odds_10min'].rank(method='min')
    
    # 5. Prediction
    logger.info("Running Prediction...")
    
    # Fill missing features with 0 or mean? Model handles NaNs? LightGBM handles NaNs.
    X_pred = df_today[features_list]
    
    preds = model.predict(X_pred)
    df_today['pred_gap'] = preds
    # Use method='first' to ensure exactly 1, 2, 3, 4, 5 without gaps or ties in display
    df_today['pred_rank'] = df_today.groupby('race_id')['pred_gap'].rank(ascending=False, method='first')
    
    # 6. Display Output
    # Need to join with Horse Name / Race Name if available
    # df_raw usually has horse_name? Loader might need adjustment or join jvd_um.
    # Loader columns: ['race_id', 'horse_number', 'horse_id']
    # If horse_name missing, we just show numbers.
    # Usually we want Horse Name.
    # Let's hope df_features preserved it or we merge back.
    if 'horse_name' not in df_today.columns and 'horse_name' in df_raw.columns:
         df_today = pd.merge(df_today, df_raw[['race_id', 'horse_number', 'horse_name']], on=['race_id', 'horse_number'], how='left')
    
    print("\n" + "="*60)
    print(f"  PREDICTION REPORT [V14 Production] for {target_date}")
    print("="*60)
    
    race_ids = sorted(df_today['race_id'].unique())
    for rid in race_ids:
        race_df = df_today[df_today['race_id'] == rid].sort_values('pred_rank')
        
        place_code = rid[4:6]
        race_no = rid[10:12]
        place_map = {'01':'札幌','02':'函館','03':'福島','04':'新潟',
                     '05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
        place_name = place_map.get(place_code, f"場所{place_code}")
        
        print(f"\n[{place_name} {race_no}R] (ID: {rid})")
        # Align headers: 馬番(4), 馬名(20), 人気(6), オッズ(8), Gap位(8), アクション
        header = f"{pad_japanese('馬番', 4)} {pad_japanese('馬名', 20)} {pad_japanese('人気', 6)} {pad_japanese('オッズ', 8)} {pad_japanese('Gap位', 8)}  {'アクション'}"
        print(header)
        print("-" * get_visual_width(header))
        
        for _, row in race_df.iterrows():
            gap = row['pred_rank'] 
            if gap > 5: continue
            
            h_no = f"{int(row['horse_number']):02}"
            h_name = row.get('horse_name', 'Unknown')
            odds = row.get('odds_10min', 0.0)
            ninki = row.get('popularity', 0)
            
            # Action Logic
            action = ""
            if gap <= 3:
                if 10.0 <= odds <= 20.0:
                    action = "🔥[PLACE BET]"
                elif 20.0 < odds <= 50.0:
                    action = "💰[HIGH VALUE]"
                elif odds < 10.0:
                    action = "⚠️[Low Odds]"
                else:
                    action = "★[Hole]"
            
            # Format row with precise visual padding
            row_str = (
                f"{pad_japanese(h_no, 4)} "
                f"{pad_japanese(h_name, 20)} "
                f"{pad_japanese(int(ninki), 6)} "
                f"{pad_japanese(f'{odds:>6.1f}', 8)} "
                f"{pad_japanese(int(gap), 8)}  "
                f"{action}"
            )
            print(row_str)

if __name__ == "__main__":
    main()
