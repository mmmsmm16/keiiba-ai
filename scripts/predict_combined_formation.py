
"""
Production Combined Prediction Script (V13 Mainstream + V14 Gap)
================================================================
Targets: Combined Formation Analysis
- V13: Predicts pure win/place probability (Axis)
- V14: Predicts undervalued Gap score (Pocket/穴)
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

# Force UTF-8 Output for Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline
import requests
import time
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv()

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
        if current_chunk:
            chunks.append(current_chunk)
            
        for chunk in chunks:
            data = {"content": f"```\n{chunk}\n```"}
            for attempt in range(3):
                response = requests.post(webhook_url, json=data)
                if response.status_code == 204:
                    time.sleep(0.5)
                    break
                elif response.status_code == 429:
                    time.sleep(1.5)
                    continue
                else:
                    break
    except Exception as e:
        print(f"Error sending Discord notification: {e}")

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

def normalize_race_id(raw_race_id):
    if raw_race_id is None:
        return None
    race_id = "".join(ch for ch in str(raw_race_id).strip() if ch.isdigit())
    if len(race_id) != 12:
        return None
    return race_id


def build_v13_features(df, feats_v13, logger, fill_value=0.0):
    missing = [c for c in feats_v13 if c not in df.columns]
    if missing:
        sample = ", ".join(missing[:10])
        logger.warning(f"V13 missing features: {len(missing)} (filled with {fill_value}). sample={sample}")
    X = df.reindex(columns=feats_v13, fill_value=fill_value)
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X.fillna(fill_value).astype(float)

def load_v13_feature_list(path, logger):
    if path.endswith('.joblib'):
        return joblib.load(path)
    try:
        feats_df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load V13 feature list: {e}")
        return []
    if 'feature' in feats_df.columns:
        return feats_df['feature'].tolist()
    if '0' in feats_df.columns:
        return feats_df['0'].tolist()
    return feats_df.iloc[:, 0].tolist()

def load_v13_prediction_type(model_path, logger):
    config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
    if not os.path.exists(config_path):
        return 'sigmoid'
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return config.get('model_params', {}).get('prediction_type', 'sigmoid')
    except Exception as e:
        logger.warning(f"V13 prediction type fallback (sigmoid). reason={e}")
        return 'sigmoid'

def add_v13_odds_features(df, logger):
    if 'odds_10min' in df.columns:
        odds_base = df['odds_10min']
    elif 'odds' in df.columns:
        odds_base = df['odds']
    else:
        logger.warning("V13 odds features: odds column missing. Using neutral defaults.")
        odds_base = pd.Series(np.nan, index=df.index)

    df['odds_calc'] = odds_base.fillna(10.0)
    df['odds_rank'] = df.groupby('race_id')['odds_calc'].rank(ascending=True, method='min')

    if 'relative_horse_elo_z' in df.columns:
        df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False, method='min')
        df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
    else:
        df['odds_rank_vs_elo'] = 0

    df['is_high_odds'] = (df['odds_calc'] >= 10).astype(int)
    df['is_mid_odds'] = ((df['odds_calc'] >= 5) & (df['odds_calc'] < 10)).astype(int)
    if 'odds' in df.columns:
        df['odds'] = df['odds_calc']
    return df

def softmax_per_race(scores, race_ids):
    df = pd.DataFrame({'race_id': race_ids, 'score': scores})
    df['score_shift'] = df.groupby('race_id')['score'].transform(lambda x: x - x.max())
    df['exp'] = np.exp(df['score_shift'])
    df['softmax'] = df.groupby('race_id')['exp'].transform(lambda x: x / x.sum())
    return df['softmax'].values

def merge_odds_fluctuation_features(df, logger):
    try:
        from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
        odds_df = compute_odds_fluctuation(df)
        if odds_df.empty:
            logger.warning("Odds fluctuation features empty. Using fallback odds.")
            return df
        odds_cols = [
            'odds_ratio_10min',
            'odds_ratio_60_10',
            'rank_diff_10min',
            'odds_log_ratio_10min',
            'odds_final',
            'odds_10min',
            'odds_60min'
        ]
        odds_df = odds_df.drop(columns=['horse_id'], errors='ignore')
        rename_map = {c: f"{c}_calc" for c in odds_cols if c in odds_df.columns}
        odds_df = odds_df.rename(columns=rename_map)
        df = pd.merge(df, odds_df, on=['race_id', 'horse_number'], how='left')
        for col in odds_cols:
            calc_col = f"{col}_calc"
            if calc_col not in df.columns:
                continue
            if col in df.columns:
                df[col] = df[col].combine_first(df[calc_col])
            else:
                df[col] = df[calc_col]
            df.drop(columns=[calc_col], inplace=True)
        return df
    except Exception as e:
        logger.warning(f"Odds fluctuation skipped: {e}")
        return df

# Config
MODEL_PROFILES = {
    # Current production models
    "BASE": {
        "v13_model_path": "models/experiments/exp_lambdarank_hard_weighted/model.pkl",
        "v13_feats_path": "models/experiments/exp_lambdarank_hard_weighted/features.csv",
        "v14_model_path": "models/experiments/exp_gap_v14_production/model_v14.pkl",
        "v14_feats_path": "models/experiments/exp_gap_v14_production/features.csv",
        "cache_dir": "data/features_v14/prod_cache",
        "blend": {
            "enabled": False,
            "v13_base_weight": 1.0,
            "v14_base_weight": 1.0,
        },
    },
    # Enhanced models (this workstream)
    "ENHANCED": {
        "v13_model_path": "models/experiments/exp_lambdarank_hard_weighted_enhanced/model.pkl",
        "v13_feats_path": "models/experiments/exp_lambdarank_hard_weighted_enhanced/features.csv",
        "v14_model_path": "models/experiments/exp_gap_v14_production_enhanced/model_v14.pkl",
        "v14_feats_path": "models/experiments/exp_gap_v14_production_enhanced/features.csv",
        "cache_dir": "data/features_v14/prod_cache_enhanced",
        "blend": {
            "enabled": True,
            # Tuned on holdout-style scan: restore hit while keeping ROI edge.
            "v13_base_weight": 0.50,
            "v14_base_weight": 0.30,
            "v13_base_model_path": "models/experiments/exp_lambdarank_hard_weighted/model.pkl",
            "v13_base_feats_path": "models/experiments/exp_lambdarank_hard_weighted/features.csv",
            "v14_base_model_path": "models/experiments/exp_gap_v14_production/model_v14.pkl",
            "v14_base_feats_path": "models/experiments/exp_gap_v14_production/features.csv",
        },
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYYMMDD')
    parser.add_argument('--race_id', type=str, help='Specific Race ID')
    parser.add_argument('--discord', action='store_true', help='Webhook output format')
    parser.add_argument('--force', action='store_true', help='Force recompute features')
    parser.add_argument(
        '--model_profile',
        type=str,
        default='BASE',
        choices=['BASE', 'ENHANCED', 'base', 'enhanced'],
        help='Model profile to use: BASE (current v13/v14) or ENHANCED (new v13/v14).'
    )
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""), help="Discord Webhook URL")
    args = parser.parse_args()
    
    # Capture output for Discord
    output_log = []
    def log_print(msg):
        print(msg)
        output_log.append(msg + "\n")
    
    # 1. Date resolution
    if args.date:
        date_str = args.date
    elif args.race_id:
        # Fallback: assuming YYYYMMDD... but unsafe for standard JRA ID
        date_str = args.race_id[:8]
    else:
        # Default to today
        date_str = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y%m%d")
        
    target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if args.race_id:
        normalized_race_id = normalize_race_id(args.race_id)
        if normalized_race_id is None:
            logger.error(f"Invalid --race_id: {args.race_id}. Expected 12 digits (YYYYVVKKDDRR).")
            return
        if normalized_race_id != args.race_id:
            logger.info(f"Normalized race_id: {args.race_id} -> {normalized_race_id}")
        args.race_id = normalized_race_id

    # 2. Resolve model profile
    profile_key = args.model_profile.upper()
    profile = MODEL_PROFILES.get(profile_key, MODEL_PROFILES["BASE"])
    V13_MODEL_PATH = profile["v13_model_path"]
    V13_FEATS_PATH = profile["v13_feats_path"]
    V14_MODEL_PATH = profile["v14_model_path"]
    V14_FEATS_PATH = profile["v14_feats_path"]
    CACHE_DIR = profile["cache_dir"]
    blend_cfg = profile.get("blend", {}) or {}
    blend_enabled = bool(blend_cfg.get("enabled", False))
    v13_base_weight = float(blend_cfg.get("v13_base_weight", 1.0))
    v14_base_weight = float(blend_cfg.get("v14_base_weight", 1.0))

    logger.info(f"Model profile: {profile_key}")
    logger.info(f"  V13 model: {V13_MODEL_PATH}")
    logger.info(f"  V14 model: {V14_MODEL_PATH}")
    if blend_enabled:
        logger.info(f"  Blend enabled: v13 base_weight={v13_base_weight:.2f}, v14 base_weight={v14_base_weight:.2f}")

    # 3. Load Models
    logger.info("Loading V13 & V14 Models...")
    try:
        model_v13 = joblib.load(V13_MODEL_PATH)
        feats_v13 = load_v13_feature_list(V13_FEATS_PATH, logger)
        v13_prediction_type = load_v13_prediction_type(V13_MODEL_PATH, logger)
        if v13_prediction_type == 'sigmoid' and getattr(model_v13, 'objective_', None) == 'lambdarank':
            v13_prediction_type = 'softmax'

        if not feats_v13:
            logger.error("V13 feature list is empty.")
            return
        
        model_v14 = joblib.load(V14_MODEL_PATH)
        feats_v14 = pd.read_csv(V14_FEATS_PATH)['feature'].tolist()

        model_v13_base = None
        feats_v13_base = None
        model_v14_base = None
        feats_v14_base = None
        if blend_enabled:
            v13_base_model_path = blend_cfg.get("v13_base_model_path")
            v13_base_feats_path = blend_cfg.get("v13_base_feats_path")
            v14_base_model_path = blend_cfg.get("v14_base_model_path")
            v14_base_feats_path = blend_cfg.get("v14_base_feats_path")
            if v13_base_model_path and v13_base_feats_path:
                model_v13_base = joblib.load(v13_base_model_path)
                feats_v13_base = load_v13_feature_list(v13_base_feats_path, logger)
                if not feats_v13_base:
                    logger.warning("V13 base feature list is empty. Disable v13 blending.")
                    model_v13_base = None
                    feats_v13_base = None
            if v14_base_model_path and v14_base_feats_path:
                model_v14_base = joblib.load(v14_base_model_path)
                feats_v14_base = pd.read_csv(v14_base_feats_path)["feature"].tolist()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    # 4. Load Data
    loader = JraVanDataLoader()
    start_history = "2016-01-01"
    
    # [Optimization] Subject Horse Filtering
    # Instead of loading full history, only load history for horses running today + today's full context.
    use_optimized_loading = True
    if args.force:
         use_optimized_loading = False
    optimized_load_failed = False
         
    if use_optimized_loading:
        logger.info(f"Optimized Data Loading Enabled (Target: {target_date}, RaceID: {args.race_id})")
        
        # 3.1 Fetch participating horse IDs
        try:
             # Build query to get horses
             if args.race_id:
                  q_horses = f"""
                  SELECT DISTINCT ketto_toroku_bango
                  FROM jvd_se
                  WHERE CONCAT(
                      LPAD(COALESCE(kaisai_nen::text, ''), 4, '0'),
                      LPAD(COALESCE(keibajo_code::text, ''), 2, '0'),
                      LPAD(COALESCE(kaisai_kai::text, ''), 2, '0'),
                      LPAD(COALESCE(kaisai_nichime::text, ''), 2, '0'),
                      LPAD(COALESCE(race_bango::text, ''), 2, '0')
                  ) = '{args.race_id}'
                  """
             else:
                  # By Date
                  y = target_date.split('-')[0]
                  md = target_date.split('-')[1] + target_date.split('-')[2]
                  q_horses = f"""
                  SELECT DISTINCT res.ketto_toroku_bango 
                  FROM jvd_se res
                  JOIN jvd_ra r ON r.kaisai_nen = res.kaisai_nen 
                    AND r.keibajo_code = res.keibajo_code 
                    AND r.kaisai_kai = res.kaisai_kai 
                    AND r.kaisai_nichime = res.kaisai_nichime 
                    AND r.race_bango = res.race_bango
                  WHERE r.kaisai_nen = '{y}' AND r.kaisai_tsukihi = '{md}'
                  """
             
             if use_optimized_loading:
                 horse_df = pd.read_sql(q_horses, loader.engine)
                 target_horse_ids = horse_df['ketto_toroku_bango'].astype(str).str.strip().tolist()
                 logger.info(f"Identified {len(target_horse_ids)} horses for optimized loading.")
                 
                 if not target_horse_ids:
                     logger.warning("No horses found for target. Falling back to simple JIT load or aborting.")
                     # If no horses, maybe data not arrived yet?
                     # Try to load target date context at least? No main logic requires horses.
                     return 

                 df_raw = loader.load_for_horses(
                     target_horse_ids=target_horse_ids,
                     target_date=target_date,
                     history_start_date=start_history,
                     skip_training=False 
                 )
             else:
                 df_raw = pd.DataFrame() # Fallback trigger
                 
        except Exception as e:
            logger.error(f"Optimized loading failed: {e}. Falling back to full load.")
            optimized_load_failed = True
            use_optimized_loading = False

    if args.race_id and optimized_load_failed and not args.force:
        logger.error("Aborting in --race_id mode to avoid expensive full-history fallback. Use --force only for emergency runs.")
        return
    
    if not use_optimized_loading:
        df_raw = loader.load(history_start_date=start_history, end_date=target_date, skip_odds=False)
    
    if df_raw.empty:
        logger.warning("No data found.")
        return

    numeric_cols = ['time', 'last_3f', 'rank', 'weight', 'weight_diff', 'impost', 'honshokin', 'fukashokin']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    if 'time' in df_raw.columns:
        min_times = df_raw.groupby('race_id')['time'].transform('min')
        df_raw['time_diff'] = (df_raw['time'] - min_times).fillna(0)
    else:
        df_raw['time_diff'] = 0

    # Ensure keys are correct types for pipeline
    df_raw['race_id'] = df_raw['race_id'].astype(str)
    df_raw['horse_number'] = pd.to_numeric(df_raw['horse_number'], errors='coerce').fillna(0).astype(int)
    
    # 4. Feature Generation
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    blocks = list(pipeline.registry.keys())
    
    # [Fix] In JIT (Optimized) mode, we load a partial dataset including new (today's) rows.
    # Existing caches likely miss these rows. We must Force recalculation to ensure features exist.
    # Since the dataset is small (JIT), recalculation is fast.
    force_calc = args.force or use_optimized_loading
    
    df_features = pipeline.load_features(df_raw, blocks, force=force_calc)

    # Merge core info
    core_cols = ['race_id', 'horse_number', 'date', 'odds', 'odds_10min', 'popularity', 'horse_name', 'start_time_str']
    available_core = [c for c in core_cols if c in df_raw.columns]
    df_merged = pd.merge(df_features, df_raw[available_core].drop_duplicates(['race_id', 'horse_number']), 
                         on=['race_id', 'horse_number'], how='left')

    # Filter target
    if 'date' in df_merged.columns:
        df_merged['date'] = pd.to_datetime(df_merged['date'])
        df_today = df_merged[df_merged['date'] == pd.to_datetime(target_date)].copy()
    else:
        # Fallback to df_raw's date if merge somehow failed to include it
        df_raw['date_dt'] = pd.to_datetime(df_raw['date'])
        target_rids = df_raw[df_raw['date_dt'] == pd.to_datetime(target_date)]['race_id'].unique()
        df_today = df_merged[df_merged['race_id'].isin(target_rids)].copy()
        df_today['date'] = pd.to_datetime(target_date)
    if args.race_id:
        df_today = df_today[df_today['race_id'] == args.race_id].copy()

    if df_today.empty:
        logger.warning(f"Target subset is empty for date {target_date} and race_id {args.race_id}")
        if not df_merged.empty:
            logger.info(f"Available Race IDs in merged data: {df_merged['race_id'].unique()[:5]}")
            if 'date' in df_merged.columns:
                logger.info(f"Available Dates in merged data: {df_merged['date'].dt.date.unique()[:5]}")
        return

    df_today = merge_odds_fluctuation_features(df_today, logger)

    if 'odds' in df_today.columns:
        df_today['odds'] = pd.to_numeric(df_today['odds'], errors='coerce')
        df_today.loc[df_today['odds'] <= 0, 'odds'] = np.nan

    # [Fix] Ensure odds_10min is numeric
    df_today['odds_10min'] = pd.to_numeric(df_today.get('odds_10min', np.nan), errors='coerce')
    df_today.loc[df_today['odds_10min'] <= 0, 'odds_10min'] = np.nan
    invalid_odds_mask = df_today['odds_10min'].isna() | (df_today['odds_10min'] <= 0)

    # Check if odds_10min exists (or is empty) and try to fill from JVD_O1
    if invalid_odds_mask.any():
        try:
            year_str = f"'{date_str[:4]}'"
            q_o1 = (
                "SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho "
                f"FROM jvd_o1 WHERE kaisai_nen = {year_str} AND kaisai_tsukihi = '{date_str[4:]}'"
            )
            if args.race_id:
                q_o1 += (
                    " AND CONCAT("
                    "LPAD(COALESCE(kaisai_nen::text, ''), 4, '0'),"
                    "LPAD(COALESCE(keibajo_code::text, ''), 2, '0'),"
                    "LPAD(COALESCE(kaisai_kai::text, ''), 2, '0'),"
                    "LPAD(COALESCE(kaisai_nichime::text, ''), 2, '0'),"
                    "LPAD(COALESCE(race_bango::text, ''), 2, '0')"
                    f") = '{args.race_id}'"
                )
            df_o1_raw = pd.read_sql(q_o1, loader.engine)
            if not df_o1_raw.empty:
                def build_rid(row):
                    try:
                        return f"{int(float(row['kaisai_nen']))}{int(float(row['keibajo_code'])):02}{int(float(row['kaisai_kai'])):02}{int(float(row['kaisai_nichime'])):02}{int(float(row['race_bango'])):02}"
                    except: return None
                df_o1_raw['race_id'] = df_o1_raw.apply(build_rid, axis=1)
                parsed = []
                for _, row in df_o1_raw.iterrows():
                    s, rid = row['odds_tansho'], row['race_id']
                    if not isinstance(s, str): continue
                    for i in range(0, len(s), 8):
                        chunk = s[i:i+8]
                        if len(chunk) < 8: break
                        try:
                            parsed.append({
                                'race_id': rid, 
                                'horse_number': int(chunk[0:2]),
                                'odds_10min_o1': int(chunk[2:6]) / 10.0,
                                'popularity_10min_o1': int(chunk[6:8])
                            })
                        except: continue
                if parsed:
                    df_o1_parsed = pd.DataFrame(parsed)
                    # Align types for merge
                    df_today['race_id'] = df_today['race_id'].astype(str)
                    df_today['horse_number'] = df_today['horse_number'].astype(int)
                    df_o1_parsed['race_id'] = df_o1_parsed['race_id'].astype(str)
                    df_o1_parsed['horse_number'] = df_o1_parsed['horse_number'].astype(int)
                    
                    df_today = pd.merge(df_today, df_o1_parsed, on=['race_id', 'horse_number'], how='left')
                    # Robust update using combine_first or explicit loc after re-masking
                    df_today['odds_10min'] = df_today['odds_10min'].fillna(df_today['odds_10min_o1'])
                    df_today['popularity_10min'] = df_today.get('popularity_10min', pd.Series(np.nan, index=df_today.index)).fillna(df_today['popularity_10min_o1'])
                    df_today = df_today.drop(columns=['odds_10min_o1', 'popularity_10min_o1'], errors='ignore')
        except Exception as e:
            logger.warning(f"Fallback jvd_o1 failed: {e}")

    invalid_odds_mask = df_today['odds_10min'].isna() | (df_today['odds_10min'] <= 0)
    if invalid_odds_mask.any():
        if 'odds' in df_today.columns:
            df_today.loc[invalid_odds_mask, 'odds_10min'] = df_today.loc[invalid_odds_mask, 'odds']

    invalid_odds_mask = df_today['odds_10min'].isna() | (df_today['odds_10min'] <= 0)
    if invalid_odds_mask.any():
        logger.warning(f"Odds fallback unresolved for {int(invalid_odds_mask.sum())} rows. Filling odds_10min=10.0.")
        df_today.loc[invalid_odds_mask, 'odds_10min'] = 10.0

    if 'odds' not in df_today.columns:
        df_today['odds'] = df_today['odds_10min']
    else:
        df_today['odds'] = pd.to_numeric(df_today['odds'], errors='coerce').fillna(df_today['odds_10min'])

    if 'odds_final' not in df_today.columns:
        df_today['odds_final'] = df_today['odds']
    df_today['odds_final'] = df_today['odds_final'].fillna(df_today['odds_10min'])

    if 'odds_ratio_10min' not in df_today.columns or df_today['odds_ratio_10min'].isna().all():
        df_today['odds_ratio_10min'] = df_today['odds_final'] / df_today['odds_10min'].replace(0, np.nan)

    if 'odds_60min' not in df_today.columns:
        df_today['odds_60min'] = df_today['odds_10min']
    if 'odds_ratio_60_10' not in df_today.columns or df_today['odds_ratio_60_10'].isna().all():
        df_today['odds_ratio_60_10'] = df_today['odds_10min'] / df_today['odds_60min'].replace(0, np.nan)

    if 'odds_log_ratio_10min' not in df_today.columns or df_today['odds_log_ratio_10min'].isna().all():
        df_today['odds_log_ratio_10min'] = np.log(df_today['odds_final'] + 1e-9) - np.log(df_today['odds_10min'] + 1e-9)

    df_today['odds_60min'] = df_today['odds_60min'].fillna(df_today['odds_10min'])
    df_today['odds_ratio_10min'] = df_today['odds_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df_today['odds_ratio_60_10'] = df_today['odds_ratio_60_10'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df_today['odds_log_ratio_10min'] = df_today['odds_log_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Derived Features for V14 (and potentially used by V13 if shared)
    df_today['odds_rank_10min'] = df_today.groupby('race_id')['odds_10min'].rank(method='min')
    if 'popularity' not in df_today.columns or df_today['popularity'].isna().all():
        df_today['popularity'] = df_today['odds_rank_10min']
    df_today['rank_diff_10min'] = df_today['popularity'] - df_today['odds_rank_10min']

    if 'field_size' not in df_today.columns:
        df_today['field_size'] = df_today.groupby('race_id')['horse_number'].transform('count')

    # 4.5 V13 Feature Preparation (after odds fill)
    logger.info("Preparing V13 features (using standard pipeline)...")
    df_v13_today = add_v13_odds_features(df_today.copy(), logger)

    # 6. Predictions
    logger.info("Running Combined Predictions...")
    
    # Debug: Check feature values
    logger.info("--- DEBUG: Feature Stats ---")
    logger.info(f"df_today shape: {df_today.shape}")
    
    # [Optimization] Detailed feature log per horse
    if not df_today.empty:
        # Columns to show
        show_cols = [
            'horse_number', 'horse_name', 'run_count', 'interval', 
            'avg_speed_index', 'horse_elo', 'jockey_win_rate_365d',
            'odds_10min'
        ]
        # Filter existing columns
        show_cols = [c for c in show_cols if c in df_today.columns]
        
        logger.info(f"--- Target Race Features (Detailed) ---")
        # Sort by horse number
        if 'horse_number' in df_today.columns:
            df_log = df_today.sort_values('horse_number')
        else:
            df_log = df_today
        logger.info("\n" + df_log[show_cols].to_string(index=False))
        
        # [Validation] Check for completely empty features
        nan_cols = [c for c in df_today.columns if df_today[c].isna().all()]
        if nan_cols:
            logger.warning(f"!!! Completely Missing (All-NaN) Features Count: {len(nan_cols)} !!!")
            logger.warning(f"Missing Cols: {nan_cols}")
        else:
            logger.info(">>> ALL FEATURES POPULATED (No 100% NaN columns) <<<")
        
    # V13
    # Check specific columns existence before building
    missing_v13 = [c for c in feats_v13 if c not in df_v13_today.columns]
    if missing_v13:
        logger.warning(f"V13 Missing Columns Count: {len(missing_v13)}")
        
    X_v13 = build_v13_features(df_v13_today, feats_v13, logger, fill_value=np.nan)
    
    # Debug X_v13 content
    logger.info(f"X_v13 shape: {X_v13.shape}")
    if 'odds' in X_v13.columns:
         logger.info(f"X_v13['odds'] Stats:\n{X_v13['odds'].describe()}")
    elif 'odds_10min' in X_v13.columns:
         logger.info(f"X_v13['odds_10min'] Stats:\n{X_v13['odds_10min'].describe()}")
    
    if hasattr(model_v13, 'predict_proba'):
        v13_scores = model_v13.predict_proba(X_v13)
        if isinstance(v13_scores, np.ndarray) and v13_scores.ndim == 2:
            v13_scores = v13_scores[:, -1]
    else:
        v13_scores = model_v13.predict(X_v13)

    if blend_enabled and model_v13_base is not None and feats_v13_base is not None:
        X_v13_base = build_v13_features(df_v13_today, feats_v13_base, logger, fill_value=np.nan)
        if hasattr(model_v13_base, 'predict_proba'):
            v13_base_scores = model_v13_base.predict_proba(X_v13_base)
            if isinstance(v13_base_scores, np.ndarray) and v13_base_scores.ndim == 2:
                v13_base_scores = v13_base_scores[:, -1]
        else:
            v13_base_scores = model_v13_base.predict(X_v13_base)
        v13_scores = (v13_base_weight * v13_base_scores) + ((1.0 - v13_base_weight) * v13_scores)
        logger.info("Applied V13 score blending (BASE+ENHANCED).")
        
    logger.info(f"V13 Raw Scores Stats: min={v13_scores.min():.4f}, max={v13_scores.max():.4f}, mean={v13_scores.mean():.4f}, std={v13_scores.std():.4f}")

    if v13_prediction_type == 'softmax':
        v13_probs = softmax_per_race(v13_scores, df_v13_today['race_id'].values)
    elif v13_prediction_type == 'sigmoid':
        v13_probs = 1 / (1 + np.exp(-v13_scores))
        v13_probs = (
            pd.Series(v13_probs, index=df_v13_today.index)
            .groupby(df_v13_today['race_id'])
            .transform(lambda x: x / x.sum())
            .values
        )
    else:
        v13_probs = v13_scores
    df_v13_today['prob_v13'] = v13_probs
    df_v13_today['score_v13'] = v13_scores
    df_today = pd.merge(
        df_today,
        df_v13_today[['race_id', 'horse_number', 'prob_v13', 'score_v13']],
        on=['race_id', 'horse_number'],
        how='left'
    )
    df_today['prob_v13'] = df_today['prob_v13'].fillna(0.0)
    df_today['score_v13'] = df_today['score_v13'].fillna(-99.0)
    # V14
    # Ensure critical features for V14
    if 'odds_final' not in df_today.columns and 'odds' in df_today.columns:
        df_today['odds_final'] = df_today['odds']
    
    missing_v14 = [c for c in feats_v14 if c not in df_today.columns]
    if missing_v14:
        logger.warning(f"V14 Missing Columns Count: {len(missing_v14)}. Filling 0.")
        for c in missing_v14:
            df_today[c] = 0.0
            
    X_v14 = df_today[feats_v14].fillna(0)
    
    # Debug X_v14
    logger.info(f"X_v14 shape: {X_v14.shape}")
    if not X_v14.empty:
        logger.info(f"X_v14 Sample Stats (Top 5 feats):\n{X_v14.iloc[:, :5].describe()}")
    
    v14_scores = model_v14.predict(X_v14)
    if blend_enabled and model_v14_base is not None and feats_v14_base is not None:
        for c in feats_v14_base:
            if c not in df_today.columns:
                df_today[c] = 0.0
        X_v14_base = df_today[feats_v14_base].fillna(0)
        v14_base_scores = model_v14_base.predict(X_v14_base)
        v14_scores = (v14_base_weight * v14_base_scores) + ((1.0 - v14_base_weight) * v14_scores)
        logger.info("Applied V14 score blending (BASE+ENHANCED).")
    df_today['gap_v14'] = v14_scores

    # Ranks
    df_today['rank_v13'] = df_today.groupby('race_id')['prob_v13'].rank(ascending=False, method='first')
    df_today['rank_v14'] = df_today.groupby('race_id')['gap_v14'].rank(ascending=False, method='first')

    # 7. Print Report
    log_print("\n" + "="*80)
    log_print(f"  COMBINED PREDICTION REPORT (V13 Axis x V14 Gap) for {target_date}")
    log_print("="*80)

    # Enhance Metadata with Start Time
    if 'hasso_jikoku' not in df_today.columns:
        # Try to fetch start time if missing (simplified approach: already in df_raw?)
        # hasso_jikoku is usually in jvd_ra. Loader loads jvd_ra?
        pass # Assumed handled or missing is fine
        
    race_ids = sorted(df_today['race_id'].unique())
    # Sort by race_id or start time if available
    # Reuse sort logic if hasso_jikoku is present
    if 'start_time_str' in df_today.columns:
         # Create temp dedicated DF for sorting
         r_sort = df_today[['race_id', 'start_time_str']].drop_duplicates().sort_values(['start_time_str', 'race_id'])
         race_ids = r_sort['race_id'].tolist()

    for rid in race_ids:
        race_df = df_today[df_today['race_id'] == rid]
        
        place_code = rid[4:6]
        race_no = rid[10:12]
        place_map = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
        place_name = place_map.get(place_code, f"場所{place_code}")
        
        # Start Time formatting
        st_str = ""
        if 'start_time_str' in race_df.columns:
            st_raw = race_df['start_time_str'].iloc[0]
            if st_raw and len(str(st_raw)) >= 4:
                st_str = f"({str(st_raw)[:2]}:{str(st_raw)[2:]}発走)"
        
        log_print(f"\n[{place_name} {race_no}R] {st_str} (ID: {rid})")
        
        # 1. V13 Tops
        log_print(f"--- V13 (本命モデル) TOP ---")
        v13_tops = race_df.sort_values('rank_v13').head(3)
        for _, row in v13_tops.iterrows():
            log_print(f"#{int(row['rank_v13'])}: [{int(row['horse_number']):02}] {pad_japanese(row['horse_name'], 16)} (Prob: {row['prob_v13']:.3f}, Score: {row['score_v13']:>6.2f})")
            
        # 2. V14 Tops
        log_print(f"\n--- V14 (穴モデル) TOP ---")
        v14_tops = race_df.sort_values('rank_v14').head(5)
        for _, row in v14_tops.iterrows():
            log_print(f"#{int(row['rank_v14'])}: [{int(row['horse_number']):02}] {pad_japanese(row['horse_name'], 16)} (Odds: {row['odds_10min']:>5.1f}, Gap: {row['gap_v14']:.3f})")

        # 3. Recommended Formation
        log_print(f"\n--- 推奨買い目 [フォーメーション] ---")
        axis_row = race_df[race_df['rank_v13'] == 1].iloc[0]
        partners_top5 = race_df[race_df['rank_v14'] <= 5].sort_values('rank_v14')
        
        axis_no = int(axis_row['horse_number'])
        partner_nos = [int(p['horse_number']) for _, p in partners_top5.iterrows() if int(p['horse_number']) != axis_no]
        
        if partner_nos:
            partner_str = ",".join([f"{n:02}" for n in partner_nos])
            log_print(f"【軸】 {axis_no:02} ({axis_row['horse_name']})")
            log_print(f"【相手】 {partner_str}")
            log_print(f"【馬連】 {axis_no:02} - ({partner_str})  [{len(partner_nos)}点]")
            log_print(f"【ワイド】 {axis_no:02} - ({partner_str})  [{len(partner_nos)}点]")
        else:
            log_print("推奨買い目なし (軸と相手が重複)")
        log_print("-" * 50)
        
    if args.discord:
        logger.info("Sending Discord Notification...")
        msg = "".join(output_log)
        send_discord_notification(args.webhook_url, msg)

if __name__ == "__main__":
    main()
