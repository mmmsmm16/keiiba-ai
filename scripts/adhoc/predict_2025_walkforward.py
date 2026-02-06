
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml

sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline
from models.calibration import ProbabilityCalibrator  # For loading calibrator pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Generating T2 Refined v3 [Walk-Forward 2025] Predictions...")
    
    MODEL_PATH = "models/experiments/exp_t2_refined_v3_2025/model.pkl"
    OUTPUT_PATH = "data/temp_t2/T2_predictions_2025_walkforward.parquet"
    CONFIG_PATH = "config/experiments/exp_t2_refined_v3_2025.yaml"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
        
    # Load model
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    
    # Load config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    feature_blocks = cfg['features']
    
    # Check if features cached
    cache_file = "data/features/temp_merge_current.parquet"
    # Even if we use cached features, we filter for 2025
    if os.path.exists(cache_file):
        logger.info(f"Loading features from cache: {cache_file}")
        df_feat = pd.read_parquet(cache_file)
        
        # Ensure 2025 data is present
        df_feat['race_id'] = df_feat['race_id'].astype(str)
        df_2025 = df_feat[df_feat['race_id'].str.startswith('2025')]
        
        if df_2025.empty:
             logger.warning("Cache exists but no 2025 data found. Generating from scratch...")
             # Fallback
             start_date = cfg['dataset'].get('train_start_date', '2014-01-01')
             loader = JraVanDataLoader()
             df_raw = loader.load(history_start_date=start_date, end_date='2025-12-31', skip_training=False, skip_odds=True)
             pipeline = FeaturePipeline(cache_dir="data/features")
             df_feat = pipeline.load_features(df_raw, feature_blocks)
        else:
             logger.info(f"Loaded {len(df_feat)} rows from cache. 2025 count: {len(df_2025)}")
    else:
        logger.info("Generating features from scratch...")
        start_date = cfg['dataset'].get('train_start_date', '2014-01-01')
        loader = JraVanDataLoader()
        df_raw = loader.load(history_start_date=start_date, end_date='2025-12-31', skip_training=False, skip_odds=True)
        pipeline = FeaturePipeline(cache_dir="data/features")
        df_feat = pipeline.load_features(df_raw, feature_blocks)
    
    # Filter for 2025 ONLY
    logger.info("Filtering for 2025 data...")
    df_feat['race_id'] = df_feat['race_id'].astype(str)
    df_feat = df_feat[df_feat['race_id'].str.startswith('2025')]
    
    # Load raw for missing columns
    cols_needed = ['date', 'rank', 'tansho_odds']
    missing = [c for c in cols_needed if c not in df_feat.columns]
    
    if missing:
         logger.info(f"Missing columns {missing} in features. Loading raw 2025 data to merge...")
         loader = JraVanDataLoader()
         df_raw_25 = loader.load(history_start_date='2025-01-01', end_date='2025-12-31', skip_training=False, skip_odds=False)
         df_raw_25['race_id'] = df_raw_25['race_id'].astype(str)
         
         cols_to_use = ['race_id', 'horse_number']
         for c in missing:
             if c in df_raw_25.columns:
                 cols_to_use.append(c)
             elif c == 'tansho_odds' and 'odds' in df_raw_25.columns:
                 df_raw_25['tansho_odds'] = df_raw_25['odds']
                 cols_to_use.append(c)
         
         df_feat = pd.merge(df_feat, df_raw_25[cols_to_use], on=['race_id', 'horse_number'], how='left')
         
    # Prepare X
    if hasattr(model, 'booster_'):
        booster = model.booster_
    else:
        booster = model
        
    model_features = booster.feature_name()
    
    # Check keys
    for c in model_features:
        if c not in df_feat.columns:
            df_feat[c] = 0
            
    X = df_feat[model_features].copy()
    
    # Categorical Encoding
    dump = booster.dump_model()
    if 'pandas_categorical' in dump and 'categorical_feature' in dump:
        cat_indices = dump['categorical_feature']
        cat_values_list = dump['pandas_categorical']
        
        for i, idx in enumerate(cat_indices):
            col_name = model_features[idx]
            cats = cat_values_list[i]
            if not cats: continue
            cat_map = {val: code for code, val in enumerate(cats)}
            
            if isinstance(cats[0], str):
                X[col_name] = X[col_name].astype(str).map(cat_map)
            else:
                X[col_name] = X[col_name].map(cat_map)

    # Sanitize
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
    # Predict
    logger.info("Generating predictions...")
    if hasattr(model, 'predict_proba'):
        preds_raw = model.predict_proba(X.values.astype(np.float32))[:, 1]
    else:
        preds_raw = model.predict(X.values.astype(np.float32))
    
    # Apply calibration using 2024 model's calibrator (same distribution mapping)
    CALIBRATOR_PATH = "models/experiments/exp_t2_refined_v3/calibrator.pkl"
    if os.path.exists(CALIBRATOR_PATH):
        logger.info(f"Loading 2024 calibrator from {CALIBRATOR_PATH}")
        calibrator = joblib.load(CALIBRATOR_PATH)
        preds = calibrator.transform(preds_raw)  # Use transform method
        logger.info("✅ Calibration applied (using 2024 calibrator)!")
    else:
        logger.warning("⚠️ Calibrator not found. Using raw predictions.")
        preds = preds_raw

        
    df_feat['pred_prob'] = preds
    
    # Rename tansho_odds to odds_final
    if 'tansho_odds' in df_feat.columns:
        df_feat['odds_final'] = pd.to_numeric(df_feat['tansho_odds'], errors='coerce').fillna(1.0)
    elif 'odds' in df_feat.columns:
        df_feat['odds_final'] = pd.to_numeric(df_feat['odds'], errors='coerce').fillna(1.0)
    else:
        df_feat['odds_final'] = 1.0
        
    if 'rank' in df_feat.columns:
        df_feat['is_win'] = (df_feat['rank'] == 1).astype(int)
    else:
        df_feat['is_win'] = 0
        
    output_cols = ['race_id', 'horse_number', 'date', 'rank', 'pred_prob', 'odds_final', 'is_win']
    for c in output_cols:
        if c not in df_feat.columns:
             df_feat[c] = 0
             
    logger.info(f"Saving to {OUTPUT_PATH}")
    df_feat[output_cols].to_parquet(OUTPUT_PATH, index=False)
    
    # Distribution Check
    logger.info("--- Raw Predictions ---")
    logger.info(f"Mean Prob: {preds_raw.mean():.4f}")
    logger.info(f"95% Prob: {np.percentile(preds_raw, 95):.4f}")
    logger.info("--- Calibrated Predictions ---")
    logger.info(f"Mean Prob: {preds.mean():.4f}")
    q95 = np.percentile(preds, 95)
    logger.info(f"95% Prob: {q95:.4f}")


if __name__ == "__main__":
    main()
