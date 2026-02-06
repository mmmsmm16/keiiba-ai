
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Generating T2 Base Model Predictions for 2025...")
    
    MODEL_PATH = "models/experiments/exp_t2_track_bias/model.pkl"
    OUTPUT_PATH = "data/temp_t2/T2_predictions_2025.parquet"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
        
    # Load model
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    
    # Load config
    cfg_path = "config/experiments/exp_t2_track_bias.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    feature_blocks = cfg['features']
    
    # Load 2025 data
    loader = JraVanDataLoader()
    df = loader.load(history_start_date='2025-01-01', end_date='2025-12-31', skip_training=False, skip_odds=True)
    
    if df.empty:
        logger.error("No 2025 data!")
        return
        
    logger.info(f"Loaded 2025 data: {len(df)} rows")
    
    # Rename columns for track_bias compatibility
    if 'frame_number' in df.columns:
        df['waku_no'] = df['frame_number']
    if 'passing_rank' in df.columns:
        df['pass_rank'] = df['passing_rank']
    
    # Generate features
    pipeline = FeaturePipeline(cache_dir="data/features")
    df_feat = pipeline.load_features(df, feature_blocks)
    
    logger.info(f"Features generated: {df_feat.shape}")
    
    # Prepare X
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'updated_at', 'created_at']
    X_cols = [c for c in df_feat.columns if c not in exclude_cols]
    X_cols = [c for c in X_cols if not pd.api.types.is_datetime64_any_dtype(df_feat[c])]
    
    # Categorical processing
    cat_cols = cfg['dataset']['categorical_features']
    for col in cat_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype('category')
    for col in X_cols:
        if col in df_feat.columns and df_feat[col].dtype == 'object':
            df_feat[col] = df_feat[col].astype('category')
    
    X = df_feat[X_cols]
    
    # Predict
    logger.info("Generating predictions...")
    df_feat['pred_prob'] = model.predict(X)
    
    # Get date and rank from original df
    df_feat['date'] = df['date'].values
    df_feat['rank'] = df['rank'].values
    
    # Evaluate
    df_feat['is_win'] = (df_feat['rank'] == 1).astype(int)
    valid_mask = df_feat['is_win'].notna()
    
    from sklearn.metrics import roc_auc_score
    if valid_mask.sum() > 0:
        auc = roc_auc_score(df_feat.loc[valid_mask, 'is_win'], df_feat.loc[valid_mask, 'pred_prob'])
        logger.info(f"2025 AUC: {auc:.4f}")
    
    # Save
    output_cols = ['race_id', 'horse_number', 'date', 'rank', 'pred_prob']
    df_feat[output_cols].to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
