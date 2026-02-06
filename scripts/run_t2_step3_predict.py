
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Generating T2 Base Model Predictions for Meta-Model...")
    
    MODEL_PATH = "models/experiments/exp_t2_refined/model.pkl"
    OUTPUT_PATH = "data/temp_t2/T2_predictions_2024_2025.parquet"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
        
    # Load model
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    
    # Load features (already generated)
    feat_path = "data/temp_t2/T2_features.parquet"
    tgt_path = "data/temp_t2/T2_targets.parquet"
    
    df_features = pd.read_parquet(feat_path)
    df_targets = pd.read_parquet(tgt_path)
    
    # Merge
    df_features['race_id'] = df_features['race_id'].astype(str)
    df_targets['race_id'] = df_targets['race_id'].astype(str)
    df_features['horse_number'] = pd.to_numeric(df_features['horse_number'], errors='coerce').fillna(0).astype(int)
    df_targets['horse_number'] = pd.to_numeric(df_targets['horse_number'], errors='coerce').fillna(0).astype(int)
    
    logger.info(f"Target cols: {df_targets.columns.tolist()}")
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='inner')
    logger.info(f"Merged cols: {df.columns.tolist()}")
    logger.info(f"Merged shape: {df.shape}")
    
    if 'date_y' in df.columns:
        df = df.rename(columns={'date_y': 'date'})
    if 'date_x' in df.columns:
        df = df.drop(columns=['date_x'])
        
    if 'date' not in df.columns:
        logger.error("Date column missing after merge!")
        # Attempt to recover date from race_id if possible or exit
        # For now just log
        
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Filter to 2024-2025 for Meta-Model training/testing
    df_meta = df[df['year'] >= 2024].copy()
    logger.info(f"Meta input data shape: {df_meta.shape}")
    
    # Prepare X
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'updated_at', 'created_at']
    X_cols = [c for c in df.columns if c not in exclude_cols]
    X_cols = [c for c in X_cols if not pd.api.types.is_datetime64_any_dtype(df[c])]
    
    # Categorical processing (same as training)
    import yaml
    cfg_path = "config/experiments/exp_t2_refined.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    cat_cols = cfg['dataset']['categorical_features']
    for col in cat_cols:
        if col in df_meta.columns:
            df_meta[col] = df_meta[col].astype('category')
    for col in X_cols:
        if col in df_meta.columns and df_meta[col].dtype == 'object':
            df_meta[col] = df_meta[col].astype('category')
    
    X_meta = df_meta[X_cols]
    
    # Filter to model features
    model_features = model.feature_name()
    # Check if all model features exist
    missing_feats = [f for f in model_features if f not in X_meta.columns]
    if missing_feats:
        logger.warning(f"Missing features in input: {missing_feats}")
        # Add missing features as 0 or nan
        for f in missing_feats:
            X_meta[f] = 0
            
    # Reorder and select only model features
    X_meta = X_meta[model_features]
    
    # Apply categorical casting strictly for model features
    for col in X_meta.columns:
        if col in cat_cols:
             X_meta[col] = X_meta[col].astype('category')
    
    # Predict
    logger.info("Generating predictions...")
    df_meta['pred_prob'] = model.predict(X_meta)
    
    # Generate targets
    df_meta['is_win'] = (df_meta['rank'] == 1).astype(int)
    
    # Evaluate on 2024 and 2025
    for year in [2024, 2025]:
        year_df = df_meta[df_meta['year'] == year]
        if not year_df.empty:
            auc = roc_auc_score(year_df['is_win'], year_df['pred_prob'])
            logger.info(f"{year} AUC: {auc:.4f}")
    
    # Save
    output_cols = ['race_id', 'horse_number', 'date', 'rank', 'pred_prob']
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_meta[output_cols].to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
