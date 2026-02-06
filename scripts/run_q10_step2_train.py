
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import traceback
import gc
import json
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.dataset import DatasetSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Phase Q Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    
    ds_cfg = cfg['dataset']
    VALID_YEAR = ds_cfg.get('valid_year', 2023)
    TRAIN_START = ds_cfg.get('train_start_date', '2015-01-01')
    TEST_END = ds_cfg.get('test_end_date', '2024-12-31')
    TARGET_TYPE = ds_cfg.get('binary_target', 'win') # win or top3
    
    logger.info(f"🚀 Starting Phase Q Training [{EXP_NAME}]")
    logger.info(f"   Valid Year: {VALID_YEAR}, Target: {TARGET_TYPE}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save config copy
    with open(os.path.join(MODEL_DIR, "config_copy.yaml"), 'w') as f:
        yaml.dump(cfg, f)
        
    # Standard Q10 paths
    FEAT_path = "data/temp_q10/Q10_features.parquet"
    TGT_path = "data/temp_q10/Q10_targets.parquet"
    
    print(f"DEBUG: Checking {FEAT_path}")
    if not os.path.exists(FEAT_path):
        logger.error(f"Missing features: {FEAT_path}")
        print(f"ERROR: Missing features: {FEAT_path}")
        return

    # 1. Load & Merge
    logger.info("Loading Data...")
    print("DEBUG: Loading Parquet...")
    try:
        df_features = pd.read_parquet(FEAT_path)
        df_targets = pd.read_parquet(TGT_path)
        print(f"DEBUG: Loaded. Feat={df_features.shape}, Tgt={df_targets.shape}")
    except Exception as e:
        print(f"ERROR loading parquet: {e}")
        traceback.print_exc()
        return
    
    print("DEBUG: Merging...")
    
    # Check output
    print(f"Feat Cols Sample: {list(df_features.columns)[:5]}")
    print(f"Tgt Cols: {list(df_targets.columns)}")
    
    # Merge
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='inner')
    
    print(f"Merged Cols Sample: {list(df.columns)[:10]}")
    
    # Check date column existence
    if 'date' not in df.columns:
        if 'date_y' in df.columns:
            print("DEBUG: renaming date_y to date")
            df.rename(columns={'date_y': 'date'}, inplace=True)
        elif 'date_x' in df.columns:
             print("DEBUG: renaming date_x to date")
             df.rename(columns={'date_x': 'date'}, inplace=True)
        else:
            print("ERROR: date col missing. Available: ", list(df.columns))
             
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
    else:
        # Fallback? No, split relies on year.
        print("CRITICAL ERROR: Cannot proceed without date.")
        return

    # Preprocessing
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff']
    
    # 1. Convert Configured Categoricals
    cat_cols = cfg['dataset']['categorical_features']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # 2. Safety Catch: Convert ANY remaining object columns to category
    # LightGBM crashes on object columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in exclude_cols: 
            logger.warning(f"Auto-converting object col to category: {col}")
            df[col] = df[col].astype('category')
            
    # Define Target
    # Support multiple targets? For now, stick to single binary target specified in config
    if TARGET_TYPE == 'win':
        label_func = lambda x: 1 if x == 1 else 0
    elif TARGET_TYPE == 'top3':
        label_func = lambda x: 1 if x <= 3 else 0
    else:
        logger.warning(f"Unknown target type {TARGET_TYPE}, defaulting to Win")
        label_func = lambda x: 1 if x == 1 else 0
        
    df['target'] = df['rank'].apply(label_func)
    
    # 2. Split
    # Using DatasetSplitter logic
    # But for Q1, we want fixed split: Train(Start~ValidYear-1), Valid(ValidYear)
    # We will manually split to be precise.
    
    logger.info("Splitting Data...")
    train_mask = (df['year'] < VALID_YEAR) & (df['year'] >= pd.to_datetime(TRAIN_START).year)
    valid_mask = (df['year'] == VALID_YEAR)
    
    # Evaluation set (Test) - 2024 (if valid is 2023)
    # Actually Q1 plan says Test=2024. So Valid should include 2023.
    # We will train on Train, Valid on Valid, and SAVE the model. Evaluation step comes later.
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    
    # Exclude metadata columns
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'prev_date', 'date_x', 'date_y',
                    'updated_at', 'created_at']
    
    # Filter features based on what is available in df
    X_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Further filter out datetime columns manually
    X_cols = [c for c in X_cols if not pd.api.types.is_datetime64_any_dtype(df[c])]
    
    logger.info(f"Train: {train_df.shape}, Valid: {valid_df.shape}")
    logger.info(f"Num Features: {len(X_cols)}")
    print(f"DEBUG: Feature columns: {X_cols[:10]} ...") # Print sample
    
    X_train = train_df[X_cols]
    y_train = train_df['target']
    X_valid = valid_df[X_cols]
    y_valid = valid_df['target']
    
    # 3. Train
    params = cfg['model_params']
    # Adjust params for LGBM API
    lgbm_params = {
        'objective': params['objective'],
        'metric': params['metric'],
        'learning_rate': params['learning_rate'],
        'num_leaves': params['num_leaves'],
        'bagging_fraction': params['bagging_fraction'],
        'feature_fraction': params['feature_fraction'],
        'bagging_freq': params['bagging_freq'],
        'verbose': params['verbose']
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
    
    logger.info("Training LightGBM...")
    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=params['n_estimators'],
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(params['early_stopping_rounds'])]
    )
    
    # 4. Save
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 5. Metrics
    y_pred = model.predict(X_valid)
    auc = roc_auc_score(y_valid, y_pred)
    logloss = log_loss(y_valid, y_pred)
    logger.info(f"Valid AUC: {auc:.4f}, LogLoss: {logloss:.4f}")
    
    metrics = {'auc': auc, 'logloss': logloss}
    with open(os.path.join(MODEL_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f)
        
    logger.info("Training Complete!")

if __name__ == "__main__":
    main()
