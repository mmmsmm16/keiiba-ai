
import sys
import os
import logging
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

def main():
    logger.info("🚀 Starting M5 Step 2: Training (Win/Top2/Top3 Binary Models)")
    
    # 0. Config
    EXP_NAME = "exp_m5_ensemble_components"
    VALID_YEAR = 2024
    TRAIN_END_YEAR = 2023 # For time decay
    
    # Time Decay Settings (Piecewise Relative)
    # Delta = TrainEnd - RaceYear
    # 0 (2023) -> 1.0, 1 (2022) -> 0.7, 2 (2021) -> 0.5, Else -> 0.3
    DECAY_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.5, 'default': 0.3}
    
    # Targets to train
    TARGETS = [
        {'name': 'win', 'label_func': lambda r: 1 if r == 1 else 0},
        {'name': 'top2', 'label_func': lambda r: 1 if r <= 2 else 0},
        {'name': 'top3', 'label_func': lambda r: 1 if r <= 3 else 0}
    ]
    
    os.makedirs(f"experiments/{EXP_NAME}", exist_ok=True)
    
    # 1. Load Data
    logger.info("Loading Features and Targets...")
    if not os.path.exists("data/temp_m5/M5_features.parquet") or not os.path.exists("data/temp_m5/M5_targets.parquet"):
        logger.error("Missing parquet files! Run Step 1 first.")
        return

    df_features = pd.read_parquet("data/temp_m5/M5_features.parquet")
    df_targets = pd.read_parquet("data/temp_m5/M5_targets.parquet")
    
    logger.info(f"Features: {df_features.shape}, Targets: {df_targets.shape}")
    
    # 2. Merge
    logger.info("Merging Features and Targets...")
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='left')
    
    del df_features
    del df_targets
    gc.collect()
    
    # Ensure year column for splitter
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    # Convert object columns to category
    logger.info("Converting object columns to category...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    # 3. Process Each Target
    for tgt in TARGETS:
        tgt_name = tgt['name']
        logger.info(f"=== Processing Target: {tgt_name} ===")
        
        # Define Label
        df['target'] = df['rank'].apply(tgt['label_func'])
        
        # Split (Re-split for each target because y changes)
        # Usually X is same, but splitter handles row filtering based on valid_year logic?
        # DatasetSplitter splits by date/year.
        splitter = DatasetSplitter()
        datasets = splitter.split_and_create_dataset(df, valid_year=VALID_YEAR)
        train_set = datasets['train']
        valid_set = datasets['valid']
        
        X_train, y_train = train_set['X'], train_set['y']
        X_valid, y_valid = valid_set['X'], valid_set['y']
        
        # Time Decay Weighting
        logger.info("Applying Time Decay Weighting...")
        if 'year' in X_train.columns:
            years = X_train['year']
        elif 'date' in X_train.columns:
            years = pd.to_datetime(X_train['date']).dt.year
        else:
            # Fallback join
            years = df.loc[X_train.index, 'year']
            
        weights = np.full(len(X_train), DECAY_WEIGHTS['default'])
        deltas = TRAIN_END_YEAR - years
        for delta, w in DECAY_WEIGHTS.items():
            if delta == 'default': continue
            mask = (deltas == delta)
            weights[mask] = w
            
        # Normalize
        weights = weights / weights.mean()
        logger.info(f"Weights: Mean={weights.mean():.4f}, Min={weights.min():.4f}")
        
        # Train LGBM (Binary)
        logger.info(f"Training LightGBM ({tgt_name})...")
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'random_state': 42,
            'early_stopping_rounds': 100,
            'verbose': -1,
            'feature_fraction': 0.8  # Slight bagging
        }
        
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=[lgb.log_evaluation(100)]
        )
        
        # Save Model
        joblib.dump(model, f"experiments/{EXP_NAME}/model_{tgt_name}.pkl")
        
        # Calibration (Platt Scaling via OOF)
        logger.info("Calibrating (Platt Scaling)...")
        # Generate OOF predictions
        oof_preds = np.zeros(len(X_train))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Ensure 'race_id' is available for grouping if needed, but KFold shuffle is fine for calibration usually
        # But wait, GroupKFold is better for time-series/race data?
        # User requested "Reproducible". KFold(shuffle=True) is reproducible with seed.
        # Strict OOF usually requires GroupKFold to match production unseen-race scenario.
        # But here we just want a distribution calibration.
        
        groups = df.loc[X_train.index, 'race_id']
        unique_groups = groups.unique()
        kf_group = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf_group.split(unique_groups)):
            train_races = unique_groups[train_idx]
            val_races = unique_groups[val_idx]
            
            # Mask
            mask_t = groups.isin(train_races)
            mask_v = groups.isin(val_races)
            
            X_t, y_t = X_train[mask_t], y_train[mask_t]
            X_v = X_train[mask_v]
            w_t = weights[mask_t] # Use weights for OOF models too? Yes.
            
            # Train OOF model (lighter)
            oof_params = params.copy()
            oof_params['n_estimators'] = 500
            oof_params.pop('early_stopping_rounds', None)
            
            d_t = lgb.Dataset(X_t, label=y_t, weight=w_t)
            m_oof = lgb.train(oof_params, d_t, valid_sets=[d_t], callbacks=None)
            
            oof_preds[mask_v] = m_oof.predict(X_v)
            
        # Fit Platt Scaling (Logistic Regression on OOF scores)
        lr = LogisticRegression(C=99999, solver='lbfgs') # Low reg
        # Handle Inf/Nan
        oof_clean = np.nan_to_num(oof_preds, nan=0.0).reshape(-1, 1)
        lr.fit(oof_clean, y_train)
        
        # Save Calibrator
        joblib.dump(lr, f"experiments/{EXP_NAME}/calibrator_{tgt_name}.pkl")
        
        # Predict Valid & Calibrate
        y_pred_raw = model.predict(X_valid)
        y_pred_calib = lr.predict_proba(y_pred_raw.reshape(-1, 1))[:, 1]
        
        # Evaluate Calibration
        brier = brier_score_loss(y_valid, y_pred_calib)
        auc = roc_auc_score(y_valid, y_pred_calib)
        logger.info(f"[{tgt_name}] Calibrated Brier: {brier:.5f}, AUC: {auc:.5f}")
        
        # Save Valid Preds
        valid_res = pd.DataFrame({
            'race_id': df.loc[X_valid.index, 'race_id'].values,
            'horse_number': df.loc[X_valid.index, 'horse_number'].values,
            'y_true': y_valid.values,
            'y_score_raw': y_pred_raw,
            'y_prob': y_pred_calib
        })
        valid_res.to_parquet(f"experiments/{EXP_NAME}/valid_preds_{tgt_name}.parquet")
        
        # Save Metadata
        meta = {
            'target': tgt_name,
            'brier': brier,
            'auc': auc,
            'decay_strategy': 'piecewise_relative',
            'calibration': 'platt',
            'timestamp': datetime.now().isoformat()
        }
        with open(f"experiments/{EXP_NAME}/metadata_{tgt_name}.json", 'w') as f:
            json.dump(meta, f, indent=4)
            
    logger.info("M5 Step 2 Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("🔥 Fatal Error in Step 2:")
        traceback.print_exc()
        sys.exit(1)
