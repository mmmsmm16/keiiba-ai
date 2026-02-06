
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.dataset import DatasetSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_calibrate(df, valid_year, targets, exp_dir):
    """
    Train and calibrate for a specific validation year.
    Matches Step 2 logic but dynamic year.
    """
    logger.info(f"--- Walkforward Fold: Valid Year {valid_year} ---")
    fold_dir = f"{exp_dir}/valid_{valid_year}"
    os.makedirs(fold_dir, exist_ok=True)
    
    DECAY_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.5, 'default': 0.3}
    train_end_year = valid_year - 1
    
    all_preds = []
    
    for tgt in targets:
        tgt_name = tgt['name']
        logger.info(f"Target: {tgt_name}")
        
        df_tgt = df.copy()
        df_tgt['target'] = df_tgt['rank'].apply(tgt['label_func'])
        
        splitter = DatasetSplitter()
        # Split by year
        datasets = splitter.split_and_create_dataset(df_tgt, valid_year=valid_year)
        train_set = datasets['train']
        valid_set = datasets['valid']
        
        X_train, y_train = train_set['X'], train_set['y']
        X_valid, y_valid = valid_set['X'], valid_set['y']
        
        # Weights
        years = X_train['year'] if 'year' in X_train.columns else pd.to_datetime(X_train['date']).dt.year
        weights = np.full(len(X_train), DECAY_WEIGHTS['default'])
        deltas = train_end_year - years
        for delta, w in DECAY_WEIGHTS.items():
            if delta == 'default': continue
            mask = (deltas == delta)
            weights[mask] = w
        weights = weights / weights.mean()
        
        # Params
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 1500, # Balanced for walkforward
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1,
            'feature_fraction': 0.8
        }
        
        # Train
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        # Save Model
        joblib.dump(model, f"{fold_dir}/model_{tgt_name}.pkl")
        
        # Calibration (Platt via GroupKFold)
        logger.info(f"Calibrating {tgt_name}...")
        groups = df_tgt.loc[X_train.index, 'race_id']
        unique_groups = groups.unique()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_groups)):
            train_races = unique_groups[train_idx]
            val_races = unique_groups[val_idx]
            m_t, m_v = groups.isin(train_races), groups.isin(val_races)
            
            d_t = lgb.Dataset(X_train[m_t], label=y_train[m_t], weight=weights[m_t])
            # Rapid OOF
            oof_params = params.copy()
            oof_params['n_estimators'] = 500
            m_oof = lgb.train(oof_params, d_t)
            oof_preds[m_v] = m_oof.predict(X_train[m_v])
            
        lr = LogisticRegression(C=1e5)
        lr.fit(oof_preds.reshape(-1, 1), y_train)
        joblib.dump(lr, f"{fold_dir}/calibrator_{tgt_name}.pkl")
        
        # Predict
        y_score_raw = model.predict(X_valid)
        y_prob = lr.predict_proba(y_score_raw.reshape(-1, 1))[:, 1]
        
        res = pd.DataFrame({
            'race_id': df_tgt.loc[X_valid.index, 'race_id'].values,
            'horse_number': df_tgt.loc[X_valid.index, 'horse_number'].values,
            'year': valid_year,
            'target': tgt_name,
            'p_raw': y_score_raw,
            'p_calib': y_prob,
            'is_hit': y_valid.values
        })
        all_preds.append(res)
        
        # Cleanup
        del df_tgt
        gc.collect()
        
    return pd.concat(all_preds)

def main():
    logger.info("🚀 Starting M5 Step 4: Walk-Forward Prediction (2022-2024)")
    
    EXP_DIR = "experiments/exp_m5_walkforward"
    os.makedirs(EXP_DIR, exist_ok=True)
    
    # Load Data
    logger.info("Loading Data...")
    df_features = pd.read_parquet("data/temp_m5/M5_features.parquet")
    df_targets = pd.read_parquet("data/temp_m5/M5_targets.parquet")
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='left')
    
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    targets = [
        {'name': 'win', 'label_func': lambda r: 1 if r == 1 else 0},
        {'name': 'top2', 'label_func': lambda r: 1 if r <= 2 else 0},
        {'name': 'top3', 'label_func': lambda r: 1 if r <= 3 else 0}
    ]
    
    valid_years = [2022, 2023, 2024]
    final_dfs = []
    
    for year in valid_years:
        res = train_and_calibrate(df, year, targets, EXP_DIR)
        final_dfs.append(res)
        
    master_preds = pd.concat(final_dfs)
    
    # Reshape: target per column
    logger.info("Reshaping predictions...")
    reshaped = []
    for (rid, hno, yr), group in master_preds.groupby(['race_id', 'horse_number', 'year']):
        row = {'race_id': rid, 'horse_number': hno, 'year': yr}
        for _, g_row in group.iterrows():
            t = g_row['target']
            row[f'p_{t}'] = g_row['p_calib']
            if t == 'top3':
                row['is_top3'] = g_row['is_hit']
            elif t == 'win':
                row['is_win'] = g_row['is_hit']
        reshaped.append(row)
        
    df_final = pd.DataFrame(reshaped)
    
    # Add Ensemble Score
    df_final['ensemble_score_w1'] = 0.5 * df_final['p_win'] + 0.3 * df_final['p_top2'] + 0.2 * df_final['p_top3']
    df_final['ensemble_score_avg'] = (df_final['p_win'] + df_final['p_top2'] + df_final['p_top3']) / 3.0
    
    logger.info(f"Saving Master Preds: {df_final.shape}")
    df_final.to_parquet(f"{EXP_DIR}/walkforward_preds_2022_2024.parquet")
    logger.info("M5 Step 4 Complete!")

if __name__ == "__main__":
    main()
