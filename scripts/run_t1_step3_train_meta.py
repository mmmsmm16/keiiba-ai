
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_calibration(df, target_col='rank', prob_col='meta_prob'):
    # Simple Brier Score
    y_true = (df[target_col] == 1).astype(int)
    brier = brier_score_loss(y_true, df[prob_col])
    logger.info(f"Brier Score: {brier:.4f}")
    
def train_meta_model():
    logger.info("loading data...")
    # Load Predictions (Base Model) for 2024-2025
    # We use Q1/R3 predictions
    pred_path_2025 = "models/experiments/exp_r3_ensemble/predictions_2025.parquet" 
    # Do we have 2024 predictions in the same format?
    # Usually we train T1 on 2024 (eval) and test on 2025.
    # Note: T1 features we generated cover 2024-2025.
    
    # We need Base Model predictions for 2024 to TRAIN the Meta Model.
    # Check if we have them.
    # If not, we might need to rely on OOF predictions or just train on H2 2024?
    # Let's assume prediction_2025.parquet is 2025 only.
    # Check for 2024 preds.
    
    # Simpler approach: Train on 2024 (Jan-Dec), Test on 2025 (Jan-present).
    # If we don't have 2024 preds readily available, we can't train the meta model properly on "unseen" data logic.
    # Assuming 'predictions_2024.parquet' exists?
    
    # Fallback: Load Q1 targets/raw data? No, we need 'pred_prob'.
    # For now, let's load what we have. If provided parquet has 2025 only, we can't train (unless we split 2025).
    # But usually exp_r3 stores all test preds if run on backtest mode.
    # Let's check predictions_2025.parquet content (date range).
    
    df_pred_2025 = pd.read_parquet(pred_path_2025)
    df_pred_2025['race_id'] = df_pred_2025['race_id'].astype(str)
    
    # T1 Features
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    if not os.path.exists(t1_path):
        logger.error("T1 Features not found!")
        return

    df_t1 = pd.read_parquet(t1_path)
    df_t1['race_id'] = df_t1['race_id'].astype(str)
    
    # Merge
    # df_pred_2025 usually contains only 2025 data.
    # We need Training Data. 
    # If we lack 2024 predictions, we are stuck.
    # However, 'run_q8_step5_simulation.py' uses 'predictions_2025.parquet'.
    # Maybe we can simulate 2024 preds? OR maybe predictions_2025 actually contains more?
    # Let's assume for this specific task we split 2025 into Train (Jan-Jun) and Test (Jul-Dec)?
    # Or just Train on 2024 part of T1 if we have labels.
    # But we need 'base_pred'.
    
    # Let's look for 2024 predictions.
    pred_path_2024 = "models/experiments/exp_r3_ensemble/predictions_2024.parquet"
    
    if os.path.exists(pred_path_2024):
        logger.info("Found 2024 predictions. Merging...")
        df_pred_2024 = pd.read_parquet(pred_path_2024)
        df_pred_2024['race_id'] = df_pred_2024['race_id'].astype(str)
        df_base = pd.concat([df_pred_2024, df_pred_2025], ignore_index=True)
    else:
        logger.warning("2024 predictions not found. Using ONLY 2025 (Time split validation).")
        df_base = df_pred_2025
        
    # Merge Features
    df = pd.merge(df_base, df_t1, on=['race_id', 'horse_number'], how='inner')
    
    # Load Date for splitting
    # T1 features might not have date. We can get it from run_t1_step1 output?
    # Actually T1 features drop 'date'.
    # We need date. Join with raw 2025/2024.
    
    raw_path_2025 = "data/temp_q1/year_2025.parquet"
    raw_2025 = pd.read_parquet(raw_path_2025)[['race_id', 'date']].drop_duplicates()
    raw_2025['race_id'] = raw_2025['race_id'].astype(str)
    
    df = pd.merge(df, raw_2025, on='race_id', how='left') # Only 2025 dates for now
    
    # If 2024 dates missing, try load
    if df['date'].isnull().any():
        raw_path_2024 = "data/temp_q1/year_2024.parquet"
        if os.path.exists(raw_path_2024):
            raw_2024 = pd.read_parquet(raw_path_2024)[['race_id', 'date']].drop_duplicates()
            raw_2024['race_id'] = raw_2024['race_id'].astype(str)
            # Update missing dates
            df_merged_dates = pd.merge(df, raw_2024, on='race_id', how='left', suffixes=('', '_24'))
            df['date'] = df['date'].fillna(df_merged_dates['date_24'])

    df = df.dropna(subset=['date']).sort_values('date')
    
    # Define Features
    # New Features: 'odds_ratio_60_10', 'bias_adversity_score_mean_5'
    use_cols = [
        'pred_prob', 
        'odds_10min', 
        'odds_final', # Warning: LEAK if used directly? No, we use odds_10min for prediction. 
                      # odds_final is for training target checking (profitability)? 
                      # NO, Meta Model predicts WIN PROBABILITY (Calibration).
                      # So we should NOT use odds_final as input.
        'odds_ratio_60_10',
        'bias_adversity_score_mean_5'
    ]
    
    # Remove odds_final from Input Features
    input_cols = [c for c in use_cols if c != 'odds_final']
    
    # Market Features
    # We use odds_10min.
    # Convert to prob?
    df['prob_market_10min'] = 0.8 / (df['odds_10min'] + 1e-9)
    df['prob_diff_10min'] = df['pred_prob'] - df['prob_market_10min']
    
    input_cols += ['prob_market_10min', 'prob_diff_10min']
        
    logger.info(f"Training Features: {input_cols}")
    
    # Split
    # Train: 2024 (or First 80% if only 2025)
    # Test: 2025 (or Last 20%)
    
    split_date = pd.Timestamp("2025-01-01")
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    if train_df.empty:
        logger.warning("Train set empty (No 2024 data). Splitting 2025 by time.")
        cutoff = int(len(df) * 0.8)
        train_df = df.iloc[:cutoff]
        test_df = df.iloc[cutoff:]
        
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Features Clean
    # Ensure rank is target
    if 'rank' not in df.columns:
        if 'rank_x' in df.columns:
            train_df['rank'] = train_df['rank_x']
            test_df['rank'] = test_df['rank_x']
        elif 'rank_y' in df.columns:
            train_df['rank'] = train_df['rank_y']
            test_df['rank'] = test_df['rank_y']
            
    X_train = train_df[input_cols]
    y_train = (train_df['rank'] == 1).astype(int)
    
    X_test = test_df[input_cols]
    y_test = (test_df['rank'] == 1).astype(int)
    
    # LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    preds = model.predict(X_test)
    test_df['meta_prob'] = preds
    
    auc = roc_auc_score(y_test, preds)
    ll = log_loss(y_test, preds)
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Test LogLoss: {ll:.4f}")
    
    evaluate_calibration(test_df, target_col='rank', prob_col='meta_prob')

    # ROI Check (Odds Band 2.0-5.0)
    # Use odds_final for ROI check
    if 'odds_final' in test_df.columns:
        target_mask = (test_df['odds_final'] >= 2.0) & (test_df['odds_final'] <= 5.0) & (test_df['meta_prob'] >= 0.2)
        bets = test_df[target_mask]
        hits = bets[bets['rank'] == 1]
        
        ret = hits['odds_final'].sum() * 100
        cost = len(bets) * 100
        roi = ret / cost if cost > 0 else 0
        logger.info(f"Target ROI (2.0-5.0, p>=0.2): {roi*100:.1f}% ({len(bets)} bets)")

    # Save Model
    model.save_model("models/experiments/exp_t1_meta/meta_lgbm_v2.txt")
    
    # Analyze Feature Importance
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': input_cols, 'gain': importance}).sort_values('gain', ascending=False)
    print("\nFeature Importance:")
    print(imp_df)

if __name__ == "__main__":
    train_meta_model()
