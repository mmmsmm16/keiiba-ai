
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import pickle

sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Training Meta-Model with T2 Base Model Predictions...")
    
    # Load predictions
    pred_path = "data/temp_t2/T2_predictions_2024_2025.parquet"
    
    df = pd.read_parquet(pred_path)
    df['race_id'] = df['race_id'].astype(str)

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Year distribution:\n{df['year'].value_counts()}")
    
    # Load T1 features (odds_ratio_60_10, bias_adversity_score_mean_5)
    # Merge T1 features
    # T1 Features
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    if os.path.exists(t1_path):
        df_t1 = pd.read_parquet(t1_path)
        df_t1['race_id'] = df_t1['race_id'].astype(str)
        
        # Deduplicate T1 features
        before_len = len(df_t1)
        df_t1 = df_t1.drop_duplicates(['race_id', 'horse_number'])
        logger.info(f"Deduplicated T1 features: {before_len} -> {len(df_t1)}")
        
        # Merge necessary features and odds
        # T1 features already contain odds_10min and odds_final
        t1_cols = ['race_id', 'horse_number', 'odds_ratio_60_10', 'bias_adversity_score_mean_5', 'odds_10min', 'odds_final']
        existing_t1_cols = [c for c in t1_cols if c in df_t1.columns]
        
        df = pd.merge(df, df_t1[existing_t1_cols], 
                      on=['race_id', 'horse_number'], how='left')
        logger.info(f"Merged T1 features. Rows: {len(df)}")
    else:
        logger.warning("T1 features not found. Training might fail without odds.")
        df['odds_ratio_60_10'] = 1.0
        df['odds_10min'] = np.nan
        df['odds_final'] = np.nan

    # Ensure Date column is available
    # Join with raw data to get dates if not already present
    if 'date' not in df.columns:
        date_dfs = []
        for year in [2024, 2025]:
            p = f"data/temp_q1/year_{year}.parquet"
            if os.path.exists(p):
                date_dfs.append(pd.read_parquet(p)[['race_id', 'date']].drop_duplicates())
        if date_dfs:
            df_dates = pd.concat(date_dfs, ignore_index=True)
            df_dates['race_id'] = df_dates['race_id'].astype(str)
            df = pd.merge(df, df_dates, on='race_id', how='left')
    
    # Feature Engineering
    # Market probability
    df['prob_market'] = 0.8 / (df['odds_10min'].fillna(10) + 1e-9)
    df['prob_diff'] = df['pred_prob'] - df['prob_market']
    
    # Fillna for optional features
    df['odds_ratio_60_10'] = df['odds_ratio_60_10'].fillna(1.0)
    if 'bias_adversity_score_mean_5' not in df.columns:
        df['bias_adversity_score_mean_5'] = 0
    df['bias_adversity_score_mean_5'] = df['bias_adversity_score_mean_5'].fillna(0)
    
    # Define features
    input_cols = [
        'pred_prob',           # Base Model prediction
        'prob_market',         # Market probability
        'prob_diff',           # Diff between model and market
        'odds_ratio_60_10',    # Odds movement (late drop)
        'bias_adversity_score_mean_5'  # Track bias feature (redundant but may help)
    ]
    
    logger.info(f"Meta Features: {input_cols}")
    
    # Filter rows with valid data
    df = df.dropna(subset=['rank', 'pred_prob', 'odds_10min'])
    logger.info(f"Rows after dropna: {len(df)}")
    
    # Split
    split_date = pd.Timestamp("2024-10-01")
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    logger.info(f"Train (<2024-10): {len(train_df)}, Valid (>=2024-10): {len(test_df)}")
    
    if train_df.empty:
        logger.error("Train set empty!")
        return
    
    X_train = train_df[input_cols]
    y_train = (train_df['rank'] == 1).astype(int)
    
    X_test = test_df[input_cols]
    y_test = (test_df['rank'] == 1).astype(int)
    
    # Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'max_depth': 4,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)]
    )
    
    # Evaluate
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    train_auc = roc_auc_score(y_train, preds_train)
    test_auc = roc_auc_score(y_test, preds_test)
    test_ll = log_loss(y_test, preds_test)
    
    logger.info(f"Train AUC: {train_auc:.4f}")
    logger.info(f"Test AUC (2025): {test_auc:.4f}")
    logger.info(f"Test LogLoss: {test_ll:.4f}")
    
    # Calibration (Brier Score)
    brier = brier_score_loss(y_test, preds_test)
    logger.info(f"Brier Score: {brier:.4f}")
    
    # ROI Simulation
    test_df['meta_prob'] = preds_test
    test_df['is_win'] = (test_df['rank'] == 1).astype(int)
    
    # ROI by strategy
    strategies = [
        ('High Prob (>=0.25)', 0.25, None, None),
        ('Middle Odds (2-10)', 0.15, 2.0, 10.0),
        ('Low Odds (1.5-5)', 0.15, 1.5, 5.0),
        ('Odds Drop (<0.9)', 0.10, None, None),  # Will filter by odds_ratio
    ]
    
    for name, prob_th, odds_low, odds_high in strategies:
        mask = test_df['meta_prob'] >= prob_th
        if odds_low is not None and odds_high is not None:
            mask = mask & (test_df['odds_final'] >= odds_low) & (test_df['odds_final'] <= odds_high)
        if 'Odds Drop' in name:
            mask = mask & (test_df['odds_ratio_60_10'] < 0.9)
        
        bets = test_df[mask]
        if len(bets) == 0:
            continue
        wins = bets[bets['is_win'] == 1]
        ret = wins['odds_final'].sum() * 100
        cost = len(bets) * 100
        roi = ret / cost if cost > 0 else 0
        hit_rate = len(wins) / len(bets) if len(bets) > 0 else 0
        logger.info(f"  {name}: ROI={roi*100:.1f}%, Bets={len(bets)}, HR={hit_rate*100:.1f}%")
    
    # Save model
    os.makedirs("models/experiments/exp_t2_meta", exist_ok=True)
    model.save_model("models/experiments/exp_t2_meta/meta_lgbm.txt")
    
    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': input_cols, 'gain': importance}).sort_values('gain', ascending=False)
    print("\nFeature Importance:")
    print(imp_df.to_string())
    
    # Save predictions
    test_df[['race_id', 'horse_number', 'date', 'rank', 'pred_prob', 'meta_prob', 'odds_final']].to_parquet(
        "data/temp_t2/T2_meta_predictions_2025.parquet", index=False
    )
    logger.info("Saved meta predictions.")

if __name__ == "__main__":
    main()
