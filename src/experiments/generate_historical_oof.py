import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import logging
from tqdm import tqdm
from scipy.special import expit, logit

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.features.odds_movement_features import calculate_odds_movement_features

def load_data(year_start=2014, year_end=2025):
    """Load Base Data"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, '../../data/processed/preprocessed_data.parquet')
    
    logger.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Filter for training history + target years
    # We need history (2014-2023) to train for 2024.
    df = df[(df['year'] >= year_start) & (df['year'] <= year_end)].copy()
    
    # Target
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['target'] = (df['rank'] == 1).astype(int)
    
    # Odds for base margin
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(1000.0)
    # Inverse odds prob
    p_final = (1.0 / df['odds']).clip(1e-4, 1.0 - 1e-4)
    df['logit_final_odds'] = logit(p_final)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    return df

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_start', type=str, default='2024-01-01')
    parser.add_argument('--predict_end', type=str, default='2025-12-31')
    parser.add_argument('--output_file', type=str, default='data/predictions/v13_oof_2024_2025_with_odds_features.parquet')
    args = parser.parse_args()

    # Helper to save
    out_file = args.output_file
    
    # 1. Load Data/Features
    # We need full history to train
    df = load_data(2014, 2025)
    
    # 2. Add Odds Features
    logger.info("Computing Odds Features...")
    odds_feats = calculate_odds_movement_features(None, start_year=2014, end_year=2025)
    
    # Merge
    df['race_id'] = df['race_id'].astype(str)
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').fillna(0).astype(int)
    
    # odds_feats keys
    odds_feats['race_id'] = odds_feats['race_id'].astype(str)
    odds_feats['horse_number'] = odds_feats['horse_number'].astype(int)
    
    df = pd.merge(df, odds_feats, on=['race_id', 'horse_number'], how='left')
    
    # Feature Selection
    # Keep Base from Allowlist + New Odds
    # For simplicity, use numeric + category minus leak columns
    ignore = ['date', 'year', 'month', 'race_id', 'target', 'rank', 'logit_final_odds']
    # Add leak cols to ignore - STRICTLY EXCLUDE VALIDATION COLS
    leaks = [
        'time', 'agari', '着順', 'expectation', 'pred_prob', 'odds', 'ninki', 'tansho_odds', 'fukusho_odds',
        'raw_time', 'last_3f', 'popularity', 'rank_str', 'passing_rank', 
        'pass_1', 'pass_2', 'pass_3', 'pass_4'
    ] 
    
    features = [c for c in df.columns if c not in ignore and c not in leaks and not c.startswith('odds_') and not c.startswith('jvd_')]
    # Explicitly include the new odds features
    new_feats = [
        'log_odds_t10', 'dlog_odds_t60_t10', 'dlog_odds_t30_t10', 
        'odds_volatility', 'rank_change_t60_t10', 'odds_drop_rate_t60_t10'
    ]
    
    # Ensure features are in df
    features = [c for c in features if c in df.columns]
    # Add new feats explicitly if not present
    for nf in new_feats:
        if nf in df.columns and nf not in features:
            features.append(nf)
            
    logger.info(f"Features ({len(features)}): {features[:10]} ...")
    
    # 3. Walk Forward
    results = []
    
    # Monthly iteration
    p_start = pd.Timestamp(args.predict_start)
    p_end = pd.Timestamp(args.predict_end)
    
    months = pd.date_range(start=p_start, end=p_end, freq='MS')
    
    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(0)
        
        test_mask = (df['date'] >= m_start) & (df['date'] <= m_end)
        train_mask = (df['date'] < m_start) & (df['year'] >= 2014) # Limit history if needed, but 10 years is fine.
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if test_df.empty: continue
        
        logger.info(f"WF Prediction: {m_start.strftime('%Y-%m')} (Train: {len(train_df)}, Test: {len(test_df)})")
        
        # Train
        X_train = train_df[features]
        y_train = train_df['target']
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1
        }
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=100, # Fast training
            valid_sets=[dtrain], # Just to monitor
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Predict
        preds = model.predict(test_df[features])
        
        # Save Cols
        res_df = test_df[['race_id', 'horse_number', 'date', 'rank', 'odds', 'target', 'frame_number']].copy() # Keep frame_number for Wakuren
        res_df['pred_prob'] = preds
        
        results.append(res_df)
    
    if not results:
        logger.warning("No results generated.")
        return

    # Concat
    all_preds = pd.concat(results)
    
    logger.info(f"Saving {len(all_preds)} OOF rows to {out_file}")
    all_preds.to_parquet(out_file)

if __name__ == "__main__":
    main()
