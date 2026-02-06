
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import logging
import argparse
import joblib
import sys
from scipy.special import logit

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
try:
    from src.features.odds_movement_features import calculate_odds_movement_features
except ImportError:
    # Fallback if running from root
    from src.features.odds_movement_features import calculate_odds_movement_features

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(parquet_path):
    """Load and preprocess Base Data identical to Phase 12"""
    logger.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
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

def train_production_model(data_path, output_model_path, feature_list_path=None):
    """
    Train the final production model using all available historical data (up to end of 2024).
    """
    logger.info("Starting Production Model Training...")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return

    df = load_data(data_path)
    
    # 2. Add Odds Movement Features
    logger.info("Calculating Odds Movement Features...")
    # Does not use input df, just loads from disk
    odds_df = calculate_odds_movement_features(None, start_year=2014, end_year=2024)
    
    # Merge
    logger.info("Merging Odds Features...")
    df['race_id'] = df['race_id'].astype(str)
    odds_df['race_id'] = odds_df['race_id'].astype(str)
    
    df = pd.merge(df, odds_df, on=['race_id', 'horse_number'], how='left')
    
    # 3. Features (Identical to Phase 12)
    # Re-using logic from generate_historical_oof.py
    ignore = ['date', 'year', 'month', 'race_id', 'target', 'rank', 'logit_final_odds']
    leaks = [
        'time', 'agari', '着順', 'expectation', 'pred_prob', 'odds', 'ninki', 'tansho_odds', 'fukusho_odds',
        'raw_time', 'last_3f', 'popularity', 'rank_str', 'passing_rank', 
        'pass_1', 'pass_2', 'pass_3', 'pass_4'
    ]
    
    # Base features
    features = [c for c in df.columns if c not in ignore and c not in leaks and not c.startswith('odds_') and not c.startswith('jvd_')]
    
    # Add T-10 Odds features explicitly
    new_feats = [
        'log_odds_t10', 'dlog_odds_t60_t10', 'dlog_odds_t30_t10', 
        'odds_volatility', 'rank_change_t60_t10', 'odds_drop_rate_t60_t10'
    ]
    
    for nf in new_feats:
        if nf in df.columns and nf not in features:
            features.append(nf)
            
    logger.info(f"Selected {len(features)} features.")
    
    # Save feature list for reproducibility
    if feature_list_path:
        os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
        joblib.dump(features, feature_list_path)
        logger.info(f"Feature list saved to {feature_list_path}")
        
    # 4. Train on 2014-2024
    train_mask = (df['year'] >= 2014) & (df['year'] <= 2024)
    train_df = df[train_mask]
    
    logger.info(f"Training Data: {len(train_df)} rows (2014-2024)")
    
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
    
    logger.info("Training LightGBM model...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500 
    )
    
    # 5. Save Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save_model(output_model_path)
    logger.info(f"Model saved to {output_model_path}")
    
if __name__ == "__main__":
    train_production_model(
        'data/processed/preprocessed_data.parquet',
        'models/production/v13_production_model.txt',
        'models/production/v13_feature_list.joblib'
    )

