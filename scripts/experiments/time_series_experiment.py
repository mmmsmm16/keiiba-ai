"""
Time-Series Features Experiment
===============================
Generates and evaluates new time-series features:
1. momentum_score: Trend of performance (rank) over recent races.
2. consistency_score: Stability of performance (std dev of rank).
3. recovery_rate: Performance change after long intervals.
4. elo_momentum: Trend of Elo rating.

Usage:
  python scripts/experiments/time_series_experiment.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from scipy.stats import linregress
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
BASE_MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"

def load_data():
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    targets['race_id'] = targets['race_id'].astype(str)
    
    df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                  on=['race_id', 'horse_number'], how='left')
    df['date'] = pd.to_datetime(df['date'])
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    return df

def calculate_slope(y):
    """Calculate slope of linear regression for a series"""
    if len(y) < 2:
        return np.nan
    try:
        # X is just 0, 1, 2...
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return slope
    except:
        return np.nan

def generate_time_series_features(df):
    logger.info("Generating time-series features...")
    
    df = df.sort_values(['horse_id', 'date']).copy()
    
    # Pre-calculate shifted ranks to avoid leakage
    # We want features based on PAST races only.
    # Group by horse_id
    grouped = df.groupby('horse_id')
    
    # 1. Momentum (Slope of ranks in last 5 races)
    # We apply rolling on 'rank' but we must shift first because current rank is unknown
    # shift(1) puts previous race rank in current row.
    # Rolling window of 5 on shifted rank.
    logger.info("  - Momentum Score...")
    # Fill nan ranks with mean or something? Or just skip.
    # We use 'rank' column which exists for training. 
    # NOTE: In production, we need to ensure we use past ranks.
    # Using shift(1) ensures we use previous accumulated data.
    
    # Normalized rank (0-1) to make slope comparable across field sizes?
    # For simplicity, let's use raw rank first, but maybe log rank is better.
    
    def calc_rolling_slope(series, window=5):
        return series.rolling(window=window).apply(calculate_slope, raw=True)

    # Shift rank first
    df['prev_rank'] = grouped['rank'].shift(1)
    
    # Calculate slope on previous 5 ranks
    # This is slow with apply, let's try a vectorized approx or optimizations if needed.
    # For 5 points, strict slope is: (Sum(xy) - n*mean_x*mean_y) / (Sum(x^2) - n*mean_x^2)
    # Since x is fixed [0,1,2,3,4], we can optimize.
    # But for now, let's verify impact first.
    
    # Optimization: Use fixed weights for slope?
    # Slope of y on x=[0,1,2,3,4] is dot(w, y) where w are precalculated weights centered.
    # x = [-2, -1, 0, 1, 2] centered. Sum(x^2) = 10.
    # Slope ~ Sum(x* (y - mean_y)) / 10
    # This is rough but faster.
    
    # Let's stick to a simpler momentum: (Recency weighted average) - (Long term average)
    # Or just average of differences.
    
    # Let's use simple difference between Lag1 and Lag5
    df['rank_change_5'] = df['prev_rank'] - grouped['rank'].shift(5)
    
    # 2. Consistency (Std Dev of ranks in last 5-10 races)
    logger.info("  - Consistency Score...")
    df['rank_std_5'] = grouped['prev_rank'].rolling(5).std().reset_index(level=0, drop=True)
    df['rank_std_10'] = grouped['prev_rank'].rolling(10).std().reset_index(level=0, drop=True)
    
    # consistency_score = 1 / (std + 1)
    df['consistency_score'] = 1 / (df['rank_std_5'].fillna(100) + 1)
    
    # 3. Recovery Rate
    # Performance when interval > 60 days
    logger.info("  - Recovery Rate...")
    # Check if previous interval was long
    # We need interval from previous row
    
    # 'interval' column exists in preprocessed data (days since prev race)
    # But that's for the CURRENT race.
    # We want to know: "How does this horse perform after a long break?"
    # We calculate average rank when previous interval > 60
    
    # Identify `long_rest` races
    df['is_long_rest'] = (df['interval'] > 60)
    
    # Calculate average rank in races where is_long_rest is True (expanding mean)
    # SHIFT IS CRITICAL: We only know recovery rate from PAST long rest races.
    
    # Custom expanding mean on filtered rows is tricky in pandas
    # Let's simple use: Last time it had long rest, what was the rank?
    
    # Mask ranks where NOT long rest
    cols_to_keep = ['horse_id', 'date', 'rank', 'is_long_rest']
    temp = df[cols_to_keep].copy()
    temp.loc[~temp['is_long_rest'], 'rank'] = np.nan
    
    # Forward fill the last "long rest rank"
    temp['last_long_rest_rank'] = temp.groupby('horse_id')['rank'].ffill().shift(1)
    df['recovery_prev_rank'] = temp['last_long_rest_rank']
    
    # 4. Elo Momentum
    # Current preprocessed data has `rating_elo`.
    # We want the trend of Elo entering this race.
    # `rating_elo` is the rating BEFORE the race (usually). 
    # Let's verify: In FeaturePipeline, rating_elo is usually lag1 rating.
    logger.info("  - Elo Momentum...")
    if 'rating_elo' in df.columns:
        df['elo_momentum_3'] = df['rating_elo'] - grouped['rating_elo'].shift(3)
        df['elo_momentum_1'] = df['rating_elo'] - grouped['rating_elo'].shift(1)
    
    return df

def run_experiment():
    logger.info("=" * 60)
    logger.info("Time-Series Features Experiment")
    logger.info("=" * 60)
    
    df = load_data()
    df = generate_time_series_features(df)
    
    # Features to test
    new_features = ['rank_change_5', 'rank_std_5', 'consistency_score', 
                   'recovery_prev_rank', 'elo_momentum_3', 'elo_momentum_1']
    
    valid_features = [f for f in new_features if f in df.columns]
    logger.info(f"Testing features: {valid_features}")
    
    # Load base model for features
    base_model = joblib.load(BASE_MODEL_PATH)
    base_features = base_model.feature_name()
    
    # All features
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'is_win', 
                    'is_top2', 'is_top3', 'rank_str', 'year'] + \
                    ['prev_rank', 'is_long_rest'] # Helpers
                    
    feature_cols = [c for c in df.columns if c in base_features or c in valid_features]
    feature_cols = [c for c in feature_cols if c not in exclude_cols]
    
    # Split
    df_train = df[df['date'].dt.year <= 2022].copy()
    df_valid = df[df['date'].dt.year == 2023].copy()
    df_test = df[df['date'].dt.year == 2024].copy()
    
    # Base Params
    params = {
        'objective': 'binary', # Use binary for quick valid
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Train
    def train_eval(features, name):
        X_tr = df_train[features].copy()
        X_va = df_valid[features].copy()
        X_te = df_test[features].copy()
        y_tr = df_train['is_win']
        y_va = df_valid['is_win']
        y_te = df_test['is_win']
        
        for c in X_tr.columns:
             if X_tr[c].dtype == 'object' or X_tr[c].dtype.name == 'category':
                X_tr[c] = X_tr[c].astype('category').cat.codes
                X_va[c] = X_va[c].astype('category').cat.codes
                X_te[c] = X_te[c].astype('category').cat.codes
             X_tr[c] = X_tr[c].fillna(-999)
             X_va[c] = X_va[c].fillna(-999)
             X_te[c] = X_te[c].fillna(-999)

        train_ds = lgb.Dataset(X_tr, label=y_tr)
        valid_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)
        
        model = lgb.train(
            params, train_ds, num_boost_round=1000, 
            valid_sets=[valid_ds], 
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_te)
        auc = roc_auc_score(y_te, pred)
        logger.info(f"{name} Test AUC: {auc:.4f}")
        return auc, model
    
    logger.info("--- Baseline ---")
    base_feats_only = [f for f in feature_cols if f not in new_features]
    base_auc, base_model_obj = train_eval(base_feats_only, "Baseline")
    
    logger.info("--- With New Features ---")
    new_auc, new_model_obj = train_eval(feature_cols, "New Features")
    
    improvement = (new_auc - base_auc) * 100
    print("\n" + "=" * 60)
    print(" Results Summary")
    print("=" * 60)
    print(f"Baseline AUC:    {base_auc:.4f}")
    print(f"New Features AUC:{new_auc:.4f}")
    print(f"Improvement:     {improvement:+.3f}%")
    
    # Feature Importance of new features
    print("\nNew Feature Importance:")
    imp = new_model_obj.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': feature_cols, 'gain': imp}).sort_values('gain', ascending=False)
    
    for f in valid_features:
        rank = imp_df[imp_df['feature'] == f].index[0] if f in imp_df['feature'].values else -1
        gain = imp_df[imp_df['feature'] == f]['gain'].values[0] if f in imp_df['feature'].values else 0
        print(f"  {f:<20}: Rank {rank+1}/{len(feature_cols)}, Gain={gain:.1f}")

if __name__ == "__main__":
    run_experiment()
