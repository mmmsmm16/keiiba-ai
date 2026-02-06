"""
NAR Model Training with JRA v13-Parity
- OOF (Out-of-Fold) Predictions via TimeSeriesSplit
- Isotonic Calibration for probability correction
- LambdaRank for ranking optimization
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax
import joblib
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
DATA_PATH = "data/nar/preprocessed_data_south_kanto.parquet"
MODEL_DIR = "models/production/nar"
MODEL_NAME = "v2_south_kanto.txt"
CALIBRATOR_NAME = "v2_calibrator.pkl"
OOF_PREDICTIONS_NAME = "v2_oof_predictions.parquet"
os.makedirs(MODEL_DIR, exist_ok=True)

# LambdaRank Best Params (from JRA v7)
PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 76,
    'min_data_in_leaf': 53,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.6,
    'bagging_freq': 7,
    'lambda_l1': 1.5e-05,
    'lambda_l2': 0.05,
    'random_state': 42,
    'verbose': -1
}

def load_data():
    """Load preprocessed NAR data"""
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Target for ranking (v12 style: 1st=3, 2nd=2, 3rd=1, else=0)
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['target'] = df['rank'].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    
    return df

def get_features(df):
    """Extract feature columns (same as training pipeline)"""
    drop_cols = [
        'race_id', 'date', 'title', 'horse_id', 'horse_name',
        'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
        'rank', 'target', 'rank_str', 'year',
        'time', 'raw_time', 'passing_rank', 'last_3f',
        'odds', 'popularity', 'weight', 'weight_diff_val', 'weight_diff_sign',
        'winning_numbers', 'payout', 'ticket_type',
        'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'track_bias_disadvantage', 'outer_frame_disadv',
        'odds_race_rank', 'popularity_race_rank',
        'odds_deviation', 'popularity_deviation',
        'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
        'trend_win_front_rate', 'trend_win_fav_rate',
        'lag1_odds', 'lag1_popularity',
        'time_index', 'last_3f_index'
    ]
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(exclude=['object', 'datetime64'])
    
    # Force all features to float64 to avoid categorical mismatch in LightGBM
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes.astype('float64')
        elif X[col].dtype != 'float64':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype('float64')
    
    return X

def generate_oof_predictions(df, features, n_splits=5):
    """
    Generate Out-of-Fold predictions using TimeSeriesSplit by year.
    This avoids look-ahead bias while creating OOF predictions for all training data.
    """
    logger.info("Generating OOF predictions with Time Series CV...")
    
    # Get unique years
    years = sorted(df['year'].unique())
    logger.info(f"Years in data: {years}")
    
    # OOF container
    oof_preds = np.zeros(len(df))
    oof_mask = np.zeros(len(df), dtype=bool)
    
    # Use expanding window: train on years [start, year-1], validate on year
    for i, val_year in enumerate(years[1:], start=1):  # Skip first year (no train data)
        train_years = years[:i]
        
        train_mask = df['year'].isin(train_years)
        val_mask = df['year'] == val_year
        
        train_df = df[train_mask]
        val_df = df[val_mask]
        
        if len(train_df) < 1000 or len(val_df) < 100:
            logger.warning(f"Skipping fold {i}: train={len(train_df)}, val={len(val_df)}")
            continue
        
        logger.info(f"Fold {i}: Train years {train_years}, Val year {val_year}")
        
        X_train = get_features(train_df)
        y_train = train_df['target'].values
        q_train = train_df.groupby('race_id').size().values
        
        X_val = get_features(val_df)
        y_val = val_df['target'].values
        q_val = val_df.groupby('race_id').size().values
        
        # Align features
        for c in X_train.columns:
            if c not in X_val.columns:
                X_val[c] = 0
        X_val = X_val[X_train.columns]
        
        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
        lgb_val = lgb.Dataset(X_val, y_val, group=q_val, reference=lgb_train)
        
        # Train
        model = lgb.train(
            PARAMS,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Predict on validation fold
        preds = model.predict(X_val)
        oof_preds[val_mask] = preds
        oof_mask[val_mask] = True
        
    logger.info(f"OOF predictions generated for {oof_mask.sum()} samples")
    return oof_preds, oof_mask

def train_final_model(df, features):
    """Train final model on all available data (for production)"""
    logger.info("Training final production model on all data...")
    
    X = get_features(df)
    y = df['target'].values
    q = df.groupby('race_id').size().values
    
    lgb_train = lgb.Dataset(X, y, group=q)
    
    model = lgb.train(
        PARAMS,
        lgb_train,
        num_boost_round=200  # Fixed iterations based on OOF best
    )
    
    return model, X.columns.tolist()

def calibrate_probabilities(df, oof_preds, oof_mask):
    """
    Fit Isotonic Regression calibrator using OOF predictions.
    Maps softmax probability -> true win probability.
    """
    logger.info("Fitting Isotonic Regression calibrator...")
    
    # Filter to rows with OOF predictions
    cal_df = df[oof_mask].copy()
    cal_df['raw_score'] = oof_preds[oof_mask]
    
    # Convert to softmax probability within race
    cal_df['prob'] = cal_df.groupby('race_id')['raw_score'].transform(lambda x: softmax(x))
    
    # Target: Win (rank == 1)
    y_true = (cal_df['rank'] == 1).astype(int).values
    X_prob = cal_df['prob'].values
    
    # Fit calibrator
    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    calibrator.fit(X_prob, y_true)
    
    # Evaluate calibration
    calibrated_prob = calibrator.predict(X_prob)
    
    # Binning check
    df_res = pd.DataFrame({'prob': X_prob, 'calib_prob': calibrated_prob, 'win': y_true})
    df_res['bin'] = pd.cut(df_res['prob'], bins=np.linspace(0, 1, 11))
    grouped = df_res.groupby('bin', observed=True)[['win', 'calib_prob', 'prob']].mean()
    
    logger.info("\n--- Calibration Check ---")
    logger.info(f"\n{grouped}")
    
    return calibrator

def main():
    logger.info("=" * 60)
    logger.info("NAR Model Training v2 (JRA v13 Parity)")
    logger.info("=" * 60)
    
    # 1. Load Data
    df = load_data()
    logger.info(f"Loaded {len(df)} records, {df['race_id'].nunique()} races")
    
    # 2. Generate OOF Predictions
    features = get_features(df).columns.tolist()
    oof_preds, oof_mask = generate_oof_predictions(df, features)
    
    # 3. Train Final Model on All Data
    model, feature_names = train_final_model(df, features)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 4. Calibrate using OOF
    calibrator = calibrate_probabilities(df, oof_preds, oof_mask)
    
    # Save calibrator
    calibrator_path = os.path.join(MODEL_DIR, CALIBRATOR_NAME)
    joblib.dump(calibrator, calibrator_path)
    logger.info(f"Calibrator saved to {calibrator_path}")
    
    # 5. Save OOF predictions for analysis
    df['oof_pred'] = oof_preds
    df['has_oof'] = oof_mask
    oof_path = os.path.join(MODEL_DIR, OOF_PREDICTIONS_NAME)
    df[['race_id', 'horse_number', 'date', 'rank', 'oof_pred', 'has_oof']].to_parquet(oof_path)
    logger.info(f"OOF predictions saved to {oof_path}")
    
    # 6. Feature Importance
    logger.info("\nFeature Importance (Top 20):")
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    logger.info(f"\n{importance}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
