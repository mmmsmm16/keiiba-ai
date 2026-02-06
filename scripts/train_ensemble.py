
import pandas as pd
import numpy as np
import joblib
import logging
import argparse
import os
import yaml
from scipy.optimize import minimize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS = {
    "win": {
        "path": "models/experiments/optuna_best_full/model.pkl",
        "calib": "models/experiments/optuna_best_full/calibrator.pkl"
    },
    "top3": {
        "path": "models/experiments/exp_t2_refined_v3_top3/model.pkl",
        "calib": "models/experiments/exp_t2_refined_v3_top3/calibrator.pkl"
    },
    "rank": {
        "path": "models/experiments/exp_lambdarank/model.pkl",
        "calib": "models/experiments/exp_lambdarank/calibrator.pkl"
    }
}

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def load_model_and_calibrator(key):
    info = MODELS[key]
    with open(info['path'], 'rb') as f:
        model = joblib.load(f)
    with open(info['calib'], 'rb') as f:
        calib = joblib.load(f)
    
    # Config for cats
    config_path = os.path.join(os.path.dirname(info['path']), "config_copy.yaml")
    cat_cols = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cat_cols = cfg['dataset'].get('categorical_features', [])
        
    return model, calib, cat_cols

def predict_calibrated(model, calib, df, cat_cols):
    # Preprocess
    try:
        feature_names = model.feature_name()
    except:
        feature_names = model.booster_.feature_name()
    
    X = df[feature_names].copy()
    for col in X.columns:
        if col in cat_cols or X[col].dtype == 'object' or X[col].dtype.name == 'category':
             X[col] = X[col].astype('category').cat.codes
             
    try:
        if hasattr(model, "predict_proba"):
             raw = model.predict_proba(X)[:, 1]
        else:
             raw = model.predict(X)
    except:
        X_vals = X.values.astype(np.float32)
        raw = model.predict(X_vals)
        
    # Calibrate
    return calib.transform(raw)

def main():
    logger.info("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
    df = df.sort_values(['date', 'race_id', 'horse_number'])
    
    # We optimize on Validation Set (2023) to find weights
    # Then verify on Test Set (2024+)
    valid_df = df[df['year'] == 2023].copy()
    test_df = df[df['year'] >= 2024].copy()
    
    logger.info(f"Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Generate Predictions
    preds_valid = {}
    preds_test = {}
    
    for key in MODELS:
        logger.info(f"Predicting {key}...")
        model, calib, cats = load_model_and_calibrator(key)
        preds_valid[key] = predict_calibrated(model, calib, valid_df, cats)
        preds_test[key] = predict_calibrated(model, calib, test_df, cats)
        
    y_valid = (valid_df['rank'] <= 3).astype(int).values # Target: Place (Top3)
    # Note: Win Model (calibrated for Win) will be used for Place? 
    # Actually, calibrated Win Prob is prob of Rank 1.
    # Calibrated Top3 Prob is prob of Rank 1-3.
    # Calibrated Rank Prob is prob of Rank 1-3 (if calibrated on rank<=3).
    # We should combine them to predict Rank 1-3.
    # Win Prob is highly correlated but lower magnitude.
    
    # Optimization Objective: LogLoss (minimize)
    from sklearn.metrics import log_loss
    
    def blend_loss(weights):
        # Normalize
        w = np.array(weights)
        w = w / w.sum()
        
        # Blend
        p_blend = (w[0] * preds_valid['win'] + 
                   w[1] * preds_valid['top3'] + 
                   w[2] * preds_valid['rank'])
        
        return log_loss(y_valid, p_blend)
        
    # Initial weights
    init_w = [0.2, 0.6, 0.2] # bias towards top3 model
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    logger.info("Optimizing Ensemble Weights...")
    res = minimize(blend_loss, init_w, bounds=bounds, method='SLSQP')
    
    best_w = res.x / res.x.sum()
    logger.info(f"Optimal Weights: Win={best_w[0]:.4f}, Top3={best_w[1]:.4f}, Rank={best_w[2]:.4f}")
    logger.info(f"Valid Loss: {res.fun:.5f}")
    
    # Evaluation on Test
    p_test_blend = (best_w[0] * preds_test['win'] + 
                    best_w[1] * preds_test['top3'] + 
                    best_w[2] * preds_test['rank'])
                    
    y_test = (test_df['rank'] <= 3).astype(int).values
    test_loss = log_loss(y_test, p_test_blend)
    logger.info(f"Test Loss (Ensemble): {test_loss:.5f}")
    
    # Compare with single models
    loss_win = log_loss(y_test, preds_test['win'])
    loss_top3 = log_loss(y_test, preds_test['top3'])
    loss_rank = log_loss(y_test, preds_test['rank'])
    
    logger.info(f"Test Loss (Win): {loss_win:.5f}")
    logger.info(f"Test Loss (Top3): {loss_top3:.5f}")
    logger.info(f"Test Loss (Rank): {loss_rank:.5f}")
    
    # Save Weights
    import json
    weights = {
        "win": float(best_w[0]),
        "top3": float(best_w[1]),
        "rank": float(best_w[2])
    }
    with open("models/ensemble_weights_place.json", "w") as f:
        json.dump(weights, f)
        
if __name__ == "__main__":
    main()
