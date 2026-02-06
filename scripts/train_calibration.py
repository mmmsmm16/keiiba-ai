
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import logging
import argparse
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path, valid_year=2023):
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Ensure Date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
    else:
        logger.error("Data missing 'date' column.")
        return None, None

    # Sort
    df = df.sort_values(['date', 'race_id', 'horse_number'])
    
    # Split
    # We need Valid (for fitting calibrator) and Test (for evaluation)
    # Assuming the Base Model was trained on Train (< Valid).
    
    valid_df = df[df['year'] == valid_year].copy()
    test_df = df[df['year'] > valid_year].copy() # 2024+
    
    logger.info(f"Valid Set ({valid_year}): {len(valid_df)} samples")
    logger.info(f"Test Set (> {valid_year}): {len(test_df)} samples")
    
    return valid_df, test_df

def preprocess_for_model(df, model, cat_cols=[]):
    try:
        feature_names = model.feature_name()
    except:
        feature_names = model.booster_.feature_name()
        
    X = df[feature_names].copy()
    
    # Preprocess categorical
    for col in X.columns:
        if col in cat_cols or X[col].dtype == 'object' or X[col].dtype.name == 'category':
             # Cast to category then to codes
             X[col] = X[col].astype('category')
             # For numpy bypass, we MUST have numbers. 
             # We rely on alphabetical sorting being consistent with training.
             X[col] = X[col].cat.codes
             # cat.codes uses -1 for NaN. LightGBM usually handles NaN as special.
             # but -1 is a valid integer. 
             # Ideally we keep it as -1 or cast to NaN? 
             # Let's keep as int codes (float) but replace -1 with NaN for "unknown"?
             # Actually LightGBM trees split on values. -1 is just another value.
             pass
             
    # Ensure all numerical
    # Try to force float to check for remaining strings
    # (The caller does .astype(np.float32) which will fail if strings remain)
    return X

def train_calibrator(model, df_valid, target_col, cat_cols):
    X_valid = preprocess_for_model(df_valid, model, cat_cols)
    y_valid = df_valid[target_col].values
    
    logger.info("Predicting validation scores...")
    try:
        # Try predict_proba (sklearn wrapper) or predict (booster)
        if hasattr(model, "predict_proba"):
             raw_preds = model.predict_proba(X_valid)[:, 1]
        else:
             raw_preds = model.predict(X_valid)
    except Exception as e:
        logger.warning(f"Prediction failed with normal input: {e}. Trying numpy bypass...")
        # Numpy bypass
        X_vals = X_valid.values.astype(np.float32)
        raw_preds = model.predict(X_vals)

    # Isotonic Regression requires 1D X
    logger.info("Fitting Isotonic Regression...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_preds, y_valid)
    
    return iso, raw_preds

def evaluate_calibration(iso, model, df_test, target_col, cat_cols, label="Test"):
    X_test = preprocess_for_model(df_test, model, cat_cols)
    y_test = df_test[target_col].values
    
    try:
        if hasattr(model, "predict_proba"):
             raw_preds = model.predict_proba(X_test)[:, 1]
        else:
             raw_preds = model.predict(X_test)
    except:
        X_vals = X_test.values.astype(np.float32)
        raw_preds = model.predict(X_vals)
        
    calibrated_preds = iso.transform(raw_preds)
    
    # ECE Calculation
    def calc_ece(probs, y_true, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='quantile')
        # Weighted average of diff
        # We need bin counts. sklearn calibration_curve doesn't return counts easily?
        # Let's do manual or approx.
        # Actually sklearn doesn't return weights. 
        # But we can approximate ECE by mean absolute error between observed and predicted in bins.
        return np.mean(np.abs(prob_true - prob_pred)) 
        
    # Valid ECE
    if raw_preds.min() < 0 or raw_preds.max() > 1:
        logger.info("Raw predictions outside [0,1] (Ranking Score). Skipping Raw ECE.")
        ece_raw = -1
        brier_raw = -1
    else:
        ece_raw = calc_ece(raw_preds, y_test)
        brier_raw = brier_score_loss(y_test, raw_preds)
        
    ece_calib = calc_ece(calibrated_preds, y_test)
    brier_calib = brier_score_loss(y_test, calibrated_preds)
    
    logger.info(f"[{label}] Calibration Results:")
    logger.info(f"  ECE: {ece_raw:.4f} -> {ece_calib:.4f}")
    logger.info(f"  Brier: {brier_raw:.4f} -> {brier_calib:.4f}")
    logger.info(f"  Mean Prob: {raw_preds.mean():.4f} -> {calibrated_preds.mean():.4f} (True: {y_test.mean():.4f})")
    
    return calibrated_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to base model pkl")
    parser.add_argument("--data_path", default="data/processed/preprocessed_data_v11.parquet")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_col", default="rank <= 3") # Special syntax
    # e.g. "rank == 1", "rank <= 3"
    parser.add_argument("--output_name", default="calibrator.pkl")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
        
    # Load Model
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return

    logger.info(f"Loading Model: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = joblib.load(f)
        
    # Identify Categoricals from Config if possible
    # (Same logic as before)
    cat_cols = []
    config_path = os.path.join(os.path.dirname(args.model_path), "config_copy.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cat_cols = cfg['dataset'].get('categorical_features', [])
    
    # Load Data
    valid_df, test_df = load_data(args.data_path)
    if valid_df is None: return

    # Parse Target
    # We need binary target for Isotonic
    logger.info(f"Target Expression: {args.target_col}")
    
    def apply_target(df, expr):
        if "rank" in expr:
            # safe eval?
            # expr like "rank <= 3"
            # df.query(expr) returns subset? No we want binary vector
            # use pd.eval
            # But we want 0/1.
            try:
                # Create mask
                mask = df.eval(expr)
                return mask.astype(int)
            except:
                logger.error(f"Failed to eval target: {expr}")
                return None
        return None

    valid_df['target_bin'] = apply_target(valid_df, args.target_col)
    test_df['target_bin'] = apply_target(test_df, args.target_col)
    
    if valid_df['target_bin'] is None: return
    
    # Train
    iso, _ = train_calibrator(model, valid_df, 'target_bin', cat_cols)
    
    # Evaluate
    evaluate_calibration(iso, model, test_df, 'target_bin', cat_cols, label="Test 2024")
    
    # Save
    save_path = os.path.join(args.output_dir, args.output_name)
    joblib.dump(iso, save_path)
    logger.info(f"Calibrator saved to {save_path}")

if __name__ == "__main__":
    main()
