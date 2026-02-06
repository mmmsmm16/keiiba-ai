
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet" # Contains rank for top2/top3 logic
MODELS_DIR = "models/experiments"
EXPERIMENTS = {
    "Win": "exp_t2_refined_v3",
    "Top2": "exp_t2_refined_v3_top2",
    "Top3": "exp_t2_refined_v3_top3"
}

def load_data():
    logger.info(f"Loading features from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    logger.info(f"Loading targets from {TARGET_PATH}...")
    try:
        targets = pd.read_parquet(TARGET_PATH)
        # Ensure index alignment if needed, or simple merge
        # Assuming both have race_id + horse_number as index or columns
        # But run_optuna_hpo.py merged on index?
        # Let's check common columns.
        common_cols = ['race_id', 'horse_number']
        if not all(col in df.columns for col in common_cols):
            # If index usage, reset
            pass
        
        # Merge targets to get accurate 'rank' if missing in v11
        if 'rank' not in df.columns:
             df = df.merge(targets[['race_id', 'horse_number', 'rank']], on=['race_id', 'horse_number'], how='left')
    except Exception as e:
        logger.warning(f"Could not load T2 targets: {e}. Using existing 'rank' column if available.")
    
    # Filter for Test Period (2024+)
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'].dt.year >= 2024].copy()
    logger.info(f"Test Set (2024+): {len(test_df)} samples")
    return test_df

def evaluate_model(name, exp_subdir, test_df):
    model_dir = os.path.join(MODELS_DIR, exp_subdir)
    # Find model file
    model_files = glob.glob(os.path.join(model_dir, "model.pkl"))
    if not model_files:
        # Fallback to txt
        model_files = glob.glob(os.path.join(model_dir, "model_*.txt"))
        
    if not model_files:
        logger.error(f"[{name}] No model file found in {model_dir}")
        return None
    
    model_path = model_files[0]
    logger.info(f"[{name}] Loading model from {model_path}...")
    
    import joblib
    if model_path.endswith('.pkl'):
        bst = joblib.load(model_path)
    else:
        bst = lgb.Booster(model_file=model_path)
    
    # Prepare features
    # Need to match features used in training.
    # LightGBM handles extra features ignoral, but missing features is bad.
    # Ideally reuse logic from train script, but here we assume features are present.
    # We should exclude non-feature columns.
    
    feature_names = bst.feature_name()
    
    # Check missing
    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        logger.warning(f"[{name}] Missing features: {missing[:5]}...")
        # Add dummies?
        for f in missing:
            test_df[f] = 0
            
    # Define X_test
    X_test = test_df[feature_names].copy()
    
    # Load config to get categorical features
    import yaml
    config_path = os.path.join(model_dir, "config.yaml")
    cat_cols = []
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            cat_cols = cfg.get('dataset', {}).get('categorical_features', [])
    
    # Cast categorical columns
    for c in cat_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype('category')
            
    # Also cast object columns just in case
    for c in X_test.columns:
        if X_test[c].dtype == 'object':
            X_test[c] = X_test[c].astype('category')
    
    # Predict with DataFrame (preserves categorical info for LightGBM)
    logger.info(f"[{name}] Predicting...")
    preds = bst.predict(X_test)
    
    # Define Target
    if name == "Win":
        y_true = (test_df['rank'] == 1).astype(int)
    elif name == "Top2":
        y_true = (test_df['rank'] <= 2).astype(int)
    elif name == "Top3":
        y_true = (test_df['rank'] <= 3).astype(int)
        
    # Calculate AUC
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Filter out NaNs in y_true (scratched horses?) -> rank is usually present or NaN
    mask = ~y_true.isna()
    if (~mask).any():
        logger.info(f"Dropping {sum(~mask)} NaN targets")
        y_true = y_true[mask]
        preds = preds[mask]
        
    auc = roc_auc_score(y_true, preds)
    logger.info(f"[{name}] Test AUC: {auc:.4f}")
    
    return auc

def main():
    test_df = load_data()
    
    results = {}
    for name, subdir in EXPERIMENTS.items():
        auc = evaluate_model(name, subdir, test_df)
        if auc is not None:
            results[name] = auc
            
    print("\n=== Final Verification Results (Test 2024+) ===")
    print(f"{'Model':<10} | {'AUC':<10}")
    print("-" * 25)
    for name, auc in results.items():
        print(f"{name:<10} | {auc:.4f}")
        
    # Check hypothesis
    if len(results) == 3:
        pass 
        # Manual check by user

if __name__ == "__main__":
    main()
