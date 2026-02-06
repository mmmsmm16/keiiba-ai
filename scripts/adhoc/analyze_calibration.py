import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
import logging
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
BINARY_MODEL_PATH = "models/binary_no_odds.pkl"
RESULTS_DIR = "reports/jra/diagnostics"

def analyze_calibration():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info("Loading model and data for calibration analysis...")
    bin_data = joblib.load(BINARY_MODEL_PATH)
    model = bin_data['model']
    feature_cols = bin_data['feature_cols']
    
    df = pd.read_parquet(CACHE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_top3'] = (df['rank'] <= 3).astype(int)
    
    # Test on late 2024 to 2025 (latest available labeled data)
    # We want to see how it performs on OOS data.
    # Since we re-trained up to 2025-12-14, let's look at that period's OOB if possible,
    # or just use 2024 data as a representative set.
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')].copy()
    
    if len(test_df) == 0:
        logger.error("No test data found for period.")
        return

    logger.info(f"Predicting on {len(test_df)} samples...")
    X = test_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype('float32')
    
    probs = model.predict_proba(X)[:, 1]
    y_true = test_df['target_top3'].values
    
    # 1. Standard Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
    
    print("\n=== Calibration Analysis (Decile-ish) ===")
    print(f"{'Bin':>5} | {'Pred Prob':>10} | {'Actual Rate':>10} | {'Count':>7}")
    print("-" * 45)
    
    # Custom Binning for Decile Analysis
    test_df['pred_prob'] = probs
    test_df['decile'] = pd.qcut(test_df['pred_prob'], 10, labels=False, duplicates='drop')
    
    decile_stats = test_df.groupby('decile').agg({
        'pred_prob': 'mean',
        'target_top3': ['mean', 'count']
    })
    
    for decile, row in decile_stats.iterrows():
        p_p = row[('pred_prob', 'mean')]
        a_r = row[('target_top3', 'mean')]
        cnt = row[('target_top3', 'count')]
        print(f"{decile:>5} | {p_p:10.1%} | {a_r:10.1%} | {int(cnt):>7}")

    # 2. Race-level Softmax Normalization Analysis
    # Does "RelScore" help in calibration within race?
    def calc_softmax(group):
        p = group['pred_prob'].values
        # Simple softmax on probabilities (scaling factor T=10 for better spread)
        exp_p = np.exp(p * 10)
        group['rel_prob'] = exp_p / exp_p.sum()
        return group
    
    test_df = test_df.groupby('race_id', group_keys=False).apply(calc_softmax)
    
    # Top 3 share within race should be related to actual finish
    # If the AI thinks 1 horse has 50% share, does it win 50% of time?
    test_df['rel_decile'] = pd.qcut(test_df['rel_prob'], 10, labels=False, duplicates='drop')
    rel_stats = test_df.groupby('rel_decile').agg({
        'rel_prob': 'mean',
        'target_top3': 'mean'
    })
    
    print("\n=== Race-level Relative Prob Analysis ===")
    print(f"{'Bin':>5} | {'Rel Prob':>10} | {'Actual Rate':>10}")
    print("-" * 35)
    for decile, row in rel_stats.iterrows():
        print(f"{decile:>5} | {row['rel_prob']:10.1%} | {row['target_top3']:10.1%}")

    logger.info("Analysis complete.")

if __name__ == "__main__":
    analyze_calibration()
