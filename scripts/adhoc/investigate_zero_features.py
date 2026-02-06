"""
Investigate Zero-Importance Features
====================================
Analyzes why certain features have zero or near-zero importance.
Checks:
1. Null Rate
2. Variance (is it constant?)
3. Correlation with Top Features (is it redundant?)
4. Skewness / Distribution

Usage:
  python scripts/adhoc/investigate_zero_features.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"

# Features to investigate (from SHAP output)
ZERO_FEATURES = [
    'is_sole_leader', 'struct_nige_count', 'last_nige_rate', 'avg_first_corner_norm',
    'sire_heavy_win_rate', 'is_first_rot', 'is_dist_short_nige', 'is_first_str',
    'apt_slp_win_rate', 'is_first_slp', 'lap_fit_interaction', 'pace_expectation_proxy',
    'fit_sashi_long', 'relative_3f_score', 'is_first_combo', 'struct_early_speed_mean',
    'struct_early_speed_sum', 'is_high_pace_warn', 'corner_acceleration',
    'avg_final_corner_pct', 'corner_advance_score', 'is_long_rest', 'front_runner_count',
    'race_pace_level_3f', 'relative_track_variant_z'
]

# Top features to check correlation against (for redundancy check)
TOP_FEATURES = [
    'grade_code', 'age', 'lag1_last_3f', 'relative_horse_elo_z', 'field_elo_mean'
]

def analyze_features():
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter to recent data (2023-2024) to reflect training/test context
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2023].copy()
    
    logger.info(f"Analyzing {len(df)} records from 2023-2024...")
    
    print("\n" + "=" * 80)
    print(f"{'Feature':<30} | {'Null%':<8} | {'Unique':<8} | {'Mean':<8} | {'Std':<8} | {'Max Corr (Feature)'}")
    print("-" * 80)
    
    # Prepare top features for correlation
    top_df = df[TOP_FEATURES].copy()
    for c in top_df.columns:
        if top_df[c].dtype == 'object':
            top_df[c] = top_df[c].astype('category').cat.codes
    top_df = top_df.fillna(0) # Simple fill for correlation check
    
    for feat in ZERO_FEATURES:
        if feat not in df.columns:
            print(f"{feat:<30} | NOT FOUND")
            continue
            
        series = df[feat]
        
        # 1. Null Rate
        null_rate = series.isnull().mean() * 100
        
        # 2. Unique Values
        n_unique = series.nunique()
        
        # 3. Stats (numeric only)
        try:
            val_mean = series.mean()
            val_std = series.std()
        except:
            val_mean = np.nan
            val_std = np.nan
            
        # 4. Max Correlation
        if pd.api.types.is_numeric_dtype(series):
            # Fill NA for correlation
            s_filled = series.fillna(0) # Assumption
            corrs = top_df.corrwith(s_filled)
            max_corr = corrs.abs().max()
            max_corr_feat = corrs.abs().idxmax()
            corr_str = f"{max_corr:.2f} ({max_corr_feat})"
        else:
            corr_str = "N/A (Cat)"
            
        print(f"{feat:<30} | {null_rate:>6.1f}% | {n_unique:>8} | {val_mean:>8.3f} | {val_std:>8.3f} | {corr_str}")

    print("-" * 80)

if __name__ == "__main__":
    analyze_features()
