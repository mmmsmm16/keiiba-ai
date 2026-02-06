"""
Deep dive into jockey/trainer stats drift
==========================================
Analyze WHY jockey_top3_rate_365d_relative_z and trainer_top3_rate_365d_relative_z are drifting.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Deep Dive: Jockey/Trainer Stats Drift Analysis")
    logger.info("=" * 60)
    
    FEAT_PATH = "data/features/temp_merge_current.parquet"
    
    df = pd.read_parquet(FEAT_PATH)
    df['race_id'] = df['race_id'].astype(str)
    df['year'] = df['race_id'].str[:4].astype(int)
    
    df_2024 = df[df['year'] == 2024]
    df_2025 = df[df['year'] == 2025]
    
    # Key features to analyze
    features = [
        'jockey_top3_rate_365d',  # Raw rate
        'jockey_top3_rate_365d_relative_z',  # Z-scored
        'trainer_top3_rate_365d',  # Raw rate
        'trainer_top3_rate_365d_relative_z',  # Z-scored
        'jockey_win_rate_365d',
        'jockey_win_rate_365d_relative_z',
    ]
    
    logger.info("\n[1] Raw vs Z-Score Distribution Comparison")
    logger.info("-" * 60)
    
    for feat in features:
        if feat not in df.columns:
            continue
        v_24 = pd.to_numeric(df_2024[feat], errors='coerce').dropna()
        v_25 = pd.to_numeric(df_2025[feat], errors='coerce').dropna()
        
        print(f"\n{feat}:")
        print(f"  2024: Mean={v_24.mean():.4f}, Std={v_24.std():.4f}, Q95={v_24.quantile(0.95):.4f}")
        print(f"  2025: Mean={v_25.mean():.4f}, Std={v_25.std():.4f}, Q95={v_25.quantile(0.95):.4f}")
        print(f"  Shift: {(v_25.mean() - v_24.mean()) / v_24.std() * 100:+.1f}% of 2024 std")
    
    # Check if specific jockeys are causing drift
    logger.info("\n\n[2] Top Jockeys Comparison")
    logger.info("-" * 60)
    
    if 'kishu_code' in df.columns or 'jockey_code' in df.columns:
        jockey_col = 'kishu_code' if 'kishu_code' in df.columns else 'jockey_code'
        
        # Top 20 jockeys by race count in 2024
        top_jockeys_24 = df_2024.groupby(jockey_col).size().nlargest(20).index.tolist()
        
        for jc in top_jockeys_24[:5]:
            j24 = df_2024[df_2024[jockey_col] == jc]['jockey_top3_rate_365d']
            j25 = df_2025[df_2025[jockey_col] == jc]['jockey_top3_rate_365d']
            
            if len(j24) > 10 and len(j25) > 10:
                print(f"\nJockey {jc}:")
                print(f"  2024: Mean={pd.to_numeric(j24, errors='coerce').mean():.4f} (n={len(j24)})")
                print(f"  2025: Mean={pd.to_numeric(j25, errors='coerce').mean():.4f} (n={len(j25)})")
    
    # Check if the z-score baseline is the problem
    logger.info("\n\n[3] Z-Score Calibration Check")
    logger.info("-" * 60)
    
    z_feat = 'jockey_top3_rate_365d_relative_z'
    if z_feat in df.columns:
        z_24 = pd.to_numeric(df_2024[z_feat], errors='coerce').dropna()
        z_25 = pd.to_numeric(df_2025[z_feat], errors='coerce').dropna()
        
        print(f"\n{z_feat}:")
        print(f"  2024: Mean={z_24.mean():.4f} (should be ~0 if properly calibrated)")
        print(f"  2025: Mean={z_25.mean():.4f}")
        print(f"\n  Interpretation:")
        if z_25.mean() > z_24.mean() + 0.1:
            print("    → 2025 jockeys have HIGHER performance relative to historical baseline")
            print("    → Model was trained expecting z~0, but sees z~+0.1 in 2025")
            print("    → This causes model to OVERESTIMATE win probability")
        elif z_25.mean() < z_24.mean() - 0.1:
            print("    → 2025 jockeys have LOWER performance relative to historical baseline")
        else:
            print("    → No significant calibration drift")
    
    # Suggest fix
    logger.info("\n\n[4] ROOT CAUSE HYPOTHESIS")
    logger.info("=" * 60)
    print("""
The _relative_z features use rolling historical statistics to normalize.

PROBLEM: 
- 2024 z-scores average ~-0.28 (below rolling mean)
- 2025 z-scores average ~-0.19 (closer to rolling mean)

This means in 2025, the CURRENT jockey/trainer performance is relatively 
BETTER compared to the historical rolling average than it was in 2024.

WHY THIS HURTS THE MODEL:
- Model learned: "negative z-score = below average = lower win chance"
- But 2025 z-scores are less negative → model thinks higher win chance
- Actual win rate may not have increased proportionally

POSSIBLE FIXES:
1. Use race-level z-score instead of global rolling
2. Use fixed historical mean/std for normalization (not rolling)
3. Re-normalize z-scores at prediction time using 2024 statistics
4. Remove _relative_z features and use raw rates only
5. Add feature: year-specific adjustment term
""")

if __name__ == "__main__":
    main()
