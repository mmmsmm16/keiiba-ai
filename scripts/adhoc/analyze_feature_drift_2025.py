"""
2025 Feature Drift Root Cause Analysis
======================================
Comprehensive analysis to identify which features are drifting and causing model degradation.

Steps:
1. Load feature importance from model
2. For top-N important features, compare 2024 vs 2025 distributions
3. Calculate drift metrics (mean shift, std change, KS statistic)
4. Correlate drifted features with prediction error
5. Output ranked list of problematic features
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import joblib
from scipy import stats

sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_model_importance(model_path):
    """Load feature importance from model."""
    model = joblib.load(model_path)
    if hasattr(model, 'booster_'):
        booster = model.booster_
    else:
        booster = model
    
    feature_names = booster.feature_name()
    importance = booster.feature_importance(importance_type='gain')
    
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return imp_df

def calculate_drift_metrics(s_2024, s_2025, feature_name):
    """Calculate various drift metrics between two series."""
    # Handle non-numeric
    if not np.issubdtype(s_2024.dtype, np.number):
        try:
            s_2024 = pd.to_numeric(s_2024, errors='coerce')
            s_2025 = pd.to_numeric(s_2025, errors='coerce')
        except:
            return None
    
    # Drop NaN
    s_2024 = s_2024.dropna()
    s_2025 = s_2025.dropna()
    
    if len(s_2024) < 100 or len(s_2025) < 100:
        return None
    
    # Basic stats
    mean_24, mean_25 = s_2024.mean(), s_2025.mean()
    std_24, std_25 = s_2024.std(), s_2025.std()
    
    # Relative mean shift (normalized by 2024 std)
    if std_24 > 0:
        mean_shift_z = (mean_25 - mean_24) / std_24
    else:
        mean_shift_z = 0
    
    # Absolute mean shift percentage
    if abs(mean_24) > 1e-6:
        mean_shift_pct = (mean_25 - mean_24) / abs(mean_24) * 100
    else:
        mean_shift_pct = 0
    
    # KS test (distribution difference)
    try:
        ks_stat, ks_pval = stats.ks_2samp(s_2024, s_2025)
    except:
        ks_stat, ks_pval = 0, 1
    
    # Quantile shifts
    q95_24 = np.percentile(s_2024, 95)
    q95_25 = np.percentile(s_2025, 95)
    q5_24 = np.percentile(s_2024, 5)
    q5_25 = np.percentile(s_2025, 5)
    
    return {
        'feature': feature_name,
        'mean_24': mean_24,
        'mean_25': mean_25,
        'mean_shift_z': mean_shift_z,
        'mean_shift_pct': mean_shift_pct,
        'std_24': std_24,
        'std_25': std_25,
        'std_ratio': std_25 / std_24 if std_24 > 0 else 1,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'q95_24': q95_24,
        'q95_25': q95_25,
        'q95_shift_pct': (q95_25 - q95_24) / abs(q95_24) * 100 if abs(q95_24) > 1e-6 else 0,
        'q5_24': q5_24,
        'q5_25': q5_25,
    }

def analyze_prediction_error_correlation(df, drifted_features, pred_col='pred_prob', target_col='is_win'):
    """Analyze which features correlate with prediction errors."""
    if target_col not in df.columns:
        logger.warning(f"{target_col} not in df, skipping error correlation")
        return None
    
    # Calculate prediction error (absolute)
    df['pred_error'] = np.abs(df[pred_col] - df[target_col])
    
    correlations = []
    for feat in drifted_features:
        if feat in df.columns:
            try:
                corr = df[[feat, 'pred_error']].corr().iloc[0, 1]
                correlations.append({'feature': feat, 'error_corr': corr})
            except:
                pass
    
    return pd.DataFrame(correlations)

def main():
    logger.info("=" * 60)
    logger.info("2025 Feature Drift Root Cause Analysis")
    logger.info("=" * 60)
    
    # Paths
    MODEL_PATH = "models/experiments/exp_t2_refined_v3_2025/model.pkl"
    FEAT_PATH = "data/features/temp_merge_current.parquet"
    PRED_PATH_25 = "data/temp_t2/T2_predictions_2025_walkforward.parquet"
    
    # Step 1: Load feature importance
    logger.info("\n[Step 1] Loading model feature importance...")
    imp_df = load_model_importance(MODEL_PATH)
    top_features = imp_df.head(50)['feature'].tolist()
    logger.info(f"  Top 50 features loaded. #1: {top_features[0]}")
    
    # Step 2: Load feature data
    logger.info("\n[Step 2] Loading feature data...")
    df = pd.read_parquet(FEAT_PATH)
    df['race_id'] = df['race_id'].astype(str)
    df['year'] = df['race_id'].str[:4].astype(int)
    
    df_2024 = df[df['year'] == 2024]
    df_2025 = df[df['year'] == 2025]
    logger.info(f"  2024: {len(df_2024)} rows, 2025: {len(df_2025)} rows")
    
    # Step 3: Calculate drift metrics for top features
    logger.info("\n[Step 3] Calculating drift metrics for top 50 features...")
    drift_results = []
    
    for feat in top_features:
        if feat not in df.columns:
            continue
        
        metrics = calculate_drift_metrics(df_2024[feat], df_2025[feat], feat)
        if metrics:
            # Add importance
            metrics['importance'] = imp_df[imp_df['feature'] == feat]['importance'].values[0]
            drift_results.append(metrics)
    
    drift_df = pd.DataFrame(drift_results)
    
    # Step 4: Rank by drift severity (combine importance + drift magnitude)
    logger.info("\n[Step 4] Ranking features by drift severity...")
    
    # Normalize metrics
    drift_df['importance_norm'] = drift_df['importance'] / drift_df['importance'].max()
    drift_df['ks_norm'] = drift_df['ks_stat']  # Already 0-1
    drift_df['mean_shift_norm'] = np.abs(drift_df['mean_shift_z']) / np.abs(drift_df['mean_shift_z']).max()
    
    # Combined score: High importance + High drift = High risk
    drift_df['risk_score'] = (
        drift_df['importance_norm'] * 0.4 + 
        drift_df['ks_norm'] * 0.3 + 
        drift_df['mean_shift_norm'] * 0.3
    )
    
    drift_df = drift_df.sort_values('risk_score', ascending=False)
    
    # Step 5: Output results
    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 DRIFTED FEATURES BY RISK SCORE")
    logger.info("=" * 60)
    
    output_cols = ['feature', 'importance', 'mean_24', 'mean_25', 'mean_shift_pct', 
                   'ks_stat', 'risk_score']
    
    print("\n")
    print(drift_df[output_cols].head(20).to_string(index=False))
    
    # Step 6: Detailed analysis of top 5 risky features
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED ANALYSIS: TOP 5 DRIFTED FEATURES")
    logger.info("=" * 60)
    
    for idx, row in drift_df.head(5).iterrows():
        feat = row['feature']
        print(f"\n### {feat}")
        print(f"  Importance Rank: #{list(imp_df['feature']).index(feat) + 1}")
        print(f"  Mean: {row['mean_24']:.4f} -> {row['mean_25']:.4f} ({row['mean_shift_pct']:+.1f}%)")
        print(f"  Std:  {row['std_24']:.4f} -> {row['std_25']:.4f} (ratio: {row['std_ratio']:.2f})")
        print(f"  Q95:  {row['q95_24']:.4f} -> {row['q95_25']:.4f} ({row['q95_shift_pct']:+.1f}%)")
        print(f"  KS Statistic: {row['ks_stat']:.4f} (p={row['ks_pval']:.2e})")
        print(f"  Risk Score: {row['risk_score']:.4f}")
    
    # Step 7: Monthly trend for worst feature
    logger.info("\n" + "=" * 60)
    logger.info("MONTHLY TREND: TOP DRIFTED FEATURE")
    logger.info("=" * 60)
    
    worst_feat = drift_df.iloc[0]['feature']
    df['month'] = df['race_id'].str[4:6].astype(int)
    
    # Ensure numeric
    df[worst_feat] = pd.to_numeric(df[worst_feat], errors='coerce')
    
    monthly = df.groupby(['year', 'month'])[worst_feat].mean().reset_index()
    monthly_pivot = monthly.pivot(index='month', columns='year', values=worst_feat)

    
    print(f"\n{worst_feat} - Monthly Mean by Year:")
    print(monthly_pivot.to_string())
    
    # Save results
    output_path = "reports/feature_drift_analysis_2025.csv"
    drift_df.to_csv(output_path, index=False)
    logger.info(f"\nFull results saved to: {output_path}")
    
    # Step 8: Suggest fixes
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    top3_risky = drift_df.head(3)['feature'].tolist()
    print(f"\nTop 3 problematic features: {top3_risky}")
    print("\nPossible fixes:")
    print("  1. Apply z-score normalization using 2024 statistics")
    print("  2. Add feature stability regularization during training")
    print("  3. Remove or cap extreme values in drifted features")
    print("  4. Use rolling window for temporal features to reduce drift")

if __name__ == "__main__":
    main()
