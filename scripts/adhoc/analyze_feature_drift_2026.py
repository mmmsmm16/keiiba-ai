
import sys
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_drift_metrics(s_old, s_new, feature_name):
    """Calculate drift metrics between two series."""
    # Ensure numeric
    s_old = pd.to_numeric(s_old, errors='coerce')
    s_new = pd.to_numeric(s_new, errors='coerce')
    
    # Null info
    null_old = s_old.isnull().mean()
    null_new = s_new.isnull().mean()
    null_diff = null_new - null_old
    
    # Drop NaN for distribution analysis
    s_old_clean = s_old.dropna()
    s_new_clean = s_new.dropna()
    
    mean_old, mean_new = s_old_clean.mean(), s_new_clean.mean()
    std_old = s_old_clean.std()
    
    # Drift Score (Z-score shift)
    drift_score = 0.0
    if std_old > 0 and not np.isnan(std_old):
        drift_score = (mean_new - mean_old) / std_old
        
    # KS test
    ks_stat, ks_pval = 0.0, 1.0
    if len(s_old_clean) > 50 and len(s_new_clean) > 50:
        try:
            ks_stat, ks_pval = stats.ks_2samp(s_old_clean, s_new_clean)
        except:
            pass
            
    return {
        'feature': feature_name,
        'null_2025': null_old,
        'null_2026': null_new,
        'null_diff': null_diff,
        'mean_2025': mean_old,
        'mean_2026': mean_new,
        'drift_score': drift_score,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval
    }

def main():
    FEAT_PATH_DIR = "data/features_v14/prod_cache"
    V13_FEATS_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'
    V14_FEATS_PATH = 'models/experiments/exp_gap_v14_production/features.csv'
    OUTPUT_CSV = "reports/feature_drift_2025_2026.csv"
    
    if not os.path.exists(FEAT_PATH_DIR):
        logger.error(f"Feature directory not found: {FEAT_PATH_DIR}")
        return

    logger.info(f"Loading features from {FEAT_PATH_DIR}...")
    # Base attributes to get years and venue
    df_base = pd.read_parquet(os.path.join(FEAT_PATH_DIR, "base_attributes.parquet"))
    df_base['race_id'] = df_base['race_id'].astype(str)
    df_base['year'] = df_base['race_id'].str[:4].astype(int)
    
    # [Refinement] JRA-only venues (01-10) to eliminate Venue Drift
    # PC-KEIBA venue codes: 01-10 are JRA. 30+ are NAR.
    is_jra = df_base['venue'].isin([str(i).zfill(2) for i in range(1, 11)])
    
    mask_2025 = (df_base['year'] == 2025) & is_jra
    mask_2026 = (df_base['year'] == 2026) & is_jra
    
    import glob
    all_files = glob.glob(os.path.join(FEAT_PATH_DIR, "*.parquet"))
    
    # Initialize with keys
    df = df_base[['race_id', 'horse_number', 'year']].copy()
    df['race_id'] = df['race_id'].astype(str)
    df['horse_number'] = df['horse_number'].astype(int)
    
    block_dfs = []
    for f in all_files:
        fname = os.path.basename(f)
        if fname == "base_attributes.parquet": continue
        try:
            df_block = pd.read_parquet(f)
            df_block['race_id'] = df_block['race_id'].astype(str)
            df_block['horse_number'] = df_block['horse_number'].astype(int)
            
            # Non-key columns
            cols_to_keep = ['race_id', 'horse_number'] + [c for c in df_block.columns if c not in ['race_id', 'horse_number', 'horse_id', 'date']] 
            block_dfs.append(df_block[cols_to_keep])
        except Exception as e:
            logger.error(f"Error loading {fname}: {e}")

    # Optimized merging
    for bdf in block_dfs:
        df = pd.merge(df, bdf, on=['race_id', 'horse_number'], how='left')

    # De-fragment
    df = df.copy()

    df_2025 = df[mask_2025].copy()
    df_2026 = df[mask_2026].copy()
    
    logger.info(f"2025 (JRA) records: {len(df_2025)}")
    logger.info(f"2026 (JRA) records: {len(df_2026)}")
    
    if len(df_2026) == 0:
        logger.error("No 2026 JRA data found in the feature file!")
        return

    # Load feature lists
    try:
        if os.path.getsize(V13_FEATS_PATH) > 0:
            v13_feats = pd.read_csv(V13_FEATS_PATH, header=None)[0].tolist()
            if len(v13_feats) > 0 and (v13_feats[0] == '0' or v13_feats[0] == 'feature'): v13_feats = v13_feats[1:]
        else:
            v13_feats = []
        
        v14_data = pd.read_csv(V14_FEATS_PATH)
        if 'feature' in v14_data.columns:
            v14_feats = v14_data['feature'].tolist()
        else:
            v14_feats = v14_data.iloc[:, 0].tolist()
    except Exception as e:
        logger.error(f"Failed to load feature lists: {e}")
        v13_feats, v14_feats = [], []

    all_target_feats = sorted(list(set(v13_feats + v14_feats)))
    logger.info(f"Analyzing {len(all_target_feats)} unique target features.")
    
    results = []
    for feat in all_target_feats:
        if feat not in df.columns:
            logger.warning(f"Feature '{feat}' not found in parquet.")
            continue
        
        s_25 = df_2025[feat]
        s_26 = df_2026[feat]
        
        # [Refinement] Handle common dummy values for certain features
        if 'weight' in feat:
            s_25 = s_25.replace([0, 999, -999], np.nan)
            s_26 = s_26.replace([0, 999, -999], np.nan)
        
        metrics = calculate_drift_metrics(s_25, s_26, feat)
        metrics['is_v13'] = feat in v13_feats
        metrics['is_v14'] = feat in v14_feats
        results.append(metrics)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results saved to {OUTPUT_CSV}")
    
    # Analysis 1: Huge Null Increase
    logger.info("\n--- TOP NULL INCREASES (2025 JRA -> 2026 JRA) ---")
    print(df_res.sort_values('null_diff', ascending=False).head(20)[['feature', 'null_2025', 'null_2026', 'null_diff']])
    
    # Analysis 2: Significant Distribution Drift
    logger.info("\n--- TOP DISTRIBUTION DRIFTS (by KS Statistic) ---")
    print(df_res.sort_values('ks_stat', ascending=False).head(20)[['feature', 'ks_stat', 'drift_score', 'null_diff']])
    
    # Analysis 3: Significant Mean Shift
    logger.info("\n--- TOP MEAN SHIFTS (by Drift Score) ---")
    df_res['abs_drift'] = df_res['drift_score'].abs()
    print(df_res.sort_values('abs_drift', ascending=False).head(20)[['feature', 'drift_score', 'mean_2025', 'mean_2026']])

if __name__ == "__main__":
    main()
