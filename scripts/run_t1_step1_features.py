
import sys
import os
import logging
import pandas as pd
import numpy as np
import traceback

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Phase T Step 1: Odds Fluctuation Feature Generation")
    
    # Feature Set: Only target the new block for now
    FEATURE_BLOCKS = ['odds_fluctuation']
    
    # Use a separate cache dir for T1 or shared?
    # Let's use T1 specific feature cache to avoid polluting Q8 if experimental
    CACHE_DIR = "data/features_t1"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Output Dir
    TEMP_DIR = "data/temp_t1"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Load Raw Data (Reuse Q1 temp which has 2024, need to fetch 2025?)
    # FeaturePipeline expects "clean_df" with basic cols, but Track Bias needs RAW results (pass_rank, waku_no).
    # Simple parquet load might miss these if they were dropped.
    
    logger.info("Loading Raw Data for Feature Generation...")
    loader = JraVanDataLoader()
    
    # We need data for 2024-2025
    # Use loader to get raw SE/RA data (which has pass_rank, waku_no)
    # This might be heavy, but necessary for Track Bias
    
    raw_dfs = []
    # Check if we can just append necessary cols to existing parquet or reload completely?
    # Reloading from DB is safer for "raw" requirement.
    
    # Try loading from cache first if exists? No, cache might be stripped.
    # Let's try DB fetch for 2024 and 2025.
    
    try:
        years = [2024, 2025]
        for y in years:
            logger.info(f"Fetching raw race data for {y} from DB...")
            # Use a custom query or strict load? 
            # loader.load(year=y) does full preprocessing usually. 
            # We want RAW DF.
            # But loader doesn't expose simple raw fetch easily except load().
            # Let's use read_sql directly for specific columns needed for Track Bias if possible,
            # OR better, use loader.load() with minimal preprocessing?
            # Actually, loader.load() returns 'df' which usually has pass_rank if 'is_training' is True?
            # Let's check loader.
            
            # Note: jvd_se doesn't have race_id directly. Construct it.
            # Map columns: umaban -> horse_number, kakutei_chakujun -> rank, corner_1 -> pass_rank, wakuban -> waku_no
            query = f"""
                SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
                       umaban, kakutei_chakujun, wakuban, corner_1, ketto_toroku_bango
                FROM jvd_se
                WHERE kaisai_nen = '{y}'
            """
            df_y = pd.read_sql(query, loader.engine)
            
            if not df_y.empty:
                # Construct race_id
                df_y['race_id'] = (
                    df_y['kaisai_nen'] + 
                    df_y['keibajo_code'] + 
                    df_y['kaisai_kai'] + 
                    df_y['kaisai_nichime'] + 
                    df_y['race_bango']
                )
                
                # Resize/Rename
                df_y = df_y.rename(columns={
                    'umaban': 'horse_number',
                    'kakutei_chakujun': 'rank',
                    'corner_1': 'pass_rank',
                    'wakuban': 'waku_no',
                    'ketto_toroku_bango': 'horse_id'
                })
            
            # RA Data for Date
            # Use kaisai_tsukihi instead of kaisai_gappi (which doesn't exist)
            # Use hasso_jikoku instead of start_time
            # Remove course_id, distance, etc. to avoid UndefinedColumn errors.
            # We only really need date and start_time for the current feature set.
            query_ra = f"""
                 SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
                        kaisai_tsukihi, hasso_jikoku
                 FROM jvd_ra
                 WHERE kaisai_nen = '{y}'
            """
            df_ra = pd.read_sql(query_ra, loader.engine)
            
            if not df_ra.empty:
                 df_ra['race_id'] = (
                    df_ra['kaisai_nen'] + 
                    df_ra['keibajo_code'] + 
                    df_ra['kaisai_kai'] + 
                    df_ra['kaisai_nichime'] + 
                    df_ra['race_bango']
                )
                 # Construct date from YYYY + MMDD
                 df_ra['date_str'] = df_ra['kaisai_nen'] + df_ra['kaisai_tsukihi']
                 df_ra['date'] = pd.to_datetime(df_ra['date_str'], format='%Y%m%d', errors='coerce')
                 df_ra = df_ra.rename(columns={'hasso_jikoku': 'start_time_str'}).drop(columns=['date_str'])

            if not df_y.empty and not df_ra.empty:
                # Merge
                df_merged = pd.merge(df_y, df_ra, on='race_id', how='left')
                # Drop duplicate cols from join if any (suffixes will protect)
                raw_dfs.append(df_merged)
                
    except Exception as e:
        logger.warning(f"DB Fetch failed: {e}. Falling back to Parquet.")
        
    if raw_dfs:
        clean_df = pd.concat(raw_dfs, ignore_index=True)
        # Ensure types
        clean_df['rank'] = pd.to_numeric(clean_df['rank'], errors='coerce')
        clean_df['waku_no'] = pd.to_numeric(clean_df['waku_no'], errors='coerce')
        # pass_rank is string '1-1-1'
        
        # Ensure merge keys type match for odds_fluctuation and track_bias
        clean_df['race_id'] = clean_df['race_id'].astype(str)
        clean_df['horse_number'] = pd.to_numeric(clean_df['horse_number'], errors='coerce').fillna(0).astype(int)
    else:
        # Fallback to parquet (Odds might work, Track Bias will fail)
        raw_dfs_pq = []
        for y in [2024, 2025]:
            fpath = f"data/temp_q1/year_{y}.parquet"
            if os.path.exists(fpath):
                raw_dfs_pq.append(pd.read_parquet(fpath))
        if raw_dfs_pq:
            clean_df = pd.concat(raw_dfs_pq, ignore_index=True)
        else:
            logger.error("No data found.")
            return

    logger.info(f"Loaded {len(clean_df)} rows.")

    # Generate Features
    logger.info(f"Generating Features: {FEATURE_BLOCKS}...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Force re-compute
    for block in FEATURE_BLOCKS:
        cache_path = os.path.join(CACHE_DIR, f"{block}.parquet")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)

    # Compute Track Bias Features (Manual Call)
    from preprocessing.features.track_bias import calculate_track_bias_features
    logger.info("Computing Track Bias Features...")
    df_bias = calculate_track_bias_features(clean_df)
    
    if not df_bias.empty:
        if 'horse_id' in df_features.columns and 'horse_id' in df_bias.columns:
             # Merge using horse_id if available to be safe, but race_id+horse_number is standard
             # df_bias returns ['race_id', 'horse_number', 'bias_adversity_score_mean_5']
             # Ensure race_id type match
             df_bias['race_id'] = df_bias['race_id'].astype(str)
             df_features = pd.merge(df_features, df_bias, on=['race_id', 'horse_number'], how='left')
        else:
             df_bias['race_id'] = df_bias['race_id'].astype(str)
             df_features = pd.merge(df_features, df_bias, on=['race_id', 'horse_number'], how='left')
        logger.info(f"Merged Track Bias features. Shape: {df_features.shape}")
    else:
        logger.warning("Track Bias features empty!")
    
    # Verify
    if 'odds_ratio_10min' not in df_features.columns:
         logger.error("❌ CRITICAL: odds_fluctuation features missing!")
         raise ValueError("Feature generation failed")
         
    # Save
    out_path = f"{TEMP_DIR}/T1_features_2024_2025.parquet"
    df_features.to_parquet(out_path)
    logger.info(f"Features saved to {out_path}")
    
    # Sample check
    print("\nSample Features:")
    sample_cols = ['race_id', 'horse_number', 'odds_final', 'odds_ratio_10min']
    if 'bias_adversity_score_mean_5' in df_features.columns:
        sample_cols.append('bias_adversity_score_mean_5')
    if 'odds_ratio_60_10' in df_features.columns:
        sample_cols.append('odds_ratio_60_10')
        
    print(df_features[sample_cols].dropna().head(10))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
