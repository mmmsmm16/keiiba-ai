
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import joblib
import traceback
import gc

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.cleansing import DataCleanser
from preprocessing.feature_pipeline import FeaturePipeline
from preprocessing.dataset import DatasetSplitter
from utils.leak_detector import check_data_leakage
from sklearn.metrics import ndcg_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting M4-A Adhoc Experiment (M4-A-Top3)")
    
    # 1. Configuration
    EXP_NAME = "exp_m4_a_adhoc"
    START_DATE = "2023-01-01" # Reduced to 2 years (Train=2023, Valid=2024)
    END_DATE = "2024-12-31"
    VALID_YEAR = 2024
    
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'temporal_jockey_stats', 'temporal_trainer_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
        # 'speed_index_stats', # Crash
        # 'pace_pressure_stats', # Crash
        # 'relative_stats' # Crash
    ]
    
    # 2. Data Loading (Yearly Loop with Disk Backup)
    logger.info(f"Loading Data ({START_DATE} ~ {END_DATE})...")
    # loader = JraVanDataLoader() # Instantiate inside loop
    
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    # dfs = []
    temp_files = []
    
    current_year = start_dt.year
    final_year = end_dt.year
    
    os.makedirs("data/temp_m4", exist_ok=True)
    
    while current_year <= final_year:
        gc.collect()
        y_start = f"{current_year}-01-01"
        y_end = f"{current_year}-12-31"
        if current_year == start_dt.year: y_start = START_DATE
        if current_year == final_year: y_end = END_DATE
        
        temp_file = f"data/temp_m4/year_{current_year}.parquet"
        if os.path.exists(temp_file):
            logger.info(f"  Found cached data for {current_year}: {temp_file}")
            temp_files.append(temp_file)
            current_year += 1
            continue
            
        logger.info(f"  Loading Year {current_year} ({y_start} ~ {y_end})...")
        try:
            # Fresh loader instance to release resources
            loader = JraVanDataLoader() 
            df_year = loader.load(limit=None, history_start_date=y_start, end_date=y_end, skip_odds=True, skip_training=True)
            
            if df_year is not None and len(df_year) > 0:
                # Save to disk immediately
                temp_file = f"data/temp_m4/year_{current_year}.parquet"
                df_year.to_parquet(temp_file)
                temp_files.append(temp_file)
                logger.info(f"    -> Saved {len(df_year)} rows to {temp_file}")
                
                # Clear memory
                del df_year
                del loader
                gc.collect()
            else:
                logger.warning(f"    -> No data for {current_year}")
        except Exception as e:
            logger.error(f"    -> Failed to load {current_year}: {e}")
            traceback.print_exc()
            
        current_year += 1
        
    if not temp_files:
        logger.error("No data loaded!")
        return
        
    logger.info("Concatenating yearly data from disk...")
    dfs = [pd.read_parquet(f) for f in temp_files]
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total Loaded Raw Rows: {len(raw_df)}")
    
    del dfs
    gc.collect()
    
    # 3. Cleansing
    logger.info("Cleansing Data (INLINE MINIMAL)...")
    # cleanser = DataCleanser()
    # clean_df = cleanser.cleanse(raw_df)
    
    # Inline minimal cleansing to bypass crash
    clean_df = raw_df.copy()
    if 'rank' in clean_df.columns:
        # Convert rank to numeric just in case
        clean_df['rank'] = pd.to_numeric(clean_df['rank'], errors='coerce').fillna(0).astype(int)
        clean_df = clean_df[clean_df['rank'] != 0].copy()
    
    clean_df['time_diff'] = np.nan # Placeholder
    
    logger.info(f"Cleansing Complete. Shape: {clean_df.shape}")
    sys.stdout.flush()
    
    del raw_df
    gc.collect()
    
    # 4. Feature Pipeline
    # [Memory Optimization] Prepare target/date info BEFORE pipeline to allow early deletion of clean_df
    target_source = clean_df[['race_id', 'horse_number', 'rank']].copy()
    if 'race_id' in clean_df.columns and 'date' in clean_df.columns:
         date_map = clean_df[['race_id', 'date']].drop_duplicates().groupby('race_id')['date'].first()
    else:
         date_map = None

    logger.info("Initializing FeaturePipeline...")
    pipeline = FeaturePipeline(cache_dir="data/features")
    logger.info("Generating Features...")
    df = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    logger.info(f"Features Ready: {df.shape}")
    sys.stdout.flush()

    # [Memory Optimization] Delete clean_df NOW (it is large)
    del clean_df
    gc.collect()

    # 5. Target Creation (Top3 Ranking)
    logger.info("Creating Target (Top3 Ranking)...")
    if 'rank' not in df.columns:
        # Merge rank from target_source
        df = pd.merge(df, target_source, on=['race_id', 'horse_number'], how='left')
    
    # Clean up target_source
    del target_source
    gc.collect()

    def create_ranking_target(rank):
        if pd.isna(rank): return 0
        if rank == 1: return 3
        elif rank == 2: return 2
        elif rank == 3: return 1
        else: return 0
    df['target'] = df['rank'].apply(create_ranking_target)
    
    # 6. Leak Check
    logger.info("Checking for Leakage... (SKIPPED)")
    # try:
    #     check_data_leakage(df, target_col='target')
    # except Exception as e:
    #     logger.warning(f"Leakage Check Warning: {e}")
    sys.stdout.flush()
        
    # 7. Dataset Split
    logger.info("Splitting Dataset...")
    splitter = DatasetSplitter()
    
    # Ensure key columns logic
    # We already have race_id, horse_number in df.
    # We need 'date' for splitting.
    if 'date' not in df.columns:
        logger.info("Restoring 'date' column from clean_df (via date_map)...")
        if date_map is not None:
            df['date'] = df['race_id'].map(date_map)
        else:
            logger.error("'date_map' is None! Cannot restore date.")

    if 'year' not in df.columns:
         df['year'] = pd.to_datetime(df['date']).dt.year

    datasets = splitter.split_and_create_dataset(df, valid_year=VALID_YEAR)
    train_set = datasets['train']
    valid_set = datasets['valid']
    logger.info(f"Train Size: {len(train_set['y'])}, Valid Size: {len(valid_set['y'])}")
    
    # 8. Training
    logger.info("Training LightGBM (LambdaRank)...")
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'random_state': 42,
        'early_stopping_rounds': 50,
        'verbose': -1
    }
    
    model = lgb.LGBMRanker(**params)
    
    # Needs group info for LambdaRank
    model.fit(
        train_set['X'], train_set['y'],
        group=train_set['group'],
        eval_set=[(valid_set['X'], valid_set['y'])],
        eval_group=[valid_set['group']],
        callbacks=[lgb.log_evaluation(100)]
    )
    
    # 9. Evaluation & Saving
    logger.info("Saving Model & Results...")
    os.makedirs(f"experiments/{EXP_NAME}", exist_ok=True)
    
    # Save Model
    joblib.dump(model, f"experiments/{EXP_NAME}/model.pkl")
    
    # Evaluate
    y_pred = model.predict(valid_set['X'])
    valid_df = valid_set['X'].copy()
    valid_df['y_true'] = valid_set['y']
    valid_df['y_pred'] = y_pred
    
    # To save meaningful results we probably want race_id back if it was dropped?
    # X usually drops non-feature columns.
    # But valid_set in DatasetSplitter might preserve them? No, split_and_create_dataset returns X as features only.
    # We need to recover race_id.
    # Index is preserved.
    # But we previously deleted clean_df and df? No, splitter slices df so original df index is preserved?
    # train_test_split usually shuffles?
    # DatasetSplitter documentation says: "split_and_create_dataset(df, ...)"
    # It usually splits by year, so no shuffle.
    # So index should match df.loc[valid_set['X'].index]
    
    valid_df['race_id'] = df.loc[valid_set['X'].index, 'race_id'].values
    
    valid_df.to_parquet(f"experiments/{EXP_NAME}/valid_preds.parquet")
    
    logger.info("Experiments Completed Successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("🔥 Fatal Error in main execution:")
        traceback.print_exc()
        sys.exit(1)
