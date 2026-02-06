
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import traceback
import gc

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.dataset import DatasetSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting M4-B Step 2: Training (M4-B-Top3)")
    
    EXP_NAME = "exp_m4_b_adhoc"
    VALID_YEAR = 2024
    
    # 1. Load Data
    logger.info("Loading Features and Targets...")
    if not os.path.exists("data/temp_m4/M4_B_features.parquet") or not os.path.exists("data/temp_m4/M4_B_targets.parquet"):
        logger.error("Missing parquet files! Run Step 1 first.")
        return

    df_features = pd.read_parquet("data/temp_m4/M4_B_features.parquet")
    df_targets = pd.read_parquet("data/temp_m4/M4_B_targets.parquet")
    
    logger.info(f"Features: {df_features.shape}, Targets: {df_targets.shape}")
    
    # 2. Merge
    logger.info("Merging Features and Targets...")
    # Targets usually smaller or same size. Inner join or Left join?
    # Features are generated from clean_df, Targets from clean_df. Should be 1:1.
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='left')
    
    # Cleanup inputs
    del df_features
    del df_targets
    gc.collect()
    
    logger.info(f"Merged Shape: {df.shape}")
    
    # 3. Target Definition
    def create_ranking_target(rank):
        if pd.isna(rank): return 0
        if rank == 1: return 3
        elif rank == 2: return 2
        elif rank == 3: return 1
        else: return 0
        
    df['target'] = df['rank'].apply(create_ranking_target)
    
    # Ensure year column for splitter
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    # Convert object columns to category for LightGBM
    logger.info("Converting object columns to category...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    # 4. Split
    logger.info("Splitting Dataset...")
    splitter = DatasetSplitter()
    datasets = splitter.split_and_create_dataset(df, valid_year=VALID_YEAR)
    train_set = datasets['train']
    valid_set = datasets['valid']
    logger.info(f"Train Size: {len(train_set['y'])}, Valid Size: {len(valid_set['y'])}")
    
    # 5. Train
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
    
    model.fit(
        train_set['X'], train_set['y'],
        group=train_set['group'],
        eval_set=[(valid_set['X'], valid_set['y'])],
        eval_group=[valid_set['group']],
        callbacks=[lgb.log_evaluation(100)]
    )
    
    # 6. Save
    logger.info("Saving Model & Results...")
    os.makedirs(f"experiments/{EXP_NAME}", exist_ok=True)
    joblib.dump(model, f"experiments/{EXP_NAME}/model.pkl")
    
    # Predict and Save Valid Preds
    y_pred = model.predict(valid_set['X'])
    valid_df = valid_set['X'].copy()
    valid_df['y_true'] = valid_set['y']
    valid_df['y_pred'] = y_pred
    
    # Restore race_id for analysis
    valid_df['race_id'] = df.loc[valid_set['X'].index, 'race_id'].values
    
    valid_df.to_parquet(f"experiments/{EXP_NAME}/valid_preds.parquet")
    logger.info("Step 2 Completed Successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("🔥 Fatal Error in Step 2:")
        traceback.print_exc()
        sys.exit(1)
