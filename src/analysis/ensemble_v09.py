
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import pickle
from sklearn.metrics import ndcg_score

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.dataset import DatasetSplitter
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.cleansing import DataCleanser
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model(artifact_dir, model_type):
    if model_type == 'catboost':
        model = cb.CatBoostRanker() # Assuming Ranker
        model.load_model(os.path.join(artifact_dir, 'model.cbm'))
        return model
    else:
        with open(os.path.join(artifact_dir, 'model.pkl'), 'rb') as f:
            return pickle.load(f)

def evaluate_ndcg(y_true, y_score, group):
    """
    Calculate mean NDCG@5 for grouped data
    """
    ndcg_list = []
    current_idx = 0
    for size in group:
        y_t = y_true[current_idx : current_idx + size]
        y_s = y_score[current_idx : current_idx + size]
        
        if np.sum(y_t) > 0:
            # ndcg_score expects [samples, features] or [1, samples]
            # y_t, y_s are 1D arrays of shape (size,)
            score = ndcg_score([y_t], [y_s], k=5)
            ndcg_list.append(score)
        
        current_idx += size
    
    return np.mean(ndcg_list) if ndcg_list else 0.0

def main(config_lgbm_path, config_cat_path, valid_year=2024):
    logger.info(f"Loading configs: {config_lgbm_path}, {config_cat_path}")
    cfg_lgbm = load_config(config_lgbm_path)
    cfg_cat = load_config(config_cat_path) # Not strictly needed if data is same
    
    # Assuming datasets are compatible (same dates/features mostly)
    # We load data using one config's settings (assuming consistency in splits)
    # Ideally v09 models use same feature blocks
    dataset_cfg = cfg_lgbm.get('dataset', {})
    start_date = dataset_cfg.get('train_start_date', '2015-01-01')
    end_date = dataset_cfg.get('test_end_date', '2025-12-31')
    jra_only = dataset_cfg.get('jra_only', False)
    # Note: Ensemble evaluation might need ODDS for ROI simulation later, 
    # but for simple NDCG we can skip odds or keep them.
    # If trained with skip_odds=True, features don't have odds. 
    # We should match training conditions for X.
    skip_odds = dataset_cfg.get('drop_market_data', False)
    
    feature_blocks = cfg_lgbm.get('features', [])
    
    # 1. Load Data
    logger.info("Loading Data...")
    loader = JraVanDataLoader()
    raw_df = loader.load(history_start_date=start_date, end_date=end_date, jra_only=jra_only, skip_odds=skip_odds)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    df = pipeline.load_features(clean_df, feature_blocks)
    
    # Create Targets
    if 'rank' in clean_df.columns:
        if 'rank' not in df.columns:
             # Merge rank back
             target_source = clean_df[['race_id', 'horse_number', 'rank']]
             df = pd.merge(df, target_source, on=['race_id', 'horse_number'], how='left')

    def create_ranking_target(rank):
        if pd.isna(rank): return 0
        if rank == 1: return 3
        elif rank == 2: return 2
        elif rank == 3: return 1
        else: return 0
        
    df['target'] = df['rank'].apply(create_ranking_target)
    
    # Split
    splitter = DatasetSplitter()
    # Ensure keys
    key_cols = ['race_id', 'date', 'horse_id', 'year']
    for k in key_cols:
        if k not in df.columns and k in clean_df.columns:
            df[k] = clean_df[k]
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    datasets = splitter.split_and_create_dataset(df, valid_year=valid_year)
    valid_set = datasets['valid']
    
    X_valid = valid_set['X']
    y_valid = valid_set['y'] # 0-3 relevance
    group_valid = valid_set['group']
    
    # 2. Load Models
    # Look for artifact dirs based on experiment names in configs
    # Or just assume standard paths or pass via args?
    # For now assume paths based on config names or pass artifact dirs?
    # Let's derive from config 'experiment_name'
    exp_name_lgbm = cfg_lgbm.get('experiment_name')
    exp_name_cat = cfg_cat.get('experiment_name')
    
    lgbm_dir = f"models/experiments/{exp_name_lgbm}"
    cat_dir = f"models/experiments/{exp_name_cat}"
    
    logger.info(f"Loading LightGBM from {lgbm_dir}")
    model_lgbm = load_model(lgbm_dir, 'lightgbm')
    
    logger.info(f"Loading CatBoost from {cat_dir}")
    model_cat = load_model(cat_dir, 'catboost')
    
    # 3. Predict
    # Prepare X for models
    
    # LightGBM: needs 'category' dtype for cats
    # Config has list?
    cat_cols_lgbm = cfg_lgbm.get('dataset', {}).get('categorical_features', [])
    auto_cat = [c for c in X_valid.columns if X_valid[c].dtype == 'object']
    cat_cols = list(set(cat_cols_lgbm + auto_cat))
    
    X_lgbm = X_valid.copy()
    for c in cat_cols:
        X_lgbm[c] = X_lgbm[c].astype('category')
        
    # CatBoost: needs strings for cats
    X_cat = X_valid.copy()
    for c in cat_cols:
        X_cat[c] = X_cat[c].fillna("missing").astype(str)
        
    logger.info("Predicting LightGBM...")
    pred_lgbm = model_lgbm.predict(X_lgbm)
    
    logger.info("Predicting CatBoost...")
    pred_cat = model_cat.predict(X_cat)
    
    # 4. Ensemble
    # Rank Averaging
    # Group by race_id (we have group sizes)
    logger.info("Calculating Ensemble...")
    
    # Reconstruct race_id for grouping if needed, or iterate by group size
    # We can iterate by group size to compute per-race ranks
    
    ensemble_scores = []
    current_idx = 0
    
    # Store metrics
    lgbm_ndcgs = []
    cat_ndcgs = []
    ens_ndcgs = []
    
    for size in group_valid:
        # Get slice
        s_l = pred_lgbm[current_idx : current_idx + size]
        s_c = pred_cat[current_idx : current_idx + size]
        y_ref = y_valid.iloc[current_idx : current_idx + size].to_numpy()
        
        # Rank within race (descending score = rank 1)
        # scipy.stats.rankdata gives rank 1=smallest. We want 1=highest info?
        # Typically we want ranks 1, 2, ... N where 1 is best. 
        # So rankdata(-score)
        from scipy.stats import rankdata
        r_l = rankdata(-s_l, method='min') # 1 is best
        r_c = rankdata(-s_c, method='min')
        
        # Average Rank
        avg_rank = (r_l + r_c) / 2.0
        # For NDCG, we need a "score" where higher is better.
        # Inverse of rank is good score proxy? Or just negative rank.
        ens_score = -avg_rank 
        
        # Evaluate NDCG for this query
        # ndcg_score requires > 1 document
        if size > 1 and np.sum(y_ref) > 0:
            lgbm_ndcgs.append(ndcg_score([y_ref], [s_l], k=5))
            cat_ndcgs.append(ndcg_score([y_ref], [s_c], k=5))
            ens_ndcgs.append(ndcg_score([y_ref], [ens_score], k=5))
            
        current_idx += size

    logger.info("=== Results (Validation 2024) ===")
    logger.info(f"LightGBM NDCG@5: {np.mean(lgbm_ndcgs):.4f}")
    logger.info(f"CatBoost NDCG@5: {np.mean(cat_ndcgs):.4f}")
    logger.info(f"Ensemble NDCG@5: {np.mean(ens_ndcgs):.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm_config", default="config/experiments/exp_v09_lgbm.yaml")
    parser.add_argument("--cat_config", default="config/experiments/exp_v09_cat.yaml")
    args = parser.parse_args()
    
    main(args.lgbm_config, args.cat_config)
