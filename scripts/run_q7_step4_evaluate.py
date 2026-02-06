
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from sklearn.metrics import log_loss, roc_auc_score, ndcg_score

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_ndcg(y_true, y_score, k=5):
    # sklearn ndcg_score expects (n_samples, n_items)
    # But here we have list of races with varying items.
    # We must compute per race and average.
    pass

def evaluate_ranking(df, k=5):
    # df has ['race_id', 'rank', 'y_prob']
    # Group by race_id
    ndcg_list = []
    recall_list = []
    
    # Pre-sort by probability desc
    df_sorted = df.sort_values(['race_id', 'y_prob'], ascending=[True, False])
    
    grouped = df_sorted.groupby('race_id')
    
    for race_id, group in grouped:
        # True relevance: 1 if rank<=3? or rank itself?
        # Usually NDCG uses graded relevance. 1st=3, 2nd=2, 3rd=1, others=0
        # Or Binary: 1st=1.
        # Let's use: 1/(rank) or Standard Graded Relevance
        # For NDCG@5, we want to know if Top5 contains winners.
        
        # Mapping rank to relevance
        # 1->3, 2->2, 3->1, else 0
        relevance = group['rank'].apply(lambda r: 3 if r==1 else (2 if r==2 else (1 if r==3 else 0))).values
        
        if relevance.sum() == 0:
            continue
            
        scores = group['y_prob'].values
        
        # NDCG@K
        # Explicitly reshape for sklearn
        try:
            val = ndcg_score([relevance], [scores], k=k)
            ndcg_list.append(val)
        except:
            pass
            
        # Recall@K (Top3 in TopK predictions)
        # TopK predicted indices
        # Since group is sorted by score desc, top K are first K
        top_k_preds = group.iloc[:k]
        
        # Count how many of Top3 horses (rank<=3) are in TopK preds
        # Actually Recall@K usually means: (Relevant items in TopK) / (Total Relevant Items)
        relevant_horses = group[group['rank'] <= 3]
        n_relevant = len(relevant_horses)
        
        if n_relevant == 0:
            continue
            
        n_found = top_k_preds[top_k_preds['rank'] <= 3].shape[0]
        recall = n_found / n_relevant
        recall_list.append(recall)
        
    return {
        f'ndcg@{k}': np.mean(ndcg_list),
        f'recall@{k}': np.mean(recall_list)
    }

def main():
    parser = argparse.ArgumentParser(description="Phase Q Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    parser.add_argument("--model_dir", type=str, help="Override model directory (for baseline eval)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    
    # Allow override
    if args.model_dir:
        MODEL_DIR = args.model_dir
        logger.info(f"Overriding Model Dir: {MODEL_DIR}")
    else:
        MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
        
    EVAL_YEAR = 2024 # Fixed Target 2024
    
    logger.info(f"🚀 Starting Phase Q Evaluation [{EXP_NAME}] for Year {EVAL_YEAR}")
    
    # Paths
    FEAT_path = "data/temp_q7/Q7_features.parquet"
    TGT_path = "data/temp_q7/Q7_targets.parquet"
    MODEL_path = os.path.join(MODEL_DIR, "model.pkl")
    CALIB_path = os.path.join(MODEL_DIR, "calibrator_win.pkl") # Assuming win target
    
    if not os.path.exists(MODEL_path):
        logger.error(f"Model not found: {MODEL_path}")
        return
        
    # Load Model
    model = joblib.load(MODEL_path)
    calibrator = None
    if os.path.exists(CALIB_path):
        calibrator = joblib.load(CALIB_path)
    
    # Load Data (Only 2024)
    logger.info("Loading Data...")
    df_features = pd.read_parquet(FEAT_path)
    df_targets = pd.read_parquet(TGT_path)
    
    # Filter by date/year before merging to save memory?
    # Targets usually have date.
    df_targets['year'] = pd.to_datetime(df_targets['date']).dt.year
    df_targets_2024 = df_targets[df_targets['year'] == EVAL_YEAR].copy()
    
    if len(df_targets_2024) == 0:
        logger.error(f"No targets found for {EVAL_YEAR}")
        return
        
    df = pd.merge(df_targets_2024, df_features, on=['race_id', 'horse_number'], how='left')
    logger.info(f"Loaded 2024 Data: {df.shape}")
    
    # Preprocessing
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff']
    cat_cols = cfg['dataset']['categorical_features']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Safety Catch
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in exclude_cols:
            df[col] = df[col].astype('category')
            
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Automatic Feature Selection based on Model
    # This allows evaluating baseline model (fewer features) on Q1 data (more features)
    if hasattr(model, 'booster_'):
        model_booster = model.booster_
    else:
        model_booster = model
        
    model_features = model_booster.feature_name()
    
    # Check coverage
    missing_feats = [f for f in model_features if f not in df.columns]
    if missing_feats:
        logger.error(f"Data is missing features required by model: {missing_feats}")
        return
        
    logger.info(f"Filtering features to match model: {len(model_features)} features")
    X_test = df[model_features]
    y_test = (df['rank'] == 1).astype(int) # Binary Win for LogLoss/AUC
    
    # Predict
    logger.info("Predicting...")
    y_pred_raw = model.predict(X_test)
    y_pred_prob = y_pred_raw
    if calibrator:
        y_pred_prob = calibrator.predict_proba(y_pred_raw.reshape(-1, 1))[:, 1]
        
    # Evaluate Global
    auc = roc_auc_score(y_test, y_pred_prob)
    ll = log_loss(y_test, y_pred_prob)
    logger.info(f"Global 2024 AUC: {auc:.4f}, LogLoss: {ll:.4f}")
    
    # Evaluate Ranking
    eval_df = df[['race_id', 'horse_number', 'rank']].copy()
    eval_df['y_prob'] = y_pred_prob
    
    metrics = evaluate_ranking(eval_df, k=5)
    logger.info(f"Ranking Metrics: {metrics}")
    
    # Segment Analysis
    logger.info("Segment Analysis...")
    segments = {}
    
    # Segment 1: Class Promotion (is_same_class_prev == 0)
    # Check if we have the feature
    if 'is_same_class_prev' in df.columns:
        seg_mask = (df['is_same_class_prev'] == 0)
        seg_df = eval_df[seg_mask]
        if len(seg_df) > 0:
            seg_metrics = evaluate_ranking(seg_df, k=5)
            logger.info(f"Segment [Class Change]: {seg_metrics}")
            segments['class_change'] = seg_metrics
            
    # Segment 2: Small Field (<= 10 horses)
    race_counts = eval_df.groupby('race_id').size()
    small_field_races = race_counts[race_counts <= 10].index
    seg_small = eval_df[eval_df['race_id'].isin(small_field_races)]
    if len(seg_small) > 0:
        s_met = evaluate_ranking(seg_small, k=5)
        logger.info(f"Segment [Small Field]: {s_met}")
        segments['small_field'] = s_met
        
    # Save Results
    results = {
        'experiment': EXP_NAME,
        'year': EVAL_YEAR,
        'global': {'auc': auc, 'logloss': ll},
        'ranking': metrics,
        'segments': segments
    }
    
    with open(os.path.join(MODEL_DIR, "eval_2024.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Evaluation Complete!")

if __name__ == "__main__":
    main()
