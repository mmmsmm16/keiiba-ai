
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import json
import joblib

# Try importing TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    print("Error: pytorch_tabnet not installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_ranking_metrics(df, k=5):
    # df has ['race_id', 'rank', 'pred_prob']
    ndcg_list = []
    recall_list = []
    
    grouped = df.groupby('race_id')
    for race_id, group in grouped:
        # Relevance: 1st=3, 2nd=2, 3rd=1
        relevance = group['rank'].apply(lambda r: 3 if r==1 else (2 if r==2 else (1 if r==3 else 0))).values
        if relevance.sum() == 0: continue
        
        scores = group['pred_prob'].values
        
        # NDCG
        try:
            ndcg_list.append(ndcg_score([relevance], [scores], k=k))
        except: pass
        
        # Recall@5 (Top3 in Top5)
        top_k = group.sort_values('pred_prob', ascending=False).iloc[:k]
        n_relevant = len(group[group['rank'] <= 3])
        if n_relevant == 0: continue
        
        n_found = top_k[top_k['rank'] <= 3].shape[0]
        recall_list.append(n_found / n_relevant)
        
    return {
        f'ndcg@{k}': np.mean(ndcg_list),
        f'recall@{k}': np.mean(recall_list)
    }

def main():
    parser = argparse.ArgumentParser(description="Phase R TabNet Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    
    EVAL_YEAR = 2024
    logger.info(f"🚀 Starting Phase R Evaluation [{EXP_NAME}] for Year {EVAL_YEAR}")
    
    # Load Encoders
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    
    # Load Data (Q8)
    FEAT_path = "data/temp_q8/Q8_features.parquet"
    TGT_path = "data/temp_q8/Q8_targets.parquet"
    
    logger.info("Loading Data...")
    df_features = pd.read_parquet(FEAT_path)
    df_targets = pd.read_parquet(TGT_path)
    
    logger.info("Merging...")
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='left')
    
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    df_eval = df[df['year'] == EVAL_YEAR].copy().reset_index(drop=True)
    logger.info(f"Eval Rows: {len(df_eval)}")
    
    # Preprocessing
    # Must match train EXACTLY.
    ds_cfg = cfg['dataset']
    cat_feat_names = ds_cfg.get('categorical_features', [])
    
    # Force numeric types if needed (obj -> num)
    for c in feature_columns:
        if c not in df_eval.columns:
            logger.warning(f"Feature {c} missing in eval data. Filling 0.")
            df_eval[c] = 0
            
        if df_eval[c].dtype == 'object' and c not in cat_feat_names:
             try: df_eval[c] = pd.to_numeric(df_eval[c])
             except: pass
             
        if pd.api.types.is_numeric_dtype(df_eval[c]):
             df_eval[c] = df_eval[c].fillna(0)
             
    # Prepare X
    X_eval = df_eval[feature_columns].copy()
    
    # Apply Encoders
    for col, mapping in encoders.items():
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype(str).map(mapping).fillna(0).astype(int)
            
    X_eval_np = X_eval.values.astype(float)
    
    logger.info("Loading Model...")
    clf = TabNetClassifier()
    # file was saved as .zip.zip due to double extension
    model_path = os.path.join(MODEL_DIR, "tabnet_model.zip.zip")
    if not os.path.exists(model_path):
         # Try single zip if double fails (fallback)
         model_path = os.path.join(MODEL_DIR, "tabnet_model.zip")
         
    clf.load_model(model_path)
    
    logger.info("Predicting...")
    # Batch prediction to avoid OOM
    batch_size = 4096
    preds = []
    
    for i in range(0, len(X_eval_np), batch_size):
        x_batch = X_eval_np[i:i+batch_size]
        out = clf.predict_proba(x_batch)[:, 1]
        preds.extend(out)
            
    df_eval['pred_prob'] = preds
    
    # Metrics
    y_true = (df_eval['rank'] == 1).astype(int)
    auc = roc_auc_score(y_true, preds)
    ll = log_loss(y_true, preds)
    
    logger.info(f"Global 2024 AUC: {auc:.4f}, LogLoss: {ll:.4f}")
    
    # Ranking
    metrics = compute_ranking_metrics(df_eval)
    logger.info(f"Ranking Metrics: {metrics}")
    
    # Save
    json.dump({'auc': auc, 'logloss': ll, 'ranking': metrics}, open(os.path.join(MODEL_DIR, "eval_2024_result.json"), 'w'), indent=4)
    
if __name__ == "__main__":
    main()
