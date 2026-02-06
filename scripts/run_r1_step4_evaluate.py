
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import json
import joblib

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.nn_baseline import SimpleMLP

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
    parser = argparse.ArgumentParser(description="Phase R MLP Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    
    EVAL_YEAR = 2024
    logger.info(f"🚀 Starting Phase R Evaluation [{EXP_NAME}] for Year {EVAL_YEAR}")
    
    # Load Model artifacts
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    embedding_dims = joblib.load(os.path.join(MODEL_DIR, "embedding_dims.pkl"))
    
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
    
    # Identify Features (Same logic as train)
    ds_cfg = cfg['dataset']
    cat_feat_names = ds_cfg.get('categorical_features', [])
    cat_feat_names = [c for c in cat_feat_names if c in df.columns] # must exist
    
    # Preprocessing
    # Numericals
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff']
    all_cols = [c for c in df.columns if c not in exclude_cols]
    num_feat_names = [c for c in all_cols if c not in cat_feat_names]
    
    # Filter num feats exactly as trained?
    # Ideally should save feature names in train.
    # Assuming logic is deterministic and columns are same.
    final_num_feats = []
    for c in num_feat_names:
        if df_eval[c].dtype == 'object':
             try: df_eval[c] = pd.to_numeric(df_eval[c])
             except: pass
        if pd.api.types.is_numeric_dtype(df_eval[c]):
            final_num_feats.append(c)
    num_feat_names = final_num_feats
    
    # Verify input shape expected by scaler
    if len(num_feat_names) != scaler.mean_.shape[0]:
        logger.warning(f"Feature count mismatch! Scaler: {scaler.mean_.shape[0]}, Data: {len(num_feat_names)}")
        # This is risky. We need exact match.
        # How to reconstruct exact list?
        # Re-run the exact filtering logic on the FULL dataset columns?
        # Or load columns from saved if we saved them... we didn't.
        # But we loaded Q8 parquet which has fixed columns.
        pass
    
    logger.info("Preprocessing...")
    X_num = df_eval[num_feat_names].fillna(0).values
    X_num = scaler.transform(X_num)
    
    X_cat_list = []
    for cat_col in cat_feat_names:
        df_eval[cat_col] = df_eval[cat_col].astype(str).fillna('nan')
        if cat_col in encoders:
            oe = encoders[cat_col]
            enc = oe.transform(df_eval[[cat_col]].values)
            enc = enc + 1 
            X_cat_list.append(enc)
        else:
            # Should not happen if config same
            pass
            
    X_cat = np.hstack(X_cat_list).astype(int) if X_cat_list else np.zeros((len(df_eval), 0), dtype=int)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP(
        num_numerical_features=len(num_feat_names),
        embedding_dims=embedding_dims,
        hidden_dims=cfg['model_params']['hidden_dims'],
        dropout_rate=cfg['model_params']['dropout_rate']
    ).to(device)
    
    model_path = os.path.join(MODEL_DIR, "model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    logger.info("Predicting...")
    batch_size = 4096
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_num), batch_size):
            xn = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32).to(device)
            xc = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long).to(device)
            out = model(xn, xc).squeeze().cpu().numpy()
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
