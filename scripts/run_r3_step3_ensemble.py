
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import joblib
import json
import gc
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from glob import glob

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.nn_baseline import SimpleMLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_ranking_metrics(df, k=5):
    ndcg_list = []
    recall_list = []
    
    grouped = df.groupby('race_id')
    for race_id, group in grouped:
        relevance = group['rank'].apply(lambda r: 3 if r==1 else (2 if r==2 else (1 if r==3 else 0))).values
        if relevance.sum() == 0: continue
        
        scores = group['pred_prob'].values
        try: ndcg_list.append(ndcg_score([relevance], [scores], k=k))
        except: pass
        
        top_k = group.sort_values('pred_prob', ascending=False).iloc[:k]
        n_relevant = len(group[group['rank'] <= 3])
        if n_relevant == 0: continue
        n_found = top_k[top_k['rank'] <= 3].shape[0]
        recall_list.append(n_found / n_relevant)
        
    return {f'ndcg@{k}': np.mean(ndcg_list), f'recall@{k}': np.mean(recall_list)}

def predict_lgbm(model_path, df, cfg):
    logger.info(f"Predicting LGBM: {model_path}")
    model = joblib.load(model_path)
    
    # LGBM needs same feature columns and categorical types
    # Load config to know features? 
    # Or just rely on what's in df. 
    # Q8 config used all cols except exclude.
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff']
    
    # We must match training features. 
    # Luckily model is loaded from pkl, sklearn/lgbm model remembers?
    # LGBM Booster remembers feature names.
    
    # Prepare DF
    # Need to convert categories
    # In Q8 train, we converted config defined cats to category type
    # And others to category if object.
    
    # Load Q8 config
    q8_cfg = load_config(cfg['config_path'])
    cat_cols = q8_cfg['dataset']['categorical_features']
    
    X = df.copy()
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    for col in X.select_dtypes(include=['object']).columns:
        if col not in exclude_cols:
            X[col] = X[col].astype('category')
            
    # Feature names
    # LGBM model should have feature_name()
    model_features = model.feature_name()
    X = X[model_features]
    
    return model.predict(X)

def predict_mlp(model_info, df):
    logger.info(f"Predicting MLP: {model_info['model_path']}")
    model_dir = model_info['model_dir']
    
    # Load Preprocessors
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    embedding_dims = joblib.load(os.path.join(model_dir, "embedding_dims.pkl"))
    mlp_cfg = load_config(model_info['config_path'])
    
    # Config Features
    ds_cfg = mlp_cfg['dataset']
    cat_feat_names = ds_cfg.get('categorical_features', [])
    exclude_cols = [
        'race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff',
        'interval_type_code', 'frame_number',
        'horse_id', # horse_id is in cat list usually, but if not, exclude unique id
        # Keep interval_days and tataki_count as they might be in legacy model
        'apt_int_win', 'apt_int_top3', 'is_first_int_type'
    ]
    
    # Reconstruct the exact numerical features list strictly from Q8 data
    # to match the Scaler trained on Q8.
    try:
        # q8_path = cfg['data_path']['features'].replace('preprocessed_data_v11', 'temp_q8/Q8_features')
        # We don't have global cfg here. Use fixed path.
        q8_path = "data/temp_q8/Q8_features.parquet"
        
        # Read 1 row to get columns and types
        q8_df = pd.read_parquet(q8_path).head(1)
        
        # Apply Train Logic
        all_cols_q8 = [c for c in q8_df.columns if c not in exclude_cols]
        num_cols_q8 = [c for c in all_cols_q8 if c not in cat_feat_names]
        print("DEBUG num_cols_q8 len:", len(num_cols_q8))
        print("DEBUG num_cols_q8:", sorted(num_cols_q8))
        
        # Check alignment
        
        final_num_feats = []
        for c in num_cols_q8:
            if q8_df[c].dtype == 'object':
                 try: q8_df[c] = pd.to_numeric(q8_df[c])
                 except: pass
            if pd.api.types.is_numeric_dtype(q8_df[c]):
                final_num_feats.append(c)
        
        num_feat_names = final_num_feats
        
    except Exception as e:
        logger.error(f"Failed to reconstruction Q8 features: {e}")
        # Fallback (risky)
        all_cols = [c for c in df.columns if c not in exclude_cols]
        num_feat_names = [c for c in all_cols if c not in cat_feat_names]

    # Align v11 df to these features
    # Fill missing with 0, Drop extra
    missing = [c for c in num_feat_names if c not in df.columns]
    if missing:
        # logger.warning(f"MLP: Missing features in v11: {missing}. Filling 0.")
        for c in missing: df[c] = 0
        
    # Strictly reindex to ensure order
    X_num = df[num_feat_names].fillna(0).values
    X_num = scaler.transform(X_num)
    
    X_cat_list = []
    for cat_col in cat_feat_names:
        if cat_col not in df.columns:
            # logger.warning(f"MLP: Missing categorical feature '{cat_col}'. Filling with 'nan'.")
            df[cat_col] = 'nan'
            
        df[cat_col] = df[cat_col].astype(str).fillna('nan')
        if cat_col in encoders:
            oe = encoders[cat_col]
            enc = oe.transform(df[[cat_col]].values)
            enc = enc + 1
            X_cat_list.append(enc)
            
    X_cat = np.hstack(X_cat_list).astype(int) if X_cat_list else np.zeros((len(df), 0), dtype=int)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP(
        num_numerical_features=len(num_feat_names),
        embedding_dims=embedding_dims,
        hidden_dims=mlp_cfg['model_params']['hidden_dims'],
        dropout_rate=mlp_cfg['model_params']['dropout_rate']
    ).to(device)
    model.load_state_dict(torch.load(model_info['model_path']))
    model.eval()
    
    batch_size = 4096
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_num), batch_size):
            xn = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32).to(device)
            xc = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long).to(device)
            out = model(xn, xc).squeeze().cpu().numpy()
            preds.extend(out)
            
    return np.array(preds)

def predict_tabnet(model_info, df):
    logger.info(f"Predicting TabNet: {model_info['model_path']}")
    model_dir = model_info['model_dir']
    
    # Load Preprocessors
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
    
    # Preprocessing
    df_proc = df.copy()
    for c in feature_columns:
        if c not in df_proc.columns: df_proc[c] = 0
        if df_proc[c].dtype == 'object':
             try: df_proc[c] = pd.to_numeric(df_proc[c])
             except: pass
        if pd.api.types.is_numeric_dtype(df_proc[c]):
             df_proc[c] = df_proc[c].fillna(0)
             
    X_eval = df_proc[feature_columns].copy()
    for col, mapping in encoders.items():
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype(str).map(mapping).fillna(0).astype(int)
            
    X_eval_np = X_eval.values.astype(float)
    
    clf = TabNetClassifier()
    # Handle zip.zip issue
    path = model_info['model_path']
    if not os.path.exists(path):
        if url.endswith('.zip.zip'): path = path[:-4]
        
    clf.load_model(path)
    
    batch_size = 4096
    preds = []
    for i in range(0, len(X_eval_np), batch_size):
        x_batch = X_eval_np[i:i+batch_size]
        out = clf.predict_proba(x_batch)[:, 1]
        preds.extend(out)
        
    return np.array(preds)

def main():
    parser = argparse.ArgumentParser(description="Phase R Ensemble Script")
    parser.add_argument("--config", type=str, required=True, help="Path to ensemble config yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # Load Data
    logger.info("Loading Data...")
    df_features = pd.read_parquet(cfg['data_path']['features'])
    
    if 'rank' in df_features.columns:
        logger.info("Rank found in features, skipping target merge.")
        df = df_features
    else:
        df_targets = pd.read_parquet(cfg['data_path']['targets'])
        df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='left')
    
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    EVAL_YEAR = cfg['dataset']['valid_year']
    df_eval = df[df['year'] == EVAL_YEAR].copy().reset_index(drop=True)
    logger.info(f"Eval Year: {EVAL_YEAR}, Rows: {len(df_eval)}")
    
    preds_dict = {}
    
    # Predict Models
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    
    # Allow override
    if hasattr(args, 'model_dir') and args.model_dir:
        MODEL_DIR = args.model_dir
        logger.info(f"Overriding Model Dir: {MODEL_DIR}")
    else:
        MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        preds_dict['lgbm'] = predict_lgbm(cfg['models']['lgbm']['model_path'], df_eval, cfg['models']['lgbm'])
        
    if 'mlp' in cfg['models']:
        preds_dict['mlp'] = predict_mlp(cfg['models']['mlp'], df_eval)
        
    if 'tabnet' in cfg['models']:
        preds_dict['tabnet'] = predict_tabnet(cfg['models']['tabnet'], df_eval)
        
    # Ensemble
    # Try Simple Average
    logger.info("Ensembling...")
    final_pred = np.zeros(len(df_eval))
    
    for name, pred in preds_dict.items():
        weight = cfg['models'][name]['weight']
        logger.info(f"Model {name} (w={weight}): AUC={roc_auc_score((df_eval['rank']==1).astype(int), pred):.4f}")
        final_pred += pred * weight
        
    final_pred /= sum([m['weight'] for m in cfg['models'].values()])
    
    df_eval['pred_prob'] = final_pred
    
    # Metrics
    y_true = (df_eval['rank'] == 1).astype(int)
    auc = roc_auc_score(y_true, final_pred)
    ll = log_loss(y_true, final_pred)
    
    logger.info(f"=== ENSEMBLE RESULTS ===")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"LogLoss: {ll:.4f}")
    
    metrics = compute_ranking_metrics(df_eval)
    logger.info(f"Ranking: {metrics}")
    
    # ROI Simulation (Simple)
    logger.info("--- ROI Simulation (Win Top 1) ---")
    
    # Load Odds from Q1 temp data (known location for raw data)
    odds_path = f"data/temp_q1/year_{EVAL_YEAR}.parquet"
    if 'odds' not in df_eval.columns:
        if os.path.exists(odds_path):
             logger.info(f"Loading Odds from {odds_path}")
             df_odds = pd.read_parquet(odds_path, columns=['race_id', 'horse_number', 'odds'])
             df_eval = pd.merge(df_eval, df_odds, on=['race_id', 'horse_number'], how='left')
        else:
             logger.warning(f"Odds file not found at {odds_path}. Skipping ROI.")

    if 'odds' in df_eval.columns:
        # Calculate Rank based on Pred Prob
        df_eval['pred_rank'] = df_eval.groupby('race_id')['pred_prob'].rank(ascending=False, method='first')
        
        # Bet on Top 1
        bets = df_eval[df_eval['pred_rank'] == 1]
        
        # Filter for valid odds
        valid_bets = bets[bets['odds'] > 0]
        n_bets = len(valid_bets)
        cost = n_bets * 100
        
        hits = valid_bets[valid_bets['rank'] == 1]
        ret = (hits['odds'] * 100).sum()
        
        roi = ret / cost if cost > 0 else 0
        hit_rate = len(hits) / n_bets if n_bets > 0 else 0
        
        logger.info(f"Valid Races (Odds>0): {n_bets} / {len(bets)}")
        logger.info(f"Hit Rate: {hit_rate:.4f} ({len(hits)}/{n_bets})")
        logger.info(f"Return: {ret:.0f} / {cost} = {roi*100:.2f}%")
        logger.info(f"ROI (Win Top1): {roi*100:.2f}%")
    

    # Save Predictions DataFrame for complex simulation (First!)
    pred_save_path = os.path.join(MODEL_DIR, f"predictions_{EVAL_YEAR}.parquet")
    cols_to_save = ['race_id', 'horse_number', 'rank', 'pred_prob', 'odds']
    # Ensure odds exists, fill 0 if not
    if 'odds' not in df_eval.columns:
        df_eval['odds'] = 0
    
    # Ensure columns exist
    for c in cols_to_save:
        if c not in df_eval.columns: df_eval[c] = 0
        
    df_eval[cols_to_save].to_parquet(pred_save_path)
    logger.info(f"Saved predictions to {pred_save_path}")

    # Save Results to JSON independently of odds
    results = {
        'experiment': 'exp_r3_ensemble',
        'year': int(EVAL_YEAR),
        'auc': float(auc),
        'logloss': float(ll),
        # Metrics dict might contain numpy types
        'ranking': {k: float(v) for k, v in metrics.items()}
    }
    if 'odds' in df_eval.columns:
        results.update({
            'roi': float(roi),
            'hit_rate': float(hit_rate),
            'return': float(ret),
            'cost': float(cost)
        })
        
    with open("models/experiments/ensemble_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
