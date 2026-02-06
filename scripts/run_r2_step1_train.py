
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import joblib
import gc
import json

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

def main():
    parser = argparse.ArgumentParser(description="Phase R TabNet Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save config
    with open(os.path.join(MODEL_DIR, "config_copy.yaml"), 'w') as f:
        yaml.dump(cfg, f)
        
    # Paths (Q8 Base)
    FEAT_path = "data/temp_q8/Q8_features.parquet"
    TGT_path = "data/temp_q8/Q8_targets.parquet"
    
    logger.info("Loading Data...")
    if not os.path.exists(FEAT_path):
        logger.error(f"Features not found: {FEAT_path}")
        return

    df_features = pd.read_parquet(FEAT_path)
    df_targets = pd.read_parquet(TGT_path)
    
    logger.info("Merging...")
    df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='left')
    del df_features, df_targets
    gc.collect()
    
    # Preprocessing
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        
    ds_cfg = cfg['dataset']
    VALID_YEAR = ds_cfg.get('valid_year', 2023)
    TRAIN_START = ds_cfg.get('train_start_date', '2015-01-01')
    TARGET_TYPE = ds_cfg.get('binary_target', 'win')
    
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff']
    
    # Features
    cat_feat_names = ds_cfg.get('categorical_features', [])
    cat_feat_names = [c for c in cat_feat_names if c in df.columns]
    
    all_cols = [c for c in df.columns if c not in exclude_cols]
    num_feat_names = [c for c in all_cols if c not in cat_feat_names]
    
    # Force convert known categoricals to proper type if needed, but TabNet needs int indices
    # Convert numericals
    final_num_feats = []
    for c in num_feat_names:
        if df[c].dtype == 'object':
             try: df[c] = pd.to_numeric(df[c])
             except: pass
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0) # Simple fill for TabNet
            final_num_feats.append(c)
    num_feat_names = final_num_feats
    
    logger.info(f"Num Features: {len(num_feat_names)}, Cat Features: {len(cat_feat_names)}")
    
    # Define Target
    if TARGET_TYPE == 'win':
        label_func = lambda x: 1 if x == 1 else 0
    df['target'] = df['rank'].apply(label_func)
    
    train_mask = (df['year'] < VALID_YEAR) & (df['year'] >= pd.to_datetime(TRAIN_START).year)
    valid_mask = (df['year'] == VALID_YEAR)
    
    train_df = df[train_mask].reset_index(drop=True)
    valid_df = df[valid_mask].reset_index(drop=True)
    
    # Prepare for TabNet
    # TabNet requires all features in single matrix, cats as ints
    # We need to map cat features to 0..N-1
    
    logger.info("Encoding Categoricals...")
    encoders = {}
    cat_dims = []
    cat_idxs = []
    
    # Combined list of features: [Nums..., Cats...]
    # Actually TabNet handles mixed if we tell it indices
    
    feature_columns = num_feat_names + cat_feat_names
    
    X_train = train_df[feature_columns].copy()
    X_valid = valid_df[feature_columns].copy()
    y_train = train_df['target'].values
    y_valid = valid_df['target'].values
    
    # Encode Cats
    for i, col in enumerate(feature_columns):
        if col in cat_feat_names:
            le = LabelEncoder()
            # Fit on train, handle valid
            # TabNet needs continuous integers 0..N
            # Unseen labels? -> Map to special 'unknown' or just standard LE logic
            # Lets convert to str first
            X_train[col] = X_train[col].astype(str)
            X_valid[col] = X_valid[col].astype(str)
            
            le.fit(X_train[col])
            
            # Handle unknown in valid
            # Map unknowns to a new class? Or most frequent?
            # Easiest: fit on full concat? No leakage...
            # Proper way: fit train. If valid has unknown, map to <UNK> if we created one.
            # If we didn't create UNK, we have problem.
            
            # Strategy: Add 'UNK' to classes explicitly?
            classes = le.classes_.tolist()
            # If we rely on LabelEncoder, it raises error on unseen.
            
            # Custom Safe Encode
            mapping = {c: i for i, c in enumerate(classes)}
            
            # Transform
            X_train[col] = X_train[col].map(mapping).fillna(0).astype(int) # 0 if NaN? no NaN usually after astype(str).
            
            # For Valid, if unknown, map to something exists? 
            # Or -1? TabNet doesn't like -1 usually.
            # Maybe map to most frequent (0)?
            
            X_valid[col] = X_valid[col].map(mapping).fillna(0).astype(int)
            
            encoders[col] = mapping # Save dictionary
            cat_dims.append(len(classes))
            cat_idxs.append(i) # Index in the feature matrix
            
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(cat_dims, os.path.join(MODEL_DIR, "cat_dims.pkl"))
    joblib.dump(cat_idxs, os.path.join(MODEL_DIR, "cat_idxs.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    
    # Convert to Numpy
    X_train_np = X_train.values.astype(float) # TabNet expects float input, casts cats internally if indexes provided
    X_valid_np = X_valid.values.astype(float)
    
    logger.info("Training TabNet...")
    
    mp = cfg['model_params']
    
    # TabNet Classifier
    clf = TabNetClassifier(
        n_d=mp['n_d'],
        n_a=mp['n_a'],
        n_steps=mp['n_steps'],
        gamma=mp['gamma'],
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1, # Default 1, TabNet authors suggest letting it learn or 1.
        optimizer_params=dict(lr=mp['optimizer_params']['lr']),
        mask_type=mp['mask_type'],
        lambda_sparse=mp['lambda_sparse'],
        verbose=1,
        device_name=cfg['resources']['device']
    )
    
    clf.fit(
        X_train=X_train_np, y_train=y_train,
        eval_set=[(X_train_np, y_train), (X_valid_np, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc', 'logloss'],
        max_epochs=mp['max_epochs'],
        patience=mp['patience'],
        batch_size=mp['batch_size'],
        virtual_batch_size=128,
        num_workers=cfg['resources']['num_workers'],
        drop_last=False
    )
    
    # Save Model
    save_path = os.path.join(MODEL_DIR, "tabnet_model.zip")
    clf.save_model(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Metrics
    preds = clf.predict_proba(X_valid_np)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    ll = log_loss(y_valid, preds)
    logger.info(f"Valid AUC: {auc:.4f}, LogLoss: {ll:.4f}")
    
    with open(os.path.join(MODEL_DIR, "metrics.json"), 'w') as f:
        json.dump({'auc': auc, 'logloss': ll, 'best_epoch': clf.best_epoch}, f)

if __name__ == "__main__":
    main()
