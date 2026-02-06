
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, log_loss
import json
import joblib
import gc

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.nn_baseline import SimpleMLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
    def __len__(self):
        return len(self.x_num)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]

def main():
    parser = argparse.ArgumentParser(description="Phase R MLP Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    EXP_NAME = cfg.get('experiment_name', 'exp_unknown')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save config
    with open(os.path.join(MODEL_DIR, "config_copy.yaml"), 'w') as f:
        yaml.dump(cfg, f)
        
    # Paths
    # Use Q8 data as base
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
    
    # Identify Categorical Features
    cat_feat_names = ds_cfg.get('categorical_features', [])
    # Verify existence
    cat_feat_names = [c for c in cat_feat_names if c in df.columns]
    
    # Identify Numerical Features
    # All other columns that are not excluded
    all_cols = [c for c in df.columns if c not in exclude_cols]
    num_feat_names = [c for c in all_cols if c not in cat_feat_names]
    
    # Check for any remaining object columns in num_feat_names and remove them or add to cats
    final_num_feats = []
    
    for c in num_feat_names:
        # Try to convert to numeric force
        # e.g. age might be object
        if df[c].dtype == 'object':
             try:
                 df[c] = pd.to_numeric(df[c])
             except:
                 pass
                 
        if pd.api.types.is_numeric_dtype(df[c]):
            final_num_feats.append(c)
        else:
            logger.warning(f"Dropping non-numeric column not in cat list: {c}")
            
    num_feat_names = final_num_feats
    
    logger.info(f"Num Features: {len(num_feat_names)}, Cat Features: {len(cat_feat_names)}")
    
    # Define Target
    if TARGET_TYPE == 'win':
        label_func = lambda x: 1 if x == 1 else 0
    else:
        label_func = lambda x: 1 if x == 1 else 0
    df['target'] = df['rank'].apply(label_func)
    
    # Split
    train_mask = (df['year'] < VALID_YEAR) & (df['year'] >= pd.to_datetime(TRAIN_START).year)
    valid_mask = (df['year'] == VALID_YEAR)
    
    train_df = df[train_mask].reset_index(drop=True)
    valid_df = df[valid_mask].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    # --- Preprocessing Impl ---
    
    # 1. Numericals: FillNa -> Scale
    logger.info("Preprocessing Numericals...")
    # FillNa with 0 (or median?) 0 is standard for many features like simple counts.
    # For standardized features, 0 is mean.
    # Let's use 0.
    X_num_train = train_df[num_feat_names].fillna(0).values
    X_num_valid = valid_df[num_feat_names].fillna(0).values
    
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_valid = scaler.transform(X_num_valid)
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # 2. Categoricals: FillNa -> OrdinalEncode
    logger.info("Preprocessing Categoricals...")
    encoders = {}
    embedding_dims = []
    
    X_cat_train_list = []
    X_cat_valid_list = []
    
    for cat_col in cat_feat_names:
        # Fill Na
        train_df[cat_col] = train_df[cat_col].astype(str).fillna('nan')
        valid_df[cat_col] = valid_df[cat_col].astype(str).fillna('nan')
        
        # Encoder
        # Use encoded_value strategy
        # However, we want 0 to be <UNK> or something.
        # Simple strategy: Fit on Train. 
        # Transform Train.
        # Transform Valid with handle_unknown.
        
        # We need distinct count.
        # But OrdinalEncoder maps to 0..N-1.
        # Unknowns? 
        # Standard: use handle_unknown='use_encoded_value', unknown_value=-1
        # Then shift +1. So Unknown=0, Others=1..N.
        
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Fit
        raw_train = train_df[[cat_col]].values
        oe.fit(raw_train)
        
        # Transform
        enc_train = oe.transform(raw_train)
        enc_valid = oe.transform(valid_df[[cat_col]].values)
        
        # Shift + 1
        enc_train = enc_train + 1
        enc_valid = enc_valid + 1
        
        # Save
        encoders[cat_col] = oe
        X_cat_train_list.append(enc_train)
        X_cat_valid_list.append(enc_valid)
        
        # Determine Embedding Dim
        num_cats = len(oe.categories_[0]) + 1 # +1 for unknown/padding
        # Rule of thumb: min(50, (x+1)//2) or sqrt
        emb_dim = min(50, (num_cats + 1) // 2)
        if 'embedding_dim_rule' in cfg['model_params']:
             # Simple rule
             pass
             
        embedding_dims.append((num_cats, emb_dim))
        
    X_cat_train = np.hstack(X_cat_train_list).astype(int) if X_cat_train_list else np.zeros((len(train_df), 0), dtype=int)
    X_cat_valid = np.hstack(X_cat_valid_list).astype(int) if X_cat_valid_list else np.zeros((len(valid_df), 0), dtype=int)
    
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(embedding_dims, os.path.join(MODEL_DIR, "embedding_dims.pkl"))
    
    # Targets
    y_train = train_df['target'].values
    y_valid = valid_df['target'].values
    
    # Datasets
    train_ds = TabularDataset(X_num_train, X_cat_train, y_train)
    valid_ds = TabularDataset(X_num_valid, X_cat_valid, y_valid)
    
    batch_size = cfg['model_params']['batch_size']
    num_workers = cfg['resources'].get('num_workers', 0)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = SimpleMLP(
        num_numerical_features=len(num_feat_names),
        embedding_dims=embedding_dims,
        hidden_dims=cfg['model_params']['hidden_dims'],
        dropout_rate=cfg['model_params']['dropout_rate']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['model_params']['learning_rate']))
    criterion = nn.BCELoss()
    
    # Training Loop
    epochs = cfg['model_params']['epochs']
    patience = cfg['model_params']['early_stopping_patience']
    
    best_loss = float('inf')
    counter = 0
    
    logger.info("Training MLP...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for xn, xc, y in train_dl:
            xn, xc, y = xn.to(device), xc.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(xn, xc).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dl)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xn, xc, y in valid_dl:
                xn, xc, y = xn.to(device), xc.to(device), y.to(device)
                outputs = model(xn, xc).squeeze()
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        avg_val_loss = val_loss / len(valid_dl)
        val_auc = roc_auc_score(all_targets, all_preds)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
            # Save metrics
            with open(os.path.join(MODEL_DIR, "metrics.json"), 'w') as f:
                json.dump({'val_loss': avg_val_loss, 'val_auc': val_auc, 'epoch': epoch+1}, f)
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered")
                break
                
    logger.info("Training Complete!")

if __name__ == "__main__":
    main()
