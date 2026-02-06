
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import logging
import pandas as pd
import numpy as np
import yaml
import torch
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

# Add src
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting TabNet Regressor Experiment (Optimized)...")
    
    # 1. Config
    CFG_PATH = "config/experiments/exp_t2_refined.yaml"
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    feature_blocks = cfg['features']
    cat_cols_cfg = cfg['dataset']['categorical_features']
    
    # 2. Load Data
    end_date = "2024-12-31"
    
    loader = JraVanDataLoader()
    df = loader.load(history_start_date="2018-01-01", end_date=end_date, skip_odds=True)
    
    # Generate Features
    pipeline = FeaturePipeline(cache_dir="data/features")
    
    # Save rank explicitly
    rank_target = df['rank'].copy()
    
    df = pipeline.load_features(df, feature_blocks)
    
    # Restore rank
    df['rank'] = rank_target.values
    
    # 3. Preprocessing for TabNet Regressor
    logger.info("Preprocessing for TabNet Regressor (Reciprocal Rank)...")
    
    # Filter valid ranks for Training
    # We want to train on valid ranks. 
    # For DNF (NaN), we can either drop or set target to 0. 
    # Let's drop NaN ranks for training stability, but kept for simplicity in code logic below (fillna).
    
    # Create Reciprocal Rank Target: 1 / rank
    # 1st -> 1.0, 2nd -> 0.5, 3rd -> 0.33... 18th -> 0.05
    # DNF -> 0.0
    df['rank_numeric'] = pd.to_numeric(df['rank'], errors='coerce')
    # Fill NaN with 99 (DNF)
    df['rank_numeric'] = df['rank_numeric'].fillna(99)
    # Ensure 0 is treated as invalid (Rank should be >= 1)
    df['rank_numeric'] = df['rank_numeric'].replace(0, 99)
    
    df['target'] = 1.0 / df['rank_numeric']
    df.loc[df['rank_numeric'] >= 99, 'target'] = 0.0
    
    # Check Target distribution
    logger.info(f"Target Stats: Min={df['target'].min()}, Max={df['target'].max()}, NaN={df['target'].isna().sum()}, Inf={np.isinf(df['target']).sum()}")
    
    df['date'] = pd.to_datetime(df['date'])
    train_mask = df['date'] < "2024-01-01"
    test_mask = df['date'] >= "2024-01-01"
    
    # Define Columns
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'rank_numeric', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'updated_at', 'created_at', 'horse_name', 'jockey_name', 'trainer_name', 'race_name', 'venue', 'weather', 'surface', 'track_condition']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_processed = df[feature_cols].copy()
    
    cat_idxs = []
    cat_dims = []
    
    cat_features = []
    num_features = []
    
    for i, col in enumerate(feature_cols):
        # Strict check based on Config + Object type
        if col in cat_cols_cfg or X_processed[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_processed[col]):
            cat_features.append(col)
        else:
            num_features.append(col)

    # 3.1 Numeric Scaling
    logger.info(f"Scaling {len(num_features)} numeric features...")
    # Ensure no NaN/Inf in Numerics
    X_processed[num_features] = X_processed[num_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    # Fit on TRAIN only
    X_processed.loc[train_mask, num_features] = scaler.fit_transform(X_processed.loc[train_mask, num_features])
    X_processed.loc[test_mask, num_features] = scaler.transform(X_processed.loc[test_mask, num_features])
    
    # 3.2 Categorical Encoding
    logger.info(f"Encoding {len(cat_features)} categorical features...")
    for i, col in enumerate(cat_features):
        X_processed[col] = X_processed[col].astype(str).fillna("MISSING")
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        
        # Get index in feature_cols list
        idx = feature_cols.index(col)
        cat_idxs.append(idx)
        cat_dims.append(len(le.classes_) + 1) # +1 for safety
    
    # Verify Indices
    for i, idx in enumerate(cat_idxs):
        col_name = feature_cols[idx]
        max_val = X_processed.iloc[:, idx].max()
        dim = cat_dims[i]
        logger.info(f"Feature {col_name}: Max={max_val}, Dim={dim}")
        if max_val >= dim:
            logger.warning(f"Feature {col_name} Max {max_val} >= Dim {dim}. Adjusting Dim to {int(max_val + 2)}.")
            cat_dims[i] = int(max_val + 2)

    # Final check for safety
    X_processed = X_processed.fillna(0)
    
    X = X_processed.values.astype(np.float32)

    # Check NaN in final X
    if np.isnan(X).any():
        logger.warning("NaN found in X after processing. Filling with 0.")
        X = np.nan_to_num(X)

    y = df['target'].values.reshape(-1, 1)

    # Revert to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_valid = X[test_mask]
    y_valid = y[test_mask]
    logger.info(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
    
    # --- Strict Validation Block ---
    logger.info("Performing strict data validation before training...")
    
    # 1. NaN/Inf Check
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        raise ValueError("X_train contains NaN or Inf!")
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        raise ValueError("y_train contains NaN or Inf!")

    # 2. Categorical Index Check
    # X_train is numpy array (float32). We need to cast back to int to check indices.
    # Note: X_train columns correspond to feature_cols order.
    # cat_idxs stores the column indices of categorical features.
    
    X_train_cat = X_train[:, cat_idxs].astype(int)
    
    for i, (col_idx, dim) in enumerate(zip(cat_idxs, cat_dims)):
        col_name = feature_cols[col_idx]
        vals = X_train_cat[:, i] # the i-th categorical feature in the subset
        
        min_val = vals.min()
        max_val = vals.max()
        
        if min_val < 0 or max_val >= dim:
            error_msg = f"Categorical Index Error! Feature '{col_name}' (idx={col_idx}) has values outside [0, {dim}). Range: [{min_val}, {max_val}]"
            logger.error(error_msg)
            # Find specific bad indices info
            bad_mask = (vals < 0) | (vals >= dim)
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) > 0:
                first_bad = bad_indices[0]
                logger.error(f"Sample invalid value at row {first_bad}: {vals[first_bad]}")
                
            raise ValueError(error_msg)
    
    logger.info("Validation passed. Starting fit...")
    # -------------------------------
    
    # 4. Train TabNetRegressor (Optimized)
    clf = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=2, # Increased to 2 for safety
        
        # Architecture Tuning (Conservative)
        n_d=16, n_a=16, 
        n_steps=3,
        
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=5e-3),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', # Changed from sparsemax
        
        device_name=device,
        verbose=5
    )
    
    # Early Stopping based on MSE
    max_epochs = 200
    patience = 20 # Stop if MSE doesn't improve
    
    # Evaluation metric: MSE
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['mse'],
        max_epochs=max_epochs, 
        patience=patience,
        batch_size=256,      # Reduced batch size
        virtual_batch_size=64, # Small GBN
        num_workers=0,
        drop_last=False
    )
    
    # 5. Evaluation
    logger.info("Predicting...")
    preds = clf.predict(X_valid) # Shape (N, 1)
    preds = preds.flatten()
    
    # Check simple MSE
    mse = mean_squared_error(y_valid, preds)
    logger.info(f"Valid MSE: {mse:.6f}")
    
    # 6. Evaluation as Ranking (AUC & ROI)
    # Target was 1/rank. Higher is better.
    # Convert back to rank prediction? No need, 'score' is what matters.
    # To calculate AUC, we need Binary Target (1st or not).
    y_binary = (df[test_mask]['rank'] == 1).astype(int).values
    
    auc = roc_auc_score(y_binary, preds)
    logger.info(f"Valid AUC (converted from Regression): {auc:.4f}")
    
    # Feature Importance
    feat_importances = clf.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': feat_importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print("\nFeature Importance (TabNet Regressor):")
    print(imp_df.head(20).to_string(index=False))

    # 7. ROI Simulation
    logger.info("Running ROI Simulation on Validation Set (2024)...")
    
    df_valid = df[test_mask].copy()
    df_valid['pred_score'] = preds
    
    # Load Odds
    logger.info("Fetching Odds for 2024...")
    try:
        query = "SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '2024'"
        df_odds = pd.read_sql(query, loader.engine)
        
        odds_data = []
        for _, row in df_odds.iterrows():
             rid = row['race_id']
             raw = str(row['odds_tansho'])
             for i in range(0, len(raw), 8):
                 hn = raw[i:i+2]
                 od = raw[i+2:i+6]
                 if hn.isdigit() and od.isdigit():
                     odds_data.append({
                         'race_id': rid,
                         'horse_number': int(hn),
                         'odds': int(od)/10.0
                     })
        df_odds_parsed = pd.DataFrame(odds_data)
        
        df_valid['horse_number'] = df_valid['horse_number'].astype(int)
        df_valid = pd.merge(df_valid, df_odds_parsed, on=['race_id', 'horse_number'], how='left')
        
    except Exception as e:
        logger.error(f"Failed to load odds: {e}")
        return

    # ROI Calculation
    # We don't have probability (0-1), we have score (approx 0.0 - 1.0 reciprocal rank).
    # 1st place target is 1.0. 2nd is 0.5.
    # A score > 0.5 implies predicted as 1st or 2nd?
    # Let's try thresholds on Score.
    
    # Scores distribution?
    print(f"\nScore Distribution: Min={preds.min():.3f}, Max={preds.max():.3f}, Mean={preds.mean():.3f}")
    
    # Strategies based on score thresholds
    # Maybe 0.3, 0.4, 0.5?
    results = []
    
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        bets = df_valid[df_valid['pred_score'] >= th].copy()
        
        for odds_th in [1.5, 2.0, 3.0]:
            sub_bets = bets[bets['odds'] >= odds_th]
            if sub_bets.empty: continue
            
            hits = sub_bets[sub_bets['rank'] == 1]
            return_amt = hits['odds'].sum() * 100
            stake = len(sub_bets) * 100
            roi = return_amt / stake * 100
            
            results.append({
                'Condition': f"Score>={th}, Odds>={odds_th}",
                'ROI': roi,
                'Bets': len(sub_bets),
                'Returns': return_amt - stake
            })
            
    res_df = pd.DataFrame(results).sort_values('ROI', ascending=False)
    print("\n========= TabNet ROI Simulation (2024) =========")
    print(res_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
