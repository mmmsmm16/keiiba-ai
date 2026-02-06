
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import log_loss, roc_auc_score

sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="T2 Training Script - 2025 Walk-Forward")
    parser.add_argument("--config", type=str, default="config/experiments/exp_t2_refined_v3.yaml", 
                        help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Override for 2025 Walk-Forward
    EXP_NAME = "exp_t2_refined_v3_2025"
    MODEL_DIR = f"models/experiments/{EXP_NAME}"
    VALID_YEAR = 2025  # Key Change: Train on <2025, Validate on 2025
    
    ds_cfg = cfg['dataset']
    TRAIN_START = ds_cfg.get('train_start_date', '2014-01-01')
    
    logger.info(f"🚀 Starting T2 Training [{EXP_NAME}] (Walk-Forward 2025)")
    logger.info(f"  Train: {TRAIN_START} ~ 2024, Valid: 2025")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save config copy
    with open(os.path.join(MODEL_DIR, "config_copy.yaml"), 'w') as f:
        yaml.dump(cfg, f)

    # Paths - Use temp_merge_current which has 2024/2025 data
    FEAT_path = "data/features/temp_merge_current.parquet"
    
    if not os.path.exists(FEAT_path):
        logger.error(f"Missing features: {FEAT_path}")
        return

    # Load
    logger.info("Loading Data...")
    try:
        df = pd.read_parquet(FEAT_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
        
    # Ensure keys
    df['race_id'] = df['race_id'].astype(str)
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').fillna(0).astype(int)

    # Load targets (rank) if missing
    if 'rank' not in df.columns:
        TGT_path = "data/temp_t2/T2_targets.parquet"
        if os.path.exists(TGT_path):
            logger.info(f"Loading targets from {TGT_path}...")
            df_targets = pd.read_parquet(TGT_path)
            df_targets['race_id'] = df_targets['race_id'].astype(str)
            df_targets['horse_number'] = pd.to_numeric(df_targets['horse_number'], errors='coerce').fillna(0).astype(int)
            
            # Only keep rank
            merge_cols = ['race_id', 'horse_number']
            if 'rank' in df_targets.columns:
                merge_cols.append('rank')
            
            df = pd.merge(df, df_targets[merge_cols], on=['race_id', 'horse_number'], how='left')
            logger.info(f"Merged rank. Non-null rank: {df['rank'].notna().sum()}")
        else:
            logger.error(f"rank column missing and {TGT_path} not found!")
            return
    
    # Check for 2025 rank missing (common issue - T2_targets usually doesn't have 2025)
    df_2025_check = df[df['race_id'].str.startswith('2025')]
    if df_2025_check['rank'].isna().all():
        logger.info("2025 rank missing - loading from database...")
        from preprocessing.loader import JraVanDataLoader
        loader = JraVanDataLoader()
        df_2025_raw = loader.load(history_start_date='2025-01-01', end_date='2025-12-31', skip_training=True, skip_odds=True)
        df_2025_raw['race_id'] = df_2025_raw['race_id'].astype(str)
        df_2025_raw['horse_number'] = pd.to_numeric(df_2025_raw['horse_number'], errors='coerce').fillna(0).astype(int)
        
        # Merge 2025 rank
        df = pd.merge(df, df_2025_raw[['race_id', 'horse_number', 'rank']], 
                      on=['race_id', 'horse_number'], how='left', suffixes=('', '_2025'))

        
        # Fill in 2025 rank where missing
        mask = df['rank'].isna() & df['rank_2025'].notna()
        df.loc[mask, 'rank'] = df.loc[mask, 'rank_2025']
        df.drop(columns=['rank_2025'], errors='ignore', inplace=True)
        logger.info(f"After 2025 merge - Non-null rank: {df['rank'].notna().sum()}")

    if 'date' not in df.columns:


        logger.error("Date column missing!")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    
    # Split
    logger.info(f"Splitting: Train < {VALID_YEAR}, Valid == {VALID_YEAR}")
    train_mask = (df['year'] < VALID_YEAR) & (df['year'] >= pd.to_datetime(TRAIN_START).year)
    valid_mask = (df['year'] == VALID_YEAR)
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    
    logger.info(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}")
    
    if train_df.empty:
        logger.error("Train Set Empty!")
        return
        
    if valid_df.empty:
        logger.error("Valid Set (2025) Empty! Check if T2_features.parquet includes 2025 data.")
        return
        
    # Prepare X, y
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'updated_at', 'created_at']
    
    # Add config-specified exclude_features (e.g., ID features)
    exclude_features = ds_cfg.get('exclude_features', [])
    if exclude_features:
        logger.info(f"Excluding features from config: {len(exclude_features)} items")
        exclude_cols.extend(exclude_features)
    
    # Filter X columns
    X_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Remove datetime cols from X
    X_cols = [c for c in X_cols if not pd.api.types.is_datetime64_any_dtype(df[c])]
    
    logger.info(f"Feature count: {len(X_cols)}")
    
    # Categoricals
    cat_cols = cfg['dataset'].get('categorical_features', [])
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
            valid_df[col] = valid_df[col].astype('category')
            
    # Auto-cat other objects
    for col in X_cols:
        if train_df[col].dtype == 'object':
             train_df[col] = train_df[col].astype('category')
             valid_df[col] = valid_df[col].astype('category')
             
    X_train = train_df[X_cols]
    y_train = (train_df['rank'] == 1).astype(int) # Win Target
    
    X_valid = valid_df[X_cols]
    y_valid = (valid_df['rank'] == 1).astype(int)
    
    # Train
    params = cfg['model_params']
    exclude_params = ['early_stopping_rounds', 'n_estimators', 'model_type']
    lgbm_params = {k:v for k,v in params.items() if k not in exclude_params}
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
    
    logger.info("Training LightGBM...")
    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=params.get('n_estimators', 3000),
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(params.get('early_stopping_rounds', 100))]
    )
    
    # Save
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # ========== OOF Calibration ==========
    logger.info("🔮 Starting OOF Calibration (Isotonic)...")
    from sklearn.model_selection import KFold
    from sklearn.isotonic import IsotonicRegression
    
    n_folds = 5
    oof_probs = np.zeros(len(y_train))
    
    # KFold on race_id to prevent leakage within races
    train_df_with_id = train_df.copy()
    unique_races = train_df_with_id['race_id'].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_races)):
        logger.info(f"  OOF Fold {fold+1}/{n_folds}...")
        train_race_ids = unique_races[train_idx]
        val_race_ids = unique_races[val_idx]
        
        mask_t = train_df_with_id['race_id'].isin(train_race_ids)
        mask_v = train_df_with_id['race_id'].isin(val_race_ids)
        
        X_t, y_t = X_train[mask_t], y_train[mask_t]
        X_v = X_train[mask_v]
        
        # Train fold model (reduced rounds for speed)
        lgb_t = lgb.Dataset(X_t, label=y_t)
        fold_params = lgbm_params.copy()
        fold_model = lgb.train(
            fold_params,
            lgb_t,
            num_boost_round=min(params.get('n_estimators', 500), 500),
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        fold_probs = fold_model.predict(X_v)
        oof_probs[mask_v.values] = fold_probs
    
    # Train calibrator on OOF predictions (Train data only - no leakage)
    oof_probs = np.clip(oof_probs, 0.0, 1.0)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_probs, y_train.values)
    
    # Save calibrator
    calibrator_path = os.path.join(MODEL_DIR, "calibrator.pkl")
    joblib.dump(calibrator, calibrator_path)
    logger.info(f"✅ Calibrator saved to {calibrator_path}")
    
    # Save OOF probs for debugging
    np.save(os.path.join(MODEL_DIR, "oof_probs.npy"), oof_probs)
    
    # ========== Metrics (with and without calibration) ==========
    preds_raw = model.predict(X_valid)
    preds_cal = calibrator.predict(preds_raw)
    
    auc_raw = roc_auc_score(y_valid, preds_raw)
    auc_cal = roc_auc_score(y_valid, preds_cal)
    ll_raw = log_loss(y_valid, preds_raw)
    ll_cal = log_loss(y_valid, preds_cal)
    
    logger.info(f"Valid (2025) AUC [Raw]: {auc_raw:.4f}, LogLoss: {ll_raw:.4f}")
    logger.info(f"Valid (2025) AUC [Calibrated]: {auc_cal:.4f}, LogLoss: {ll_cal:.4f}")

    # Prob Distribution (compare raw vs calibrated)
    logger.info("--- Raw Predictions ---")
    logger.info(f"Mean Prob: {preds_raw.mean():.4f}")
    logger.info(f"Max Prob: {preds_raw.max():.4f}")
    logger.info(f"95% Prob: {np.percentile(preds_raw, 95):.4f}")
    logger.info("--- Calibrated Predictions ---")
    logger.info(f"Mean Prob: {preds_cal.mean():.4f}")
    logger.info(f"Max Prob: {preds_cal.max():.4f}")
    logger.info(f"95% Prob: {np.percentile(preds_cal, 95):.4f}")

    
    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': X_cols, 'gain': importance}).sort_values('gain', ascending=False)
    print("\nTop 20 Features:")
    print(imp_df.head(20))
    
    # Save Importance
    imp_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    
    logger.info("✅ Training Complete!")

if __name__ == "__main__":
    main()
