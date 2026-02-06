
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import traceback
import json
from sklearn.metrics import log_loss, roc_auc_score

sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader # if needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="T2 Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    EXP_NAME = cfg.get('experiment_name', 'exp_t2_unknown')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    
    ds_cfg = cfg['dataset']
    VALID_YEAR = ds_cfg.get('valid_year', 2023)
    TRAIN_START = ds_cfg.get('train_start_date', '2015-01-01')
    
    logger.info(f"🚀 Starting T2 Training [{EXP_NAME}]")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save config copy
    with open(os.path.join(MODEL_DIR, "config_copy.yaml"), 'w') as f:
        yaml.dump(cfg, f)

    # Paths
    FEAT_path = "data/temp_t2/T2_features.parquet"
    TGT_path = "data/temp_t2/T2_targets.parquet"
    
    if not os.path.exists(FEAT_path):
        logger.error(f"Missing features: {FEAT_path}")
        return

    # Load
    logger.info("Loading Data...")
    try:
        df_features = pd.read_parquet(FEAT_path)
        df_targets = pd.read_parquet(TGT_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
        
    # Merge
    # df_features might usually have metadata, but let's be safe
    # Ensure keys type match
    df_features['race_id'] = df_features['race_id'].astype(str)
    df_targets['race_id'] = df_targets['race_id'].astype(str)
    
    # Convert horse_number to int if needed, but usually float/int mismatch can handle if values ok.
    # Better safe:
    df_features['horse_number'] = pd.to_numeric(df_features['horse_number'], errors='coerce').fillna(0).astype(int)
    df_targets['horse_number'] = pd.to_numeric(df_targets['horse_number'], errors='coerce').fillna(0).astype(int)
    
    logger.info("Merging Features and Targets...")
    # Both features and targets have 'date', so include it in merge keys to avoid duplicates
    if 'date' in df_features.columns and 'date' in df_targets.columns:
        df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number', 'date'], how='inner')
    else:
        df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='inner')
    
    # Check Date
    if 'date' not in df.columns:
        # TGT usually has date
        logger.error("Date column missing after merge!")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Split
    logger.info(f"Splitting: Train < {VALID_YEAR}, Valid == {VALID_YEAR}, Test > {VALID_YEAR}")
    train_mask = (df['year'] < VALID_YEAR) & (df['year'] >= pd.to_datetime(TRAIN_START).year)
    valid_mask = (df['year'] == VALID_YEAR)
    test_mask = (df['year'] > VALID_YEAR)
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    test_df = df[test_mask].copy()
    
    if train_df.empty:
        logger.error("Train Set Empty!")
        return
        
    # Prepare X, y
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'updated_at', 'created_at']
    
    # Add config-specified exclude_features (e.g., ID features)
    exclude_features = ds_cfg.get('exclude_features', [])
    if exclude_features:
        logger.info(f"Excluding features from config: {exclude_features}")
        exclude_cols.extend(exclude_features)
    
    # Filter X columns
    X_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Remove datetime cols from X
    X_cols = [c for c in X_cols if not pd.api.types.is_datetime64_any_dtype(df[c])]
    
    logger.info(f"Feature count: {len(X_cols)}")
    
    # Categoricals
    cat_cols = cfg['dataset']['categorical_features']
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
    
    # Determine Target
    target_type = ds_cfg.get('binary_target', 'win') # win, top2, top3
    logger.info(f"Target Type: {target_type}")
    
    if target_type == 'top2':
        y_train = (train_df['rank'] <= 2).astype(int)
    elif target_type == 'top3':
        y_train = (train_df['rank'] <= 3).astype(int)
    else:
        y_train = (train_df['rank'] == 1).astype(int) # Default Win
    
    X_valid = valid_df[X_cols]
    if target_type == 'top2':
        y_valid = (valid_df['rank'] <= 2).astype(int)
    elif target_type == 'top3':
        y_valid = (valid_df['rank'] <= 3).astype(int)
    else:
        y_valid = (valid_df['rank'] == 1).astype(int)
    
    if not test_df.empty:
        # Preprocess test categorical BEFORE extracting X_test
        for col in cat_cols:
            if col in test_df.columns:
                test_df[col] = test_df[col].astype('category')
        for col in X_cols:
            if col in test_df.columns and test_df[col].dtype == 'object':
                 test_df[col] = test_df[col].astype('category')
        
        X_test = test_df[X_cols]
        if target_type == 'top2':
            y_test = (test_df['rank'] <= 2).astype(int)
        elif target_type == 'top3':
            y_test = (test_df['rank'] <= 3).astype(int)
        else:
            y_test = (test_df['rank'] == 1).astype(int)
        logger.info(f"Test (2024+): {X_test.shape}")
    
    if 'bias_adversity_score_mean_5' in X_cols:
        logger.info("CONFIRMED: bias_adversity_score_mean_5 is in feature set.")
    else:
        logger.warning("WARNING: bias_adversity_score_mean_5 NOT FOUND in features!")
    
    # Train
    params = cfg['model_params']
    exclude_params = ['early_stopping_rounds', 'n_estimators', 'model_type']
    lgbm_params = {k:v for k,v in params.items() if k not in exclude_params}
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
    
    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=params['n_estimators'],
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(params['early_stopping_rounds'])]
    )
    
    # Save
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # --- Evaluation ---
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator()

    # Need race_id and odds for advanced validation
    # Warning: valid_df contains these, but X_valid removed them.
    # We should pass valid_df column values directly.
    # We preserved 'odds_10min' or 'odds_final' in df? 
    # Check if odds cols were excluded or kept in original df.
    # They were in exclude_cols, so dropped from X, but valid_df still has them.
    
    odds_col = 'odds_10min' if 'odds_10min' in valid_df.columns else 'odds_final'
    if odds_col not in valid_df.columns:
        logger.warning("Odds column not found in validation DataFrame. ROI calcs will be skipped.")
        valid_odds = None
        test_odds = None
    else:
        valid_odds = valid_df[odds_col].fillna(0)
        test_odds = test_df[odds_col].fillna(0) if not test_df.empty else None

    # Predict Validation
    logger.info("Evaluating on Validation Set...")
    valid_preds = model.predict(X_valid)
    valid_metrics = evaluator.evaluate(
        y_true=y_valid, 
        y_prob=valid_preds, 
        group_ids=valid_df['race_id'], 
        odds=valid_odds
    )
    
    logger.info(f"== Validation Metrics ({VALID_YEAR}) ==")
    for k, v in valid_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Predict Test
    if not test_df.empty:
        logger.info("Evaluating on Test Set (2024+)...")
        test_preds = model.predict(X_test)
        test_metrics = evaluator.evaluate(
            y_true=y_test, 
            y_prob=test_preds, 
            group_ids=test_df['race_id'], 
            odds=test_odds
        )
        logger.info(f"== Test Metrics ({ds_cfg.get('test_end_date', '2024+')}) ==")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    
    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': X_cols, 'gain': importance}).sort_values('gain', ascending=False)
    print("\nTop 20 Features:")
    print(imp_df.head(20))
    
    # Save Importance
    imp_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    
    # Save Metrics to file for tracking
    metrics_summary = {
        'validation': valid_metrics,
        'test': test_metrics if not test_df.empty else {}
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    logger.info("Training and Evaluation Completed.")

if __name__ == "__main__":
    main()
