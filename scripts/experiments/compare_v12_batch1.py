
"""
Compare v12 Batch 1 Model with Baselines
========================================
Models:
1. v12 Batch 1 (LambdaRank)
2. exp_lambdarank (Baseline LambdaRank)
3. exp_t2_refined_v3 (Production Binary Model)

Metrics:
- NDCG@3
- Win Return (Strategy: Bet Top 1)
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import ndcg_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scripts.experiments.exp_lambdarank_v12_batch1 import load_data, prepare_lgb_dataset, evaluate_ndcg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
V12_MODEL = "models/experiments/exp_lambdarank_v12_batch1/model.pkl"
BASE_LTR_MODEL = "models/experiments/exp_lambdarank/model.pkl"
PROD_MODEL = "models/experiments/exp_t2_refined_v3/model.pkl"

DATA_PATH = "data/processed/preprocessed_data_v12.parquet"

def evaluate_roi(model, df, x_cols, model_name="Model", X_pred=None):
    """Simple ROI calc: Bet Top1 Horse (Win)"""
    logger.info(f"Evaluating ROI for {model_name}...")
    
    # Predict
    if X_pred is not None:
        preds = model.predict(X_pred)
    else:
        preds = model.predict(df[x_cols])
    
    # Create eval df (group by race)
    df_eval = df[['race_id', 'horse_number', 'rank', 'odds', 'is_win']].copy()
    df_eval['score'] = preds
    
    results = []
    
    for rid, grp in df_eval.groupby('race_id'):
        # Top 1 selection
        # Sort descending by score
        top1 = grp.sort_values('score', ascending=False).iloc[0]
        
        bet_amount = 100
        return_amount = 0
        if top1['rank'] == 1:
            return_amount = bet_amount * top1['odds']
            
        results.append({
            'race_id': rid,
            'bet': bet_amount,
            'return': return_amount
        })
        
    df_res = pd.DataFrame(results)
    total_bet = df_res['bet'].sum()
    total_return = df_res['return'].sum()
    roi = (total_return / total_bet) * 100 if total_bet > 0 else 0
    
    print(f"[{model_name}] Total Bet: {total_bet}, Return: {total_return:.0f}, ROI: {roi:.2f}%")
    return roi

def main():
    logger.info("Loading Data...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"{DATA_PATH} not found.")
        return

    df = load_data()
    # Test set only (2024)
    df_test = df[df['date'].dt.year == 2024].copy()
    if 'is_win' not in df_test.columns and 'rank' in df_test.columns:
        df_test['is_win'] = (df_test['rank'] == 1).astype(int)
    logger.info(f"Test Set: {len(df_test)} rows")
    
    # Feature Columns management
    # Each model might use different features.
    # v12 model uses features in 'features.csv' or model.feature_name()
    # prod model uses exp_t2_refined_v3 features.
    # We rely on LightGBM to handle missing columns? No, strict match usually.
    # But usually models store feature_name().
    
    models = [
        ("v12_Batch1", V12_MODEL),
        ("Baseline_LTR", BASE_LTR_MODEL),
        ("Prod_Binary", PROD_MODEL)
    ]
    
    for name, path in models:
        logger.info(f"\n--- {name} ---")
        if not os.path.exists(path):
            logger.warning(f"{path} not found. Skipping.")
            continue
            
        model = joblib.load(path)
        
        # Get feature names from model
        model_feats = model.feature_name()
        
        # Check if all features exist in df_test
        missing = [f for f in model_feats if f not in df_test.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features for {name}: {missing[:5]}...")
            # If critical features missing, metrics will be garbage.
            # v12 data should have v3 features if loader/pipeline is superset.
            # But v3 features might use old names?
            # Usually v12 is superset.
        
        # Prepare X
        # LightGBM handles unused columns fine if we pass dataframe with correct columns?
        # Actually `model.predict(df[model_feats])`
        
        try:
            # Prepare data (Numpy bypass to avoid categorical mismatch)
            X_test_sub = df_test[model_feats].copy()
            
            # Categorical handling (Codes) as per production script
            for c in X_test_sub.columns:
                if X_test_sub[c].dtype.name == 'category' or X_test_sub[c].dtype == 'object':
                    # Use existing codes if possible, or blind encode (Risk of mismatch but matches prod script logic)
                    X_test_sub[c] = X_test_sub[c].astype('category').cat.codes
                else:
                    X_test_sub[c] = X_test_sub[c].fillna(-999.0)
            
            # Numpy Bypass
            X_val = X_test_sub.values.astype(np.float32)
            
            # NDCG
            ndcg = evaluate_ndcg(model, X_val, df_test['relevance'], df_test['race_id'].values, k=3)
            print(f"[{name}] NDCG@3: {ndcg:.4f}")
            
            # ROI
            evaluate_roi(model, df_test, model_feats, model_name=name, X_pred=X_val)
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

if __name__ == "__main__":
    main()
