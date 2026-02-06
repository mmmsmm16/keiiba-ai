
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the LambdaRank model (Phase 2 model)
MODEL_PATH = "models/experiments/exp_lambdarank/model.pkl"
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

# Features to verify
TARGET_FEATURES = [
    'last_nige_rate',
    'sire_heavy_win_rate',
    'avg_first_corner_norm'
]

def verify_features():
    logger.info("="*60)
    logger.info("🔍 VERIFYING RESTORED FEATURES (LambdaRank)")
    logger.info("="*60)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    # 1. Check Feature Existence
    model_features = model.feature_name()
    missing = [f for f in TARGET_FEATURES if f not in model_features]
    
    if missing:
        logger.error(f"❌ CRITICAL: The following features are MISSING from the model: {missing}")
        logger.error("   The retraining provided did not include these features.")
        sys.exit(1)
    else:
        logger.info(f"✅ All target features found in model columns.")
        
    # 2. Check Feature Importance (Gain) - Lightweight check
    logger.info("Checking feature importance (Gain)...")
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': model_features, 'gain': importance})
    imp_df = imp_df.sort_values('gain', ascending=False).reset_index(drop=True)
    imp_df['rank'] = imp_df.index + 1
    
    logger.info("\n📊 Feature Importance Check:")
    failed = False
    for f in TARGET_FEATURES:
        row = imp_df[imp_df['feature'] == f]
        if row.empty:
            logger.error(f"   ❌ {f}: Not found in importance list (This shouldn't happen)")
            failed = True
            continue
            
        rank = row['rank'].values[0]
        gain = row['gain'].values[0]
        total = len(imp_df)
        
        status = "✅ OK" if gain > 0 else "⚠️ ZERO IMPORTANCE"
        if gain == 0: failed = True
        
        logger.info(f"   - {f:<25} : Rank {rank:>3}/{total}  (Gain: {gain:.4f})  {status}")
        
    if failed:
        logger.error("\n❌ VERIFICATION FAILED: Some features have zero importance or are missing.")
        sys.exit(1)
        
    logger.info("\n✅ VERIFICATION PASSED: All features are present and contributing.")

if __name__ == "__main__":
    verify_features()
