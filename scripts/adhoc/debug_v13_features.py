
import pandas as pd
import joblib
import os
import sys
import logging
from datetime import datetime, timedelta

# Add workspace path
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
V13_OPTIMIZED_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'
V13_MODEL_PATH = 'models/experiments/exp_lambdarank_hard_weighted/model.pkl'

def load_v13_feature_list(path, logger):
    try:
        feats_df = pd.read_csv(path)
        if 'feature' in feats_df.columns:
            return feats_df['feature'].tolist()
        if '0' in feats_df.columns:
            return feats_df['0'].tolist()
        return feats_df.iloc[:, 0].tolist()
    except Exception as e:
        logger.error(f"Failed to load V13 feature list: {e}")
        return []

def main():
    logger.info("--- DEBUG V13 FEATURE GENERATION ---")
    
    # 1. Load Feature List
    if not os.path.exists(V13_OPTIMIZED_PATH):
        logger.error(f"Feature list not found: {V13_OPTIMIZED_PATH}")
        return
        
    feats_v13 = load_v13_feature_list(V13_OPTIMIZED_PATH, logger)
    logger.info(f"Expected Features Count: {len(feats_v13)}")
    logger.info(f"Example Expected: {feats_v13[:5]}")
    
    # 2. Load Sample Data (Today's race - 2026-01-25)
    target_date = "2026-01-25"
    logger.info(f"Loading data for {target_date}...")
    
    loader = JraVanDataLoader()
    # Use short history for debug speed? No, V13 needs long history.
    # But for debug, we assume cache exists or we can compute on small subset.
    # To check "Generation", strictly we need history.
    # Let's try loading just 2026 to see if basic cols appear.
    df_raw = loader.load(history_start_date="2026-01-01", end_date=target_date, skip_odds=False)
    
    if df_raw.empty:
        logger.error("No data loaded!")
        return
        
    logger.info(f"Loaded Raw Data: {df_raw.shape}")
    
    # 3. Generate Features
    CACHE_DIR = 'data/features_v14/prod_cache' # Use valid cache dir
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    blocks = list(pipeline.registry.keys())
    
    # Limit to small subset of df_raw for speed?
    # FeaturePipeline usually processes whole DF.
    logger.info("Generating Features (subset blocks)...")
    # Just try loading features
    df_features = pipeline.load_features(df_raw, blocks)
    logger.info(f"Generated Features Shape: {df_features.shape}")
    
    # 4. Check Intersection
    generated_cols = set(df_features.columns)
    missing_cols = [c for c in feats_v13 if c not in generated_cols]
    
    logger.info(f"Missing Columns Count: {len(missing_cols)}")
    if missing_cols:
        logger.warning(f"Missing Samples: {missing_cols[:10]}")
        
    # 5. Check Values
    common_cols = [c for c in feats_v13 if c in generated_cols]
    if common_cols:
        sample_df = df_features[common_cols].head()
        logger.info(f"Sample Values:\n{sample_df.iloc[:, :5]}")
        
        # Check for all-zeros
        sums = df_features[common_cols].sum()
        zero_cols = sums[sums == 0].index.tolist()
        logger.warning(f"Columns with SUM=0 (suspicious): {len(zero_cols)}")
        if zero_cols:
            logger.warning(f"Zero Col Samples: {zero_cols[:10]}")
            
    # 6. Model Prediction Check
    if os.path.exists(V13_MODEL_PATH):
        logger.info("Loading Model to test prediction...")
        model = joblib.load(V13_MODEL_PATH)
        
        # Build X
        X = df_features.reindex(columns=feats_v13, fill_value=0.0).fillna(0.0)
        logger.info(f"X shape: {X.shape}")
        
        try:
            preds = model.predict(X)
            logger.info(f"Predictions stats: Min={preds.min()}, Max={preds.max()}, Mean={preds.mean()}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
if __name__ == "__main__":
    main()
