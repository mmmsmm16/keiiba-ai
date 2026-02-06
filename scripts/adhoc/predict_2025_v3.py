
import sys
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml

sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Generating T2 Refined v3 Predictions for 2025...")
    
    MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
    OUTPUT_PATH = "data/temp_t2/T2_predictions_2025_only.parquet"
    CONFIG_PATH = "config/experiments/exp_t2_refined_v3.yaml"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
        
    # Load model
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    
    # Load config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    feature_blocks = cfg['features']
    
    # Load 2025 data
    loader = JraVanDataLoader()
    # Need to load enough history for lags? Loader handles lookback if date partitioning is correct.
    # But for simplicity, we rely on cached features if possible.
    # However, pipeline.load_features() will re-compute features from raw data.
    # To avoid cache issues, we just run on 2025 data.
    # Wait, loader.load() with start_date 2025 might missing history for lags.
    # But FeaturePipeline has internal logic or we assume features are enough.
    # Actually, we should check if we can just load T1 features directly?
    # T1 parquet file already exists!
    
    # Force generation from scratch to ensure full feature set
    # T1_features usually only contains odds related columns
    # t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    if False: # os.path.exists(t1_path):
        logger.info("Loading features directly from T1 parquet...")
        df_feat = pd.read_parquet(t1_path)
        # Filter for 2025
        # Parse year from race_id or assume it has 'date' if T1 was generated that way.
        # But check_t1_db.py showed T1 has race_id.
        logger.info(f"T1 columns: {list(df_feat.columns[:5])}")
        
        # We need 'date' and 'rank' for output.
        # T1 might not have them if it generated only features.
        # Let's load raw data to map date/rank.
        df_raw = loader.load(history_start_date='2025-01-01', end_date='2025-12-31', skip_training=False, skip_odds=True)
        if df_raw.empty:
            logger.error("No 2025 raw data found")
            return
            
        # Merge features with raw info
        # Key: race_id, horse_number
        df_feat['race_id'] = df_feat['race_id'].astype(str)
        df_raw['race_id'] = df_raw['race_id'].astype(str)
        
        # Filter T1 to 2025 based on race_id (YYYY...)
        df_feat = df_feat[df_feat['race_id'].str.startswith('2025')]
        
        df_merged = pd.merge(df_feat, df_raw[['race_id', 'horse_number', 'date', 'rank']], on=['race_id', 'horse_number'], how='inner')
        logger.info(f"Merged 2025 data: {len(df_merged)} rows")
        df_feat = df_merged
        
    else:
        # Check if full cache exists
        cache_file = "data/features/temp_merge_current.parquet"
        if os.path.exists(cache_file):
            logger.info(f"Loading features from cache: {cache_file}")
            df_feat = pd.read_parquet(cache_file)
            
            # Ensure 2025 data is present
            df_feat['race_id'] = df_feat['race_id'].astype(str)
            df_2025 = df_feat[df_feat['race_id'].str.startswith('2025')]
            
            if df_2025.empty:
                 logger.warning("Cache exists but no 2025 data found. Generating from scratch...")
                 # Fallback to generation
                 start_date = cfg['dataset'].get('train_start_date', '2014-01-01')
                 df_raw = loader.load(history_start_date=start_date, end_date='2025-12-31', skip_training=False, skip_odds=True)
                 pipeline = FeaturePipeline(cache_dir="data/features")
                 df_feat = pipeline.load_features(df_raw, feature_blocks)
            else:
                 logger.info(f"Loaded {len(df_feat)} rows from cache. 2025 count: {len(df_2025)}")
                 # We still need date/rank/odds if they are missing in cache (usually present in merge)
                 # temp_merge_current includes Keys + Features. Raw cols like 'rank', 'date', 'odds' might be missing if not in blocks.
                 # Let's check columns later.
        else:
            logger.info("Generating features from scratch...")
            start_date = cfg['dataset'].get('train_start_date', '2014-01-01')
            logger.info(f"Loading FULL history from {start_date} to ensure correct lag features...")
            
            df_raw = loader.load(history_start_date=start_date, end_date='2025-12-31', skip_training=False, skip_odds=True)
            pipeline = FeaturePipeline(cache_dir="data/features")
            
            df_feat = pipeline.load_features(df_raw, feature_blocks)
        
        # Filter for 2025 ONLY after feature generation/loading
        logger.info("Filtering for 2025 data...")
        df_feat['race_id'] = df_feat['race_id'].astype(str)
        df_feat = df_feat[df_feat['race_id'].str.startswith('2025')]
        
        # We need raw info for output (rank, date, etc) if not in df_feat
        # If loaded from cache, we might miss them if they are not features.
        # But 'date' is usually a feature input. 'rank' is target.
        # If missing, we load raw 2025 data to merge.
        cols_needed = ['date', 'rank', 'tansho_odds']
        missing = [c for c in cols_needed if c not in df_feat.columns]
        
        if missing:
             logger.info(f"Missing columns {missing} in features. Loading raw 2025 data to merge...")
             # skip_odds=False to ensure we get odds for output
             df_raw_25 = loader.load(history_start_date='2025-01-01', end_date='2025-12-31', skip_training=False, skip_odds=False)
             df_raw_25['race_id'] = df_raw_25['race_id'].astype(str)
             
             # Merge
             # Note: df_raw_25 might have 'odds' or 'tansho_odds'. JraVanDataLoader maps columns. usually 'odds'.
             # Let's check columns available.
             cols_to_use = ['race_id', 'horse_number']
             for c in missing:
                 if c in df_raw_25.columns:
                     cols_to_use.append(c)
                 elif c == 'tansho_odds' and 'odds' in df_raw_25.columns:
                     df_raw_25['tansho_odds'] = df_raw_25['odds']
                     cols_to_use.append(c)
             
             df_feat = pd.merge(df_feat, df_raw_25[cols_to_use], on=['race_id', 'horse_number'], how='left')
             
        logger.info(f"Features for 2025: {len(df_feat)} rows")

    if df_feat.empty:
        logger.error("No features available")
        return

    # Prepare X
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'updated_at', 'created_at']
    X_cols = [c for c in df_feat.columns if c not in exclude_cols]
    # Remove datetime cols
    X_cols = [c for c in X_cols if not pd.api.types.is_datetime64_any_dtype(df_feat[c])]
    
    # Categorical processing
    cat_cols = cfg['dataset'].get('categorical_features', [])
    for col in cat_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype('category')

    # --- Debug: Check Feature Distribution (2024 vs 2025) ---
    logger.info("--- Feature Distribution Check ---")
    # We expect 2024 data to be present and 2025 data to be present
    df_feat['year'] = pd.to_datetime(df_feat['date']).dt.year
    
    check_cols = ['horse_elo', 'speed_index_ewm', 'jockey_win_rate', 'impost_change', 'field_size']
    for c in check_cols:
        if c in df_feat.columns:
            mean_24 = df_feat[df_feat['year']==2024][c].mean()
            mean_25 = df_feat[df_feat['year']==2025][c].mean()
            null_24 = df_feat[df_feat['year']==2024][c].isnull().mean()
            null_25 = df_feat[df_feat['year']==2025][c].isnull().mean()
            logger.info(f"Feature {c}:")
            logger.info(f"  2024: Mean={mean_24:.4f}, Null={null_24:.2%}")
            logger.info(f"  2025: Mean={mean_25:.4f}, Null={null_25:.2%}")
    logger.info("----------------------------------")
            
    # Model expects specific columns?
    # LGBM handles extra cols usually, but matching order is safer.
    # We rely on feature names.
    
    # Predict
    logger.info("Generating predictions...")
    
    # Align features with model
    if hasattr(model, 'booster_'):
        booster = model.booster_
    else:
        # Raw Booster
        booster = model
        
    model_features = booster.feature_name()
    dump = booster.dump_model()
    
    # Check key missing
    missing_cols = [c for c in model_features if c not in df_feat.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        for c in missing_cols:
            df_feat[c] = 0 # Impute
            
    # Reorder
    X = df_feat[model_features].copy()
    
    # Manually Map Categories to Integers to support Numpy prediction
    # This bypasses all pandas metadata strictness and string errors
    if 'pandas_categorical' in dump and 'categorical_feature' in dump:
        logger.info("Applying Manual Categorical Encoding...")
        cat_indices = dump['categorical_feature']
        cat_values_list = dump['pandas_categorical']
        
        for i, idx in enumerate(cat_indices):
            col_name = model_features[idx]
            cats = cat_values_list[i]
            if not cats: continue
            
            # Create mapping dict: value -> code (index)
            # cats is the list of values. Index is the code.
            cat_map = {val: code for code, val in enumerate(cats)}
            
            if col_name in X.columns:
                # Apply mapping
                # Use map, fill NaN/unknown with NaN (or -1?)
                # LightGBM handles NaN in categorical usually.
                # But to be safe for int conversion, let's look at what LGBM does.
                # Usually NaN is fine if we leave it as NaN float.
                
                # We need to handle type mismatch (int vs str)
                # Ensure input column matches cat type
                if isinstance(cats[0], str):
                    X[col_name] = X[col_name].astype(str).map(cat_map)
                else:
                    # If cats are int
                    X[col_name] = X[col_name].map(cat_map)
                
                # Fill missing categories with NaN (or -1 if we want)
                # X[col_name] = X[col_name].fillna(-1)
                
                logger.info(f"Encoded {col_name}")

    # Ensure all object columns are numeric now
    # If any remaining objects (not in model's cat list?), convert to numeric or drop
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.warning(f"Column {col} is still object. Trying to convert to numeric.")
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Final Sanitization Loop
    logger.info("Performing final data sanitization...")
    for col in X.columns:
        # Check if convertible
        try:
            pd.to_numeric(X[col], errors='raise')
        except:
            # If fail, force coerce
            # logger.warning(f"Column {col} is not numeric. Coercing...")
            # Check what's inside
            # logger.info(f"Sample values in {col}: {X[col].unique()[:3]}")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
    # Verify
    for col in X.columns:
         try:
             X[col].astype(np.float32)
         except Exception as e:
             logger.error(f"FINAL CHECK FAILED for {col}: {e}")
             logger.error(f"Sample: {X[col].unique()[:5]}")
             # Last ditch: fill 0
             X[col] = 0

    logger.info("Generating predictions (using numpy)...")
    
    # Ensure strict column order
    X = X[model_features]
    
    # Convert to float32 (compatible with LGBM)
    X_vals = X.values.astype(np.float32)
    
    # Predict using numpy
    if hasattr(model, 'predict_proba'):
        # Classifier
        preds = model.predict_proba(X.values)[:, 1]
    else:
        # Regressor or LGBM Booster
        preds = model.predict(X.values)
        
    df_feat['pred_prob'] = preds
    
    # Merge with raw data to get rank, date, odds if missing
    # Ensure df_raw exists (it does in fresh generation mode)
    if 'rank' not in df_feat.columns:
        logger.info("Merging rank and date from raw data...")
        # Check available columns in df_raw
        cols_to_merge = ['race_id', 'horse_number']
        for c in ['rank', 'date', 'tansho_odds']:
            if c in df_raw.columns:
                cols_to_merge.append(c)
        
        df_feat = pd.merge(df_feat, df_raw[cols_to_merge], on=['race_id', 'horse_number'], how='left')
        
        # Rename tansho_odds to odds_final if needed
        if 'tansho_odds' in df_feat.columns and 'odds_final' not in df_feat.columns:
             df_feat['odds_final'] = pd.to_numeric(df_feat['tansho_odds'], errors='coerce')

    # Create is_win
    if 'rank' in df_feat.columns:
        df_feat['is_win'] = (df_feat['rank'] == 1).astype(int)
    else:
        df_feat['is_win'] = 0

    output_cols = ['race_id', 'horse_number', 'date', 'rank', 'pred_prob', 'odds_final', 'is_win']

    # Ensure output cols exist
    if 'odds_final' not in df_feat.columns:
        df_feat['odds_final'] = 1.0 # Default if missing
        
    for c in output_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0

    logger.info(f"Saving 2025 predictions to {OUTPUT_PATH}")
    df_feat[output_cols].to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
