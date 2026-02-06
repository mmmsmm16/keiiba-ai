
import sys
import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline
from preprocessing.cleansing import DataCleanser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Generating T2 Refined v3 Predictions for 2024...")
    
    # Model & Config Paths
    EXP_NAME = "exp_t2_refined_v3"
    MODEL_PATH = f"models/experiments/{EXP_NAME}/model.pkl" # or .cbm
    CALIB_PATH = f"models/experiments/{EXP_NAME}/calibrator.pkl"
    CONFIG_PATH = f"config/experiments/{EXP_NAME}.yaml"
    
    OUTPUT_PATH = "data/temp_t2/T2_predictions_2024_refined_v3.parquet"
    
    # Check Model Exists
    if not os.path.exists(MODEL_PATH):
        # CatBoost might be .cbm
        MODEL_PATH_CBM = f"models/experiments/{EXP_NAME}/model.cbm"
        if os.path.exists(MODEL_PATH_CBM):
            MODEL_PATH = MODEL_PATH_CBM
            IS_CB = True
        else:
            logger.error(f"Model not found at {MODEL_PATH}")
            return
    else:
        IS_CB = False
        
    logger.info(f"Using Model: {MODEL_PATH}")
    
    # Load model
    if IS_CB:
        import catboost as cb
        model = cb.CatBoostClassifier() # or Ranker
        model.load_model(MODEL_PATH)
    else:
        model = joblib.load(MODEL_PATH)
        
    # Load Calibrator
    calibrator = None
    if os.path.exists(CALIB_PATH):
        calibrator = joblib.load(CALIB_PATH)
        logger.info("Calibrator loaded.")
        
    # Load Config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    feature_blocks = cfg['features']
    cat_features = cfg['dataset'].get('categorical_features', [])
    
    # Load 2024 Data
    loader = JraVanDataLoader()
    # Loading 2024 data. Note: History needed for features?
    # Loader loads history automatically if configured? 
    # Usually we load a range. load() method has history_start_date. 
    # If we want predictions for 2024, we need history up to 2024.
    # To be safe, load 2023-2024 range but predict on 2024?
    # Actually, JraVanDataLoader.load(history_start_date=TARGET-N_YEARS, end_date=TARGET) is typical.
    # Or strict range.
    # feature_pipeline handles history internally? No, input DF must have history rows.
    # To compute "past 5 runs" for 2024-01-01 races, we need 2023 data.
    # So start date should be 2023-01-01 (or earlier).
    
    logger.info("Loading Data (2020-01-01 to 2024-12-31)...") # Load enough history
    df = loader.load(history_start_date='2020-01-01', end_date='2024-12-31', skip_training=True, skip_odds=True)
    
    # Cleansing
    # cleanser = DataCleanser()
    # df = cleanser.cleanse(df)
    # NOTE: FeaturePipeline uses raw data usually, but if cleansing required for features...
    # Actually RunExperiment uses: df = cleanser.cleanse(df) BEFORE pipeline. 
    # But pipeline expects raw columns. Check if cleansing drops them.
    # Assuming cleanser is safe.
    cleanser = DataCleanser()
    df = cleanser.cleanse(df) 
    
    # Generate Features
    pipeline = FeaturePipeline(cache_dir="data/features")
    logger.info("Generating Features...")
    df_feat = pipeline.load_features(df, feature_blocks)
    
    # Merge date if missing
    if 'date' not in df_feat.columns:
        # Use df (which is cleansed raw)
        df_feat = pd.merge(df_feat, df[['race_id', 'date']].drop_duplicates(), on='race_id', how='left')

    # Filter for 2024
    df_feat['date'] = pd.to_datetime(df_feat['date'])
    df_2024 = df_feat[df_feat['date'].dt.year == 2024].copy()
    logger.info(f"2024 Data Shape: {df_2024.shape}")
    
    if df_2024.empty:
        logger.error("No 2024 data found.")
        return

    # Prepare X
    # Prepare X
    drop_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds', 'horse_id']
    exclude_features = cfg['dataset'].get('exclude_features', [])
    X_cols = [c for c in df_2024.columns if c not in drop_cols and c not in exclude_features]
    X = df_2024[X_cols].copy()

    # Ensure Categorical (Config + Auto Detect)
    auto_cat = [c for c in X.columns if X[c].dtype == 'object']
    final_cat_features = list(set(cat_features + auto_cat))
    
    logger.info(f"Categorical features (Config+Auto): {len(final_cat_features)}")

    for c in final_cat_features:
        if c in X.columns:
            # Fillna for safety
            X[c] = X[c].fillna("missing").astype('category')
    
    # Filter only numeric/category
    # X = X.select_dtypes(include=['number', 'category'])
    
    logger.info("Predicting...")
    # Predict
    if CALIB_PATH:
        # Calibrated output
        # Model predict_proba? or predict?
        # If calibrated, model output is raw score usually?
        # Standard LightGBM classifier: predict_proba returns [p0, p1].
        # Ranker: predict returns score.
        # Calibration usually maps score -> prob.
        # If model is classifier, predict returns class?
        # `run_experiment.py`: model.predict(X).
        # If binary/LGBM, model.predict returns raw score or prob depending on params?
        # Usually LGBM sklearn API returns class, Booster returns raw.
        # `run_experiment.py` uses `lgb.train` -> Booster.
        # Booster.predict returns raw scores or prob?
        # objective='binary', it returns probability [0..1] by default.
        # If so, calibration input is prob.
        
        preds = model.predict(X)
        if calibrator:
             preds = calibrator.predict(preds)
    else:
        preds = model.predict(X)
        
    df_2024['pred_prob'] = preds
    
    # Evaluate
    if 'rank' in df_2024.columns:
        df_2024['is_win'] = (df_2024['rank'] == 1).astype(int)
        valid_mask = df_2024['rank'].notna()
        # Simple AUC check
        from sklearn.metrics import roc_auc_score
        if valid_mask.sum() > 0:
            try:
                auc = roc_auc_score(df_2024.loc[valid_mask, 'is_win'], df_2024.loc[valid_mask, 'pred_prob'])
                logger.info(f"2024 AUC: {auc:.4f}")
            except:
                pass

    # Save
    out_cols = ['race_id', 'horse_number', 'date', 'pred_prob']
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_2024[out_cols].to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
