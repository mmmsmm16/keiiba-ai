import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle
import logging
from sklearn.isotonic import IsotonicRegression

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_v12_models():
    lgbm_path = "models/experiments/v12_win_lgbm/model.pkl"
    cat_path = "models/experiments/v12_win_cat/model.cbm"
    with open(lgbm_path, 'rb') as f:
        lgbm_model = pickle.load(f)
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier()
    cat_model.load_model(cat_path)
    return lgbm_model, cat_model

def get_data_and_predict(config_path, years=[2024, 2025]):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    loader = JraVanDataLoader()
    start_year = min(years)
    df_raw = loader.load(history_start_date=f"{start_year}-01-01", jra_only=True)
    cleanser = DataCleanser()
    df_clean = cleanser.cleanse(df_raw)
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_target = df_clean[df_clean['date'].dt.year.isin(years)].copy()
    pipeline = FeaturePipeline(cache_dir="data/features")
    df_features = pipeline.load_features(df_target, config.get('features', []))
    lgbm_model, cat_model = load_v12_models()
    feature_names = lgbm_model.feature_name()
    X = df_features.copy()
    cat_cols = ['horse_id', 'age', 'sex', 'jockey_id', 'trainer_id', 'sire_id', 'grade_code', 'kyoso_joken_code', 'surface', 'venue', 'prev_grade', 'dist_change_category', 'interval_category']
    cols_to_merge = ['race_id', 'horse_number', 'date', 'rank', 'odds']
    for col in cols_to_merge:
        if col not in X.columns:
            X = pd.merge(X, df_target[['race_id', 'horse_number', col]], on=['race_id', 'horse_number'], how='left')
    if 'year' not in X.columns:
        X['year'] = pd.to_datetime(X['date']).dt.year

    # Inference
    X_lgbm_df = X[feature_names].copy()
    for col in X_lgbm_df.columns:
        if X_lgbm_df[col].dtype == 'object' or isinstance(X_lgbm_df[col].dtype, pd.CategoricalDtype) or col in cat_cols:
            X_lgbm_df[col] = pd.factorize(X_lgbm_df[col])[0]
        X_lgbm_df[col] = pd.to_numeric(X_lgbm_df[col], errors='coerce').fillna(0)
    X_lgbm_values = X_lgbm_df.values.astype(np.float32)
    X_cat = X[feature_names].copy()
    for col in cat_cols:
        if col in X_cat.columns:
            vals = pd.to_numeric(X_cat[col], errors='coerce').fillna(-1)
            X_cat[col] = vals.astype(int).astype(str).replace('-1', 'None').replace('nan', 'None')
    lgbm_probs = lgbm_model.predict(X_lgbm_values)
    cat_probs = cat_model.predict_proba(X_cat)[:, 1]
    res = X[cols_to_merge + ['year']].copy()
    res['v12_score'] = (lgbm_probs + cat_probs) / 2
    engine = loader.engine
    df_hr = pd.read_sql(f"SELECT * FROM jvd_hr WHERE kaisai_nen >= '2025'", engine)
    df_hr['race_id'] = (
        df_hr['kaisai_nen'].astype(str).str.strip() + 
        df_hr['keibajo_code'].astype(str).str.strip() + 
        df_hr['kaisai_kai'].astype(str).str.strip().str.zfill(2) + 
        df_hr['kaisai_nichime'].astype(str).str.strip().str.zfill(2) + 
        df_hr['race_bango'].astype(str).str.strip().str.zfill(2)
    )
    res['race_id'] = res['race_id'].astype(str).str.strip()
    return res, df_hr

def run_comparison(df, df_hr):
    test = df[df['year'] == 2025].copy()
    
    # Place Payout mapping
    place_maps = []
    for i in range(1, 4):
        h_col, p_col = f'haraimodoshi_fukusho_{i}a', f'haraimodoshi_fukusho_{i}b'
        if h_col in df_hr.columns and p_col in df_hr.columns:
            tmp = df_hr[['race_id', h_col, p_col]].copy()
            tmp.columns = ['race_id', 'horse_number', 'payout_place']
            place_maps.append(tmp)
    df_place_div = pd.concat(place_maps).dropna()
    df_place_div['horse_number'] = df_place_div['horse_number'].apply(lambda x: int(float(x)) if str(x).replace('.','',1).isdigit() else -1)
    df_place_div['payout_place'] = pd.to_numeric(df_place_div['payout_place'], errors='coerce').fillna(0)
    test = pd.merge(test, df_place_div, on=['race_id', 'horse_number'], how='left').fillna({'payout_place': 0})

    # Strategy: Model Rank 1
    test['model_rank'] = test.groupby('race_id')['v12_score'].rank(ascending=False, method='min')
    rank1 = test[test['model_rank'] == 1].copy()
    
    comparisons = []
    # Win ROI
    win_roi = (rank1[rank1['rank'] == 1]['odds'].sum() / (len(rank1) if len(rank1) > 0 else 1)) * 100
    # Place ROI
    place_roi = (rank1['payout_place'].sum() / (len(rank1) * 100 if len(rank1) > 0 else 1)) * 100

    comparisons.append({'Bet': 'Win', 'ROI': win_roi, 'Bets': len(rank1)})
    comparisons.append({'Bet': 'Place', 'ROI': place_roi, 'Bets': len(rank1)})

    return pd.DataFrame(comparisons)

if __name__ == "__main__":
    df, df_hr = get_data_and_predict("config/experiments/exp_v12_win_lgbm.yaml")
    res = run_comparison(df, df_hr)
    print("\n--- v12 ModelRank1 ROI Comparison (2025 Test) ---")
    print(res.to_string(index=False))
