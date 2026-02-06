import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle
import logging
from sklearn.isotonic import IsotonicRegression
from itertools import product

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
    
    logger.info(f"Loading LGBM model from {lgbm_path}...")
    with open(lgbm_path, 'rb') as f:
        lgbm_model = pickle.load(f)
    
    logger.info(f"Loading CatBoost model from {cat_path}...")
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier()
    cat_model.load_model(cat_path)
        
    return lgbm_model, cat_model

def get_predictions_for_years(config_path, years=[2024, 2025]):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    loader = JraVanDataLoader()
    start_year = min(years)
    df_raw = loader.load(history_start_date=f"{start_year}-01-01", end_date="2025-12-31", jra_only=True)
    
    cleanser = DataCleanser()
    df_clean = cleanser.cleanse(df_raw)
    
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_target = df_clean[df_clean['date'].dt.year.isin(years)].copy()
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    feature_blocks = config.get('features', [])
    df_features = pipeline.load_features(df_target, feature_blocks)
    
    # Ensure year and key-based features
    if 'year' not in df_features.columns:
        df_features = pd.merge(df_features, df_target[['race_id', 'horse_number', 'date']], on=['race_id', 'horse_number'], how='left')
        df_features['year'] = pd.to_datetime(df_features['date']).dt.year

    lgbm_model, cat_model = load_v12_models()
    feature_names = lgbm_model.feature_name()
    X = df_features[feature_names].copy()
    
    # Categorical logic (Same as run_experiment.py)
    lgbm_cat_cols = [
        "jockey_id", "trainer_id", "sire_id", "horse_id", "sex", 
        "grade_code", "kyoso_joken_code", "surface", "venue", 
        "prev_grade", "dist_change_category", "interval_category"
    ]
    auto_cat = [c for c in X.columns if X[c].dtype == 'object']
    lgbm_cat_cols = list(set(lgbm_cat_cols + auto_cat))
    cat_cat_cols = list(set(lgbm_cat_cols + ["age"]))
    
    # LGBM predict (returns probabilities for binary)
    X_lgbm = X.copy()
    for col in lgbm_cat_cols:
        if col in X_lgbm.columns:
            X_lgbm[col] = X_lgbm[col].astype('category')
    lgbm_probs = lgbm_model.predict(X_lgbm)
    
    # CatBoost predict (returns probabilities)
    X_cat = X.copy()
    for col in cat_cat_cols:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str).replace('nan', '')
    # Get probabilities for class 1
    cat_probs = cat_model.predict_proba(X_cat)[:, 1]
    
    res_df = df_target[['race_id', 'horse_number', 'date', 'rank', 'odds', 'odds_10min']].copy()
    res_df['year'] = res_df['date'].dt.year
    res_df['lgbm_prob'] = lgbm_probs
    res_df['cat_prob'] = cat_probs
    # Mean ensemble of probabilities
    res_df['ensemble_prob'] = (lgbm_probs + cat_probs) / 2
    
    return res_df

def run_grid_search(df):
    train_df = df[df['year'] == 2024].copy()
    test_df = df[df['year'] == 2025].copy()
    
    logger.info("Calibrating probabilities with Isotonic Regression (2024 data)...")
    ir = IsotonicRegression(out_of_bounds='clip')
    # Target: Rank 1
    y_train = (train_df['rank'] == 1).astype(int)
    ir.fit(train_df['ensemble_prob'], y_train)
    
    # Calibrate 2025
    test_df['win_prob_calibrated'] = ir.predict(test_df['ensemble_prob'])
    # Normalize per race
    test_df['win_prob'] = test_df.groupby('race_id')['win_prob_calibrated'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    valid_sim = test_df.dropna(subset=['odds_10min']).copy()
    
    # Grid Parameters
    odds_caps = [30.0, 50.0, 100.0, 999.0]
    min_probs = [0.03, 0.05, 0.10, 0.15]
    ev_thresholds = [1.0, 1.2, 1.4, 1.6]
    
    results = []
    
    logger.info("Running Grid Search Simulation...")
    for cap, prob_min, ev_t in product(odds_caps, min_probs, ev_thresholds):
        valid_sim['ev'] = valid_sim['win_prob'] * valid_sim['odds_10min']
        
        mask = (valid_sim['odds_10min'] <= cap) & \
               (valid_sim['win_prob'] >= prob_min) & \
               (valid_sim['ev'] >= ev_t)
        
        bets = valid_sim[mask]
        
        if len(bets) >= 50: # Minimum bets to be meaningful
            total_bet = len(bets) * 100
            total_payout = (bets[bets['rank'] == 1]['odds'] * 100).sum()
            roi = (total_payout / total_bet) * 100
            hit_rate = (len(bets[bets['rank'] == 1]) / len(bets)) * 100
            
            results.append({
                'Odds_Cap': cap,
                'Min_Prob': prob_min,
                'EV_Threshold': ev_t,
                'ROI': roi,
                'HitRate': hit_rate,
                'BetCount': len(bets)
            })
            
    return pd.DataFrame(results)

def main():
    config_path = "config/experiments/exp_v12_win_lgbm.yaml"
    logger.info("Step 1: Get Predictions (2024-2025)")
    df = get_predictions_for_years(config_path)
    
    logger.info("Step 2: Calibration & Simulation")
    grid_results = run_grid_search(df)
    
    grid_results = grid_results.sort_values('ROI', ascending=False)
    
    print("\nv12 Win Specialist ROI Grid Search (2025 Test)")
    print("===============================================")
    print(grid_results.head(20).to_string(index=False))
    
    # Output to CSV
    grid_results.to_csv('reports/v12_win_specialist_grid_results.csv', index=False)
    logger.info("Results saved to reports/v12_win_specialist_grid_results.csv")
    
    # Check for Holy Grail
    holy_grail = grid_results[(grid_results['ROI'] > 100) & (grid_results['BetCount'] >= 100)]
    if not holy_grail.empty:
        print("\n🏆 Holy Grail / Sweet Spot FOUND!")
        print(holy_grail.to_string(index=False))
    else:
        print("\n❌ No segment with ROI > 100 and BetCount >= 100 found.")

if __name__ == "__main__":
    main()
