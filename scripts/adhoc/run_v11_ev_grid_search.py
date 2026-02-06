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

def load_v11_models():
    lgbm_path = "models/experiments/v11_lgbm_enhanced/model.pkl"
    cat_path = "models/experiments/v11_cat_enhanced/model.cbm"
    
    logger.info(f"Loading LGBM model from {lgbm_path}...")
    with open(lgbm_path, 'rb') as f:
        lgbm_model = pickle.load(f)
    
    logger.info(f"Loading CatBoost model from {cat_path}...")
    from catboost import CatBoostRanker
    cat_model = CatBoostRanker()
    cat_model.load_model(cat_path)
        
    return lgbm_model, cat_model

def get_predictions_for_years(config_path, years=[2024, 2025]):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    dataset_cfg = config.get('dataset', {})
    
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
    
    if 'year' not in df_features.columns:
        df_features = pd.merge(df_features, df_target[['race_id', 'horse_number', 'date']], on=['race_id', 'horse_number'], how='left')
        df_features['year'] = pd.to_datetime(df_features['date']).dt.year

    lgbm_model, cat_model = load_v11_models()
    feature_names = lgbm_model.feature_name()
    X = df_features[feature_names].copy()
    
    lgbm_cat_cols = [
        "jockey_id", "trainer_id", "sire_id", "horse_id", "sex", 
        "grade_code", "kyoso_joken_code", "surface", "venue", 
        "prev_grade", "dist_change_category", "interval_category"
    ]
    auto_cat = [c for c in X.columns if X[c].dtype == 'object']
    lgbm_cat_cols = list(set(lgbm_cat_cols + auto_cat))
    
    cat_cat_cols = list(set(lgbm_cat_cols + ["age"]))
    
    # LGBM predict
    X_lgbm = X.copy()
    for col in lgbm_cat_cols:
        if col in X_lgbm.columns:
            X_lgbm[col] = X_lgbm[col].astype('category')
    lgbm_preds = lgbm_model.predict(X_lgbm)
    
    X_cat = X.copy()
    for col in cat_cat_cols:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str).replace('nan', '')
    cat_preds = cat_model.predict(X_cat)
    
    res_df = df_target[['race_id', 'horse_number', 'date', 'rank', 'odds', 'odds_10min']].copy()
    res_df['year'] = res_df['date'].dt.year
    res_df['ensemble_score'] = (lgbm_preds + cat_preds) / 2
    
    return res_df

def run_grid_search(df):
    train_df = df[df['year'] == 2024].copy()
    test_df = df[df['year'] == 2025].copy()
    
    logger.info("Calibrating win probabilities (2024)...")
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(train_df['ensemble_score'], (train_df['rank'] == 1).astype(int))
    
    test_df['win_prob_raw'] = ir.predict(test_df['ensemble_score'])
    test_df['win_prob'] = test_df.groupby('race_id')['win_prob_raw'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    valid_sim = test_df.dropna(subset=['odds_10min']).copy()
    
    # Grid Parameters
    odds_caps = [30.0, 50.0, 999.0]
    min_probs = [0.03, 0.05, 0.10]
    ev_thresholds = [0.8, 1.0, 1.2]
    
    results = []
    
    logger.info("Starting Grid Search...")
    for cap, prob_min, ev_t in product(odds_caps, min_probs, ev_thresholds):
        # Calculate EV first
        valid_sim['ev_current'] = valid_sim['win_prob'] * valid_sim['odds_10min']
        
        # Apply Filters
        mask = (valid_sim['odds_10min'] <= cap) & \
               (valid_sim['win_prob'] >= prob_min) & \
               (valid_sim['ev_current'] >= ev_t)
        
        bets = valid_sim[mask]
        
        if len(bets) > 0:
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
        else:
            results.append({
                'Odds_Cap': cap,
                'Min_Prob': prob_min,
                'EV_Threshold': ev_t,
                'ROI': 0,
                'HitRate': 0,
                'BetCount': 0
            })
            
    return pd.DataFrame(results)

def main():
    config_path = "config/experiments/exp_v11_lgbm.yaml"
    logger.info("Fetching data and predictions (2024-2025)...")
    df = get_predictions_for_years(config_path, years=[2024, 2025])
    
    logger.info("Running Multi-Filter Grid Search...")
    report_df = run_grid_search(df)
    
    # Sort by ROI descending
    report_df = report_df.sort_values('ROI', ascending=False)
    
    print("\nv11 EV Grid Search Report (Phase 8.5)")
    print("======================================")
    print(report_df.to_string(index=False))
    
    # Find Sweet Spot
    sweet_spot = report_df[(report_df['ROI'] > 100) & (report_df['BetCount'] >= 100)]
    if not sweet_spot.empty:
        print("\n🏆 Sweet Spot Found!")
        print(sweet_spot.to_string(index=False))
    else:
        print("\n❌ No Sweet Spot (ROI > 100 & Bet > 100) found with current filters.")
    
    report_df.to_csv('reports/v11_ev_grid_search_results.csv', index=False)
    logger.info("Results saved to reports/v11_ev_grid_search_results.csv")

if __name__ == "__main__":
    main()
