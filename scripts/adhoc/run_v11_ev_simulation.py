import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
    # Load data for requested years
    start_year = min(years)
    df_raw = loader.load(history_start_date=f"{start_year}-01-01", end_date="2025-12-31", jra_only=True)
    
    cleanser = DataCleanser()
    df_clean = cleanser.cleanse(df_raw)
    
    # Filter years
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_target = df_clean[df_clean['date'].dt.year.isin(years)].copy()
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    feature_blocks = config.get('features', [])
    df_features = pipeline.load_features(df_target, feature_blocks)
    
    # Ensure year and key-based features
    if 'year' not in df_features.columns:
        df_features = pd.merge(df_features, df_target[['race_id', 'horse_number', 'date']], on=['race_id', 'horse_number'], how='left')
        df_features['year'] = pd.to_datetime(df_features['date']).dt.year

    # Load models
    lgbm_model, cat_model = load_v11_models()
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
    
    # LGBM predict
    X_lgbm = X.copy()
    for col in lgbm_cat_cols:
        if col in X_lgbm.columns:
            X_lgbm[col] = X_lgbm[col].astype('category')
    lgbm_preds = lgbm_model.predict(X_lgbm)
    
    # CatBoost predict
    X_cat = X.copy()
    for col in cat_cat_cols:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str).replace('nan', '')
    cat_preds = cat_model.predict(X_cat)
    
    # Results DF
    res_df = df_target[['race_id', 'horse_number', 'date', 'rank', 'odds', 'odds_10min']].copy()
    res_df['year'] = res_df['date'].dt.year
    res_df['lgbm_score'] = lgbm_preds
    res_df['cat_score'] = cat_preds
    
    # Ensemble Score: Simple Mean of Raw Scores (since they are both LambdaRank centered around 0)
    # Alternatively normalized scores, but let's try mean first.
    res_df['ensemble_score'] = (res_df['lgbm_score'] + res_df['cat_score']) / 2
    
    return res_df

def calibrate_and_simulate(df):
    # 1. Calibration (Train on 2024)
    train_df = df[df['year'] == 2024].copy()
    test_df = df[df['year'] == 2025].copy()
    
    logger.info(f"Calibrating with {len(train_df)} records from 2024...")
    ir = IsotonicRegression(out_of_bounds='clip')
    
    # y: 1 if Win else 0
    y_train = (train_df['rank'] == 1).astype(int)
    ir.fit(train_df['ensemble_score'], y_train)
    
    # Apply to 2025
    test_df['win_prob_raw'] = ir.predict(test_df['ensemble_score'])
    
    # 2. Normalization (Race-wise)
    logger.info("Normalizing probabilities...")
    test_df['win_prob'] = test_df.groupby('race_id')['win_prob_raw'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    # 3. EV Calculation (using 10min odds)
    # If odds_10min is NaN, skip or use a proxy? User specified Odds_10min.
    # Let's drop records with missing Odds_10min for simulation accuracy.
    valid_sim = test_df.dropna(subset=['odds_10min']).copy()
    valid_sim['ev_10min'] = valid_sim['win_prob'] * valid_sim['odds_10min']
    
    # 4. Simulation
    simulation_results = []
    thresholds = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    logger.info("Running ROI Simulation for different EV thresholds...")
    for t in thresholds:
        bets = valid_sim[valid_sim['ev_10min'] > t]
        if len(bets) == 0: continue
        
        # calculate ROI (Final Odds for return)
        total_bet = len(bets) * 100
        total_payout = (bets[bets['rank'] == 1]['odds'] * 100).sum()
        roi = (total_payout / total_bet) * 100
        hit_rate = (len(bets[bets['rank'] == 1]) / len(bets)) * 100
        
        # Gap Impact: Odds_10min vs Odds_Final
        # Positive gap means odds decreased (Bad for us)
        # bets['odds_drop'] = bets['odds_10min'] - bets['odds']
        # avg_drop = bets['odds_drop'].mean()
        
        simulation_results.append({
            'EV_Threshold': t,
            'ROI': roi,
            'HitRate': hit_rate,
            'BetCount': len(bets),
            'AvgProb': bets['win_prob'].mean(),
            'AvgOdds10m': bets['odds_10min'].mean(),
            'AvgOddsFinal': bets['odds'].mean()
        })
        
    return pd.DataFrame(simulation_results), valid_sim

def main():
    config_path = "config/experiments/exp_v11_lgbm.yaml"
    logger.info("Fetching data and predictions (2024-2025)...")
    df = get_predictions_for_years(config_path, years=[2024, 2025])
    
    logger.info("Analyzing Simulation...")
    report_df, sim_df = calibrate_and_simulate(df)
    
    print("\nv11 EV Simulation Report (Real-World Setting)")
    print("==============================================")
    print(report_df.to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='EV_Threshold', y='ROI', data=report_df, marker='o', label='ROI (%)')
    plt.axhline(100, color='red', linestyle='--', label='Break Even')
    plt.title('EV Threshold vs ROI (Calibration: 2024, Test: 2025)')
    plt.xlabel('EV Threshold (calculated with 10min odds)')
    plt.ylabel('ROI (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('reports/v11_ev_simulation_roi.png')
    
    report_df.to_csv('reports/v11_ev_simulation_results.csv', index=False)
    logger.info("Results saved to reports/v11_ev_simulation_results.csv and reports/v11_ev_simulation_roi.png")

if __name__ == "__main__":
    main()
