import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import pickle
import logging
import argparse

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.dataset import DatasetSplitter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_simulation(config_path: str):
    # Load config to get feature blocks
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Features: {feature_blocks}")
    
    # Load Model
    model_path = f"models/experiments/{exp_name}/model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded.")

    # Load 2025 Data
    loader = JraVanDataLoader()
    # 2025 data
    raw_df = loader.load(history_start_date='2020-01-01', end_date='2025-12-31', jra_only=True)
    
    # Filter for 2025 only (simulate_year)
    # But FeaturePipeline needs history.
    # We will filter after feature eng.
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    df = pipeline.load_features(clean_df, feature_blocks) 
    
    # Splitter logic (just to get 'year' and sort)
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(clean_df['date']).dt.year
        
    test_df = df[df['year'] == 2025].copy()
    logger.info(f"2025 Test Data: {len(test_df)} records")
    
    # Prepare Features
    # Must match model features
    model_features = model.feature_name()
    X_test = test_df[model_features]
    
    # Predict (EV)
    logger.info("Predicting Expected Returns...")
    preds = model.predict(X_test)
    test_df['pred_return'] = preds
    
    # Debug: Prediction Stats
    p_mean = preds.mean()
    p_std = preds.std()
    p_min = preds.min()
    p_max = preds.max()
    logger.info(f"Prediction Stats: Mean={p_mean:.4f}, Std={p_std:.4f}, Min={p_min:.4f}, Max={p_max:.4f}")
    
    # Debug: Odds 10min Coverage
    if 'odds_10min' in test_df.columns:
        valid_odds_count = test_df['odds_10min'].notnull().sum()
        total_count = len(test_df)
        logger.info(f"Odds 10min coverage: {valid_odds_count}/{total_count} ({valid_odds_count/total_count:.1%})")
        
        # Check specific sample
        sample = test_df[['race_id', 'horse_number', 'odds_10min']].head(5)
        print("Sample Data:\n", sample)
    else:
        logger.warning("odds_10min column missing in Test DF!")

    # Merge Actual Results for Evaluation
    # We need Rank and Odds
    # clean_df has them. Merge on keys.
    target_cols = ['race_id', 'horse_number', 'rank', 'odds', 'horse_name', 'date']
    test_df = pd.merge(test_df, clean_df[target_cols], on=['race_id', 'horse_number'], how='left', suffixes=('', '_dup'))
    
    # Simulation Logic
    # Bet if pred_return > Threshold
    thresholds = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    
    report_lines = []
    report_lines.append("\n--- 2025 Return Regression Simulation ---\n")
    report_lines.append(f"| {'Threshold':<10} | {'Bets':<6} | {'Hits':<6} | {'Hit Rate':<8} | {'Return':<8} | {'Cost':<6} | {'ROI':<8} |")
    report_lines.append(f"|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*10}|{'-'*10}|{'-'*8}|{'-'*10}|")
    
    best_roi = 0.0
    best_th = 0.0
    
    for th in thresholds:
        bets = test_df[test_df['pred_return'] >= th]
        n_bets = len(bets)
        cost = n_bets * 100 # 100 yen per bet
        
        if n_bets == 0:
            line = f"| {th:<10.2f} | 0      | 0      | 0.0%     | 0        | 0      | 0.0%     |"
            report_lines.append(line)
            continue
            
        hits = bets[bets['rank'] == 1]
        n_hits = len(hits)
        
        # Return = Sum(Odds * 100)
        # Note: odds column in clean_df is float (e.g. 5.4).
        total_return = hits['odds'].sum() * 100
        
        roi = total_return / cost * 100
        hit_rate = n_hits / n_bets * 100
        
        if roi > best_roi and n_bets > 100: # Min 100 bets to confirm
            best_roi = roi
            best_th = th
            
        line = f"| {th:<10.2f} | {n_bets:<6} | {n_hits:<6} | {hit_rate:<7.1f}% | {int(total_return):<8} | {cost:<6} | {roi:<7.1f}% |"
        report_lines.append(line)

    # Deep Dive into Top Bets
    report_lines.append("\n--- Top 10 Predictions ---")
    top10 = test_df.sort_values('pred_return', ascending=False).head(10)
    for _, row in top10.iterrows():
        line = f"Date: {row['date']}, Race: {row['race_id']}, Horse: {row['horse_name']}, Pred: {row['pred_return']:.2f}, Odds: {row['odds']}, Rank: {row['rank']}"
        report_lines.append(line)
        
    # Feature Importance
    report_lines.append("\n--- Feature Importance ---")
    importance = model.feature_importance(importance_type='gain')
    names = model.feature_name()
    imp_df = pd.DataFrame({'feature': names, 'gain': importance}).sort_values('gain', ascending=False)
    for _, row in imp_df.head(20).iterrows():
        report_lines.append(f"{row['feature']}: {row['gain']:.2f}")

    # Save Results
    report_content = "\n".join(report_lines)
    print(report_content)
    
    with open(f"models/experiments/{exp_name}/simulation_report.txt", "w", encoding='utf-8') as f:
        f.write(report_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_simulation(args.config)
