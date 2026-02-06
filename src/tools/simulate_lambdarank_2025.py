import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from scipy.special import softmax

# プロジェクトルートへのパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    default_config = "config/experiments/exp_v06_lambdarank.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, help="Path to config yaml")
    parser.add_argument("--year", type=int, default=2025, help="Year to simulate")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    
    logger.info(f"🚀 LambdaRank Simulation {args.year} for {exp_name}")

    # 1. Load Data
    loader = JraVanDataLoader()
    start_date = f"{args.year}-01-01"
    end_date = f"{args.year}-12-31"
    load_start = f"{args.year-1}-01-01" 
    
    logger.info(f"Loading data ({load_start} ~ {end_date})...")
    raw_df = loader.load(history_start_date=load_start, end_date=end_date, jra_only=True)
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Features
    pipeline = FeaturePipeline(cache_dir="data/features")
    df_features = pipeline.load_features(clean_df, feature_blocks)
    
    # Merge Meta
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds', 'horse_name']
    df_sim = pd.merge(
        df_features, 
        clean_df[meta_cols], 
        on=['race_id', 'horse_number'], 
        how='inner'
    )
    
    # Filter Year
    df_sim['date'] = pd.to_datetime(df_sim['date'])
    df_target = df_sim[(df_sim['date'] >= start_date) & (df_sim['date'] <= end_date)].copy()
    
    # 3. Predict & Softmax
    model_path = f"models/experiments/{exp_name}/model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Feature Alignment
    model_features = model.feature_name()
    if 'age' in df_target.columns:
        df_target['age'] = pd.to_numeric(df_target['age'], errors='coerce')

    X = pd.DataFrame(index=df_target.index)
    for feat in model_features:
        if feat in df_target.columns:
            X[feat] = df_target[feat]
        else:
            X[feat] = 0.0

    logger.info("Predicting raw scores...")
    # LambdaRank output is raw score (unbounded)
    scores = model.predict(X)
    df_target['score'] = scores
    
    # Calculate Softmax per Race
    logger.info("Calculating Softmax Probabilities...")
    
    def calc_softmax(group):
        group['pred_prob'] = softmax(group['score'])
        return group

    df_target = df_target.groupby('race_id', group_keys=False).apply(calc_softmax)
    
    # 4. EV Simulation
    logger.info("Simulating EV Strategy...")
    # EV = P(Win) * Odds
    df_target['ev'] = df_target['pred_prob'] * df_target['odds']
    
    stats_list = []
    thresholds = np.arange(0.5, 3.05, 0.1) # Wider range for LambdaRank
    
    for th in thresholds:
        bets = df_target[
            (df_target['ev'] >= th) & 
            (df_target['rank'] > 0) & 
            (df_target['odds'].notna())
        ].copy()
        
        count = len(bets)
        if count == 0:
            stats_list.append({'th': th, 'bets': 0, 'roi': 0, 'hit': 0})
            continue
            
        hits = bets[bets['rank'] == 1]
        return_amount = hits['odds'].sum() * 100
        cost_amount = count * 100
        roi = (return_amount / cost_amount) * 100
        hit_rate = (len(hits) / count) * 100
        
        stats_list.append({
            'threshold': th,
            'bet_count': count,
            'hit_rate': hit_rate,
            'roi': roi,
            'profit': return_amount - cost_amount
        })
        
    df_stats = pd.DataFrame(stats_list)
    print("\nXXX LambdaRank Strategy Result (Softmax EV) XXX")
    print(df_stats)
    
    # Save
    out_csv = f"reports/strategy_lambdarank_{args.year}.csv"
    df_stats.to_csv(out_csv, index=False)
    
    # Save detailed log with probs
    log_csv = f"reports/simulation_{exp_name}_{args.year}.csv"
    df_target[['race_id', 'horse_name', 'rank', 'odds', 'score', 'pred_prob', 'ev']].to_csv(log_csv, index=False)
    logger.info(f"Saved stats to {out_csv} and log to {log_csv}")

if __name__ == "__main__":
    main()
