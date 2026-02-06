import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
import argparse
import yaml

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_optimization(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    
    # Load Model
    model_path = f"models/experiments/{exp_name}/model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load 2025 Data (2020- is too large for memory if we repeat, but let's use it)
    loader = JraVanDataLoader()
    raw_df = loader.load(history_start_date='2020-01-01', end_date='2025-12-31', jra_only=True)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    df = pipeline.load_features(clean_df, feature_blocks) 
    
    # 2025 Filter
    df['year'] = pd.to_datetime(clean_df['date']).dt.year
    test_df = df[df['year'] == 2025].copy()
    
    # Predict
    model_features = model.feature_name()
    test_df['pred_return'] = model.predict(test_df[model_features])
    
    # Target Data Merge
    # We need Rank, Odds, and metadata for Segment analysis (Surface, Distance, Date)
    target_cols = ['race_id', 'horse_number', 'rank', 'odds', 'date', 'surface', 'distance', 'venue']
    test_df = pd.merge(test_df, clean_df[target_cols], on=['race_id', 'horse_number'], how='left')
    
    # Race-level Stats
    race_stats = test_df.groupby('race_id').agg({
        'pred_return': ['max', 'count'],
        'date': 'first'
    })
    race_stats.columns = ['max_pred', 'n_horses', 'date']
    
    # Simulation Parameters
    ken_thresholds = np.linspace(0.4, 1.6, 25) # Up to 1.6
    top_ks = [1, 2, 3]
    odds_filters = [0, 5, 10] # Only bet if odds_10min >= filter
    
    results = []
    
    for k in top_ks:
        # Rank within race
        test_df['pred_rank'] = test_df.groupby('race_id')['pred_return'].rank(ascending=False, method='min')
        
        for odds_f in odds_filters:
            for th in ken_thresholds:
                # Selected Races: Max Pred >= th
                selected_races = race_stats[race_stats['max_pred'] >= th].index
                
                # Selected Bets: In Selected Races AND within Top K AND matching Odds Filter
                mask = (test_df['race_id'].isin(selected_races)) & (test_df['pred_rank'] <= k)
                if odds_f > 0:
                    mask &= (test_df['odds_10min'] >= odds_f)
                
                bets = test_df[mask]
            
            n_bets = len(bets)
            if n_bets == 0: continue
            
            hits = bets[bets['rank'] == 1]
            n_hits = len(hits)
            
            total_cost = n_bets * 100
            total_return = (hits['odds'] * 100).sum()
            roi = total_return / total_cost * 100
            
            # Bet Rate (Based on total races in 2025)
            total_races = len(race_stats)
            selected_race_count = len(selected_races)
            bet_race_rate = selected_race_count / total_races * 100
            
            results.append({
                'Top-K': k,
                'KenTh': th,
                'OddsF': odds_f,
                'Bets': n_bets,
                'ROI': roi,
                'BetRaceRate': bet_race_rate,
                'HitRate': n_hits / n_bets * 100
            })

    # Optional: Proportional Betting (on Top 1)
    k = 1
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_return'].rank(ascending=False, method='min')
    for th in ken_thresholds:
        selected_races = race_stats[race_stats['max_pred'] >= th].index
        bets = test_df[(test_df['race_id'].isin(selected_races)) & (test_df['pred_rank'] <= k)].copy()
        if bets.empty: continue
        
        # BetAmount proportional to pred_return (but >= 100)
        bets['amount'] = (bets['pred_return'] * 100).round(-1) # Round to 10
        total_cost = bets['amount'].sum()
        
        hits = bets[bets['rank'] == 1]
        total_return = (hits['odds'] * hits['amount']).sum()
        roi = total_return / total_cost * 100
        
        results.append({
            'Top-K': '1-Prop',
            'KenTh': th,
            'Bets': len(bets),
            'ROI': roi,
            'BetRaceRate': len(selected_races) / len(race_stats) * 100,
            'HitRate': len(hits) / len(bets) * 100
        })

    # --- Segment Analysis ---
    print("\n--- Segment Analysis (Top-K=2, KenTh=1.0) ---")
    res_segments = []
    
    # Add surface and distance (bucketed)
    test_df['dist_bucket'] = pd.cut(test_df['distance'], bins=[0, 1400, 1800, 2200, 10000], labels=['Sprint', 'Mile', 'Middle', 'Long'])
    
    k = 2
    th = 1.0
    selected_races = race_stats[race_stats['max_pred'] >= th].index
    
    segments = {
        'surface': test_df['surface'].unique(),
        'dist_bucket': test_df['dist_bucket'].unique()
    }
    
    for seg_col, vals in segments.items():
        for val in vals:
            mask = (test_df['race_id'].isin(selected_races)) & (test_df['pred_rank'] <= k) & (test_df[seg_col] == val)
            bets = test_df[mask]
            if len(bets) < 50: continue
            
            hits = bets[bets['rank'] == 1]
            roi = (hits['odds'] * 100).sum() / (len(bets) * 100) * 100
            res_segments.append({
                'Segment': f"{seg_col}={val}",
                'Bets': len(bets),
                'ROI': roi,
                'HitRate': len(hits) / len(bets) * 100
            })
            
    # --- Undervalued Underdogs ---
    print("\n--- Underdog Analysis (Top-K=2, KenTh=1.0) ---")
    for min_odds in [10, 20, 30]:
        mask = (test_df['race_id'].isin(selected_races)) & (test_df['pred_rank'] <= 2) & (test_df['odds_10min'] >= min_odds)
        bets = test_df[mask]
        if len(bets) < 30: continue
        hits = bets[bets['rank'] == 1]
        roi = (hits['odds'] * 100).sum() / (len(bets) * 100) * 100
        res_segments.append({
            'Segment': f"Underdog_Odds>={min_odds}",
            'Bets': len(bets),
            'ROI': roi,
            'HitRate': len(hits) / len(bets) * 100
        })

    seg_df = pd.DataFrame(res_segments)
    print(seg_df.sort_values('ROI', ascending=False).to_markdown())

    res_df = pd.DataFrame(results)
    
    # Find Holy Grail
    holy_grail = res_df[(res_df['BetRaceRate'] >= 10.0) & (res_df['ROI'] > 100.0)].sort_values('ROI', ascending=False)
    
    print("\n--- Strategy Optimization Results (ROI) ---")
    # Multi-dimensional pivot using pivot_table
    roi_pivot = pd.pivot_table(res_df, values='ROI', index=['KenTh', 'OddsF'], columns='Top-K')
    print(roi_pivot)
    
    print("\n--- Bet Race Rate (%) ---")
    rate_pivot = pd.pivot_table(res_df, values='BetRaceRate', index=['KenTh', 'OddsF'], columns='Top-K')
    print(rate_pivot)

    if not holy_grail.empty:
        print("\n🏆 HOLY GRAIL FOUND!")
        print(holy_grail.head(5))
    else:
        print("\nNo condition met ROI > 100% and Bet Rate > 10%.")
        best = res_df[res_df['BetRaceRate'] >= 5.0].sort_values('ROI', ascending=False).head(5)
        print("Best alternatives (Rate > 5%):")
        print(best)

    # Save to report
    report_path = f"models/experiments/{exp_name}/optimization_strategy.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Strategy Optimization Report\n\n")
        f.write(res_df.to_markdown())
    
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run_optimization(parser.parse_args().config)
