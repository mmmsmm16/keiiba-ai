import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze(input_path: str):
    df = pd.read_parquet(input_path)
    # 1位のみ抽出
    top1 = df[df['rank_pred'] == 1].copy()
    
    # 1. Entropy binning
    logger.info("Analyzing Entropy vs ROI...")
    top1['entropy_bin'] = pd.qcut(top1['entropy'], 10)
    ent_stats = top1.groupby('entropy_bin').agg({
        'payout_win': lambda x: x.sum() / (len(x) * 100),
        'payout_place': lambda x: x.sum() / (len(x) * 100),
        'race_id': 'count'
    }).rename(columns={'payout_win': 'roi_win', 'payout_place': 'roi_place', 'race_id': 'count'})
    print("\n--- Entropy Bins vs ROI ---")
    print(ent_stats)
    
    # 2. Field Size binning
    logger.info("Analyzing Field Size vs ROI...")
    size_stats = top1.groupby('field_size').agg({
        'payout_win': lambda x: x.sum() / (len(x) * 100),
        'payout_place': lambda x: x.sum() / (len(x) * 100),
        'race_id': 'count'
    }).rename(columns={'payout_win': 'roi_win', 'payout_place': 'roi_place', 'race_id': 'count'})
    print("\n--- Field Size vs ROI ---")
    print(size_stats)
    
    # 3. p1 binning
    logger.info("Analyzing p1 vs ROI...")
    top1['p1_bin'] = pd.qcut(top1['p1'], 10)
    p1_stats = top1.groupby('p1_bin').agg({
        'payout_win': lambda x: x.sum() / (len(x) * 100),
        'payout_place': lambda x: x.sum() / (len(x) * 100),
        'race_id': 'count'
    }).rename(columns={'payout_win': 'roi_win', 'payout_place': 'roi_place', 'race_id': 'count'})
    print("\n--- p1 Bins vs ROI ---")
    print(p1_stats)

if __name__ == "__main__":
    analyze("reports/simulations/v13_e1_enriched_2022_2024.parquet")
