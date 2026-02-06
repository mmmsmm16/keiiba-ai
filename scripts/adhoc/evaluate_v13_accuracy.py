"""
Evaluate V13 Model Accuracy
===========================
Calculates Win Rate, Top-3 Rate, and ROI for V13 model Rank 1 predictions.
Compares results with Popularity 1 (Basline).
"""
import pandas as pd
import numpy as np
import os

V13_OOF_2024 = 'data/predictions/v13_oof_2024_clean.parquet'
V13_OOF_2025 = 'data/predictions/v13_oof_2025_clean.parquet'
GROUND_TRUTH = 'data/processed/preprocessed_data_v13_active.parquet'

def evaluate_oof(path, label, gt_df):
    if not os.path.exists(path):
        print(f"Skipping {label}: File not found.")
        return
    
    oof = pd.read_parquet(path)
    oof['race_id'] = oof['race_id'].astype(str)
    
    # Merge with Ground Truth to get real odds and ranks
    df = pd.merge(oof[['race_id', 'horse_number', 'pred_prob']], 
                  gt_df[['race_id', 'horse_number', 'odds', 'popularity', 'rank']], 
                  on=['race_id', 'horse_number'])
    
    # Clean rank
    df['actual_rank'] = pd.to_numeric(df['rank'], errors='coerce').fillna(99).astype(int)
    df['is_win'] = (df['actual_rank'] == 1).astype(int)
    
    # Model Rank
    df['v13_rank'] = df.groupby('race_id')['pred_prob'].rank(ascending=False, method='first')
    
    top1 = df[df['v13_rank'] == 1].copy()
    n_races = top1['race_id'].nunique()
    
    # Win Rate
    win_rate = top1['is_win'].mean()
    # Top 3
    top3_rate = (top1['actual_rank'] <= 3).mean()
    
    # ROI
    invest = n_races * 100
    ret_win = (top1[top1['is_win'] == 1]['odds'].sum()) * 100
    roi_win = ret_win / invest
    
    print(f"\n=== V13 Evaluation: {label} ===")
    print(f"Races:     {n_races}")
    print(f"Win Rate:  {win_rate:>.1%}")
    print(f"Top-3 Rate: {top3_rate:>.1%}")
    print(f"Win ROI:   {roi_win:>.1%}")
    print(f"Avg Odds:  {top1['odds'].mean():.2f}")

    # Baseline: Popularity 1
    # Note: Use Ground Truth to find pop1
    gt_races = gt_df[gt_df['race_id'].isin(df['race_id'].unique())].copy()
    gt_races['actual_rank'] = pd.to_numeric(gt_races['rank'], errors='coerce').fillna(99).astype(int)
    pop1 = gt_races[gt_races['popularity'] == 1]
    
    pop1_win_rate = (pop1['actual_rank'] == 1).mean()
    pop1_top3_rate = (pop1['actual_rank'] <= 3).mean()
    pop1_roi = (pop1[pop1['actual_rank'] == 1]['odds'].sum() * 100) / (len(pop1) * 100)
    
    print(f"\n--- Baseline: Popularity 1 ---")
    print(f"Win Rate:  {pop1_win_rate:>.1%}")
    print(f"Top-3 Rate: {pop1_top3_rate:>.1%}")
    print(f"Win ROI:   {pop1_roi:>.1%}")
    print(f"Avg Odds:  {pop1['odds'].mean():.2f}")

if __name__ == "__main__":
    print("Loading Ground Truth...")
    gt = pd.read_parquet(GROUND_TRUTH)
    gt['race_id'] = gt['race_id'].astype(str)
    
    evaluate_oof(V13_OOF_2024, "2024 (OOF)", gt)
    evaluate_oof(V13_OOF_2025, "2025 (OOF)", gt)
