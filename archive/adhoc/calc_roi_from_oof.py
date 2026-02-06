
import pandas as pd
import numpy as np
import os
import sys

def main():
    # Load calibrated OOF data (most complete)
    path = 'data/predictions/v13_wf_2025_full_retrained_oof_calibrated.parquet'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows.")
    
    # Check columns
    print("Columns:", df.columns.tolist())
    
    # Needs: race_id, odds, rank, popularity, score (or prob)
    # If popularity is missing, we might need to join with raw data, but usually OOF keeps IDs.
    # Let's check if popularity is there.
    
    # Assuming 'score' is the model score.
    # Assuming 'odds' and 'rank' are present.
    
    stats = {
        'model_top1': {'bet': 0, 'return': 0, 'hits': 0, 'races': 0},
        'pop_top1': {'bet': 0, 'return': 0, 'hits': 0, 'races': 0},
        'random': {'bet': 0, 'return': 0, 'hits': 0, 'races': 0},
    }
    
    # Filter valid races (rank available)
    # df = df[df['rank'].notnull()] # rank might be string or float
    
    # Load popularity separately
    pop_path = 'data/processed/preprocessed_data.parquet'
    if os.path.exists(pop_path):
        print(f"Loading popularity from {pop_path}...")
        # Only load necessary columns
        pop_df = pd.read_parquet(pop_path, columns=['race_id', 'horse_number', 'popularity', 'odds'])
        
        # Ensure types for merge (Handle NaNs)
        df['race_id'] = df['race_id'].astype(str)
        df = df.dropna(subset=['horse_number'])
        df['horse_number'] = df['horse_number'].astype(float).astype(int)
        
        pop_df['race_id'] = pop_df['race_id'].astype(str)
        pop_df = pop_df.dropna(subset=['horse_number'])
        pop_df['horse_number'] = pop_df['horse_number'].astype(float).astype(int)
        
    # Merge
    df = pd.merge(df, pop_df, on=['race_id', 'horse_number'], how='left')
    
    # Filter Valid Odds
    initial_races = df['race_id'].nunique()
    df = df.dropna(subset=['odds', 'rank'])
    valid_races_count = df['race_id'].nunique()
    
    print(f"Total Races in OOF: {initial_races}")
    print(f"Valid Results (Odds & Rank exists): {valid_races_count}")
    
    # Group by race
    for rid, grp in df.groupby('race_id'):
        # Check if rank/odds exist for all
        # Skip if incomplete data? No, try best effort.
        print("Popularity merged.")
    else:
        print("Preprocessed data not found for popularity.")

    # Group by race
    for rid, grp in df.groupby('race_id'):
        # Check if rank/odds exist for all
        # Skip if incomplete data? No, try best effort.
        
        # --- Model Top 1 ---
        # Pick best score via pred_logit
        score_col = 'pred_logit' if 'pred_logit' in grp.columns else 'pred_prob'
        if score_col in grp.columns:
            top1 = grp.loc[grp[score_col].idxmax()]
            stats['model_top1']['bet'] += 100
            stats['model_top1']['races'] += 1
            if top1['rank'] == 1:
                stats['model_top1']['hits'] += 1
                stats['model_top1']['return'] += int(top1['odds'] * 100) if not pd.isna(top1['odds']) else 100
        
        # --- Popularity Top 1 ---
        if 'popularity' in grp.columns:
            # Drop NaNs for idxmin
            valid_pop = grp.dropna(subset=['popularity'])
            if not valid_pop.empty:
                pop1 = valid_pop.loc[valid_pop['popularity'].idxmin()]
                stats['pop_top1']['bet'] += 100
                stats['pop_top1']['races'] += 1
                if pop1['rank'] == 1:
                    stats['pop_top1']['hits'] += 1
                    stats['pop_top1']['return'] += int(pop1['odds'] * 100) if not pd.isna(pop1['odds']) else 100
        
        # --- Random (Simulate 100 times?) No, just Expected Value: Sum(1/odds)? No.
        # Just pick random 1 horse?
        # Let's skip random for now, comparison is Model vs Pop.

    # Report
    print("\n" + "="*50)
    print(f" STRICT ROI CHECK REPORT (From OOF: {path})")
    print("="*50)

    for key, val in stats.items():
        if val['races'] == 0: continue
        roi = val['return'] / val['bet'] * 100 if val['bet'] > 0 else 0
        hit_rate = val['hits'] / val['races'] * 100
        print(f"[{key}]")
        print(f"  Races: {val['races']}")
        print(f"  Hits:  {val['hits']} ({hit_rate:.1f}%)")
        print(f"  Bet:   {val['bet']:,} yen")
        print(f"  Ret:   {val['return']:,} yen")
        print(f"  ROI:   {roi:.2f}%")
        print("-" * 30)

if __name__ == '__main__':
    main()
