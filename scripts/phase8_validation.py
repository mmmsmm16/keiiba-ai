"""
Phase 8 Rigorous Validation Script
- Check population intersection between Final and Snapshot conditions
- Compare ledgers (tickets, bet, payout) for proof of identical purchases
"""
import pandas as pd
import numpy as np
import hashlib
import os

def load_preprocessed_2025_jra():
    """Load preprocessed data for 2025 JRA only"""
    df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
    # Filter 2025
    df['rid_str'] = df['race_id'].astype(str)
    df_2025 = df[df['rid_str'].str.match(r'^2025\d{8}$')].copy()
    print(f"Preprocessed 2025 JRA: {len(df_2025)} rows, {df_2025['race_id'].nunique()} races")
    return df_2025

def load_predictions():
    """Load v13 predictions"""
    pred_df = pd.read_parquet('data/predictions/v13_market_residual_2025_infer.parquet')
    print(f"Predictions: {len(pred_df)} rows, {pred_df['race_id'].nunique()} races")
    return pred_df

def load_snapshot_odds():
    """Load T-10m snapshot odds"""
    snap_df = pd.read_parquet('data/odds_snapshots/2025_win_T-10m_jra_only.parquet')
    print(f"Snapshot Odds: {len(snap_df)} rows, {snap_df['race_id'].nunique()} races")
    return snap_df

def analyze_population():
    """Analyze population differences"""
    print("=" * 60)
    print("POPULATION ANALYSIS")
    print("=" * 60)
    
    df = load_preprocessed_2025_jra()
    pred = load_predictions()
    snap = load_snapshot_odds()
    
    # Race ID sets
    preprocessed_races = set(df['race_id'].unique())
    pred_races = set(pred['race_id'].unique())
    snap_races = set(snap['race_id'].unique())
    
    print(f"\nPreprocessed JRA 2025 races: {len(preprocessed_races)}")
    print(f"Predictions races: {len(pred_races)}")
    print(f"Snapshot races: {len(snap_races)}")
    
    # Intersections
    pred_and_snap = pred_races.intersection(snap_races)
    all_three = preprocessed_races.intersection(pred_races).intersection(snap_races)
    
    print(f"\nPredictions ∩ Snapshot: {len(pred_and_snap)}")
    print(f"All three sources ∩: {len(all_three)}")
    
    # Differences
    only_in_pred = pred_races - snap_races
    only_in_snap = snap_races - pred_races
    
    print(f"\nOnly in Predictions (missing snapshot): {len(only_in_pred)}")
    print(f"Only in Snapshot (missing predictions): {len(only_in_snap)}")
    
    if only_in_pred:
        print(f"  Sample missing snapshot: {list(only_in_pred)[:5]}")
    if only_in_snap:
        print(f"  Sample missing predictions: {list(only_in_snap)[:5]}")
    
    return {
        'preprocessed_races': len(preprocessed_races),
        'pred_races': len(pred_races),
        'snap_races': len(snap_races),
        'intersection_all': len(all_three),
        'intersection_pred_snap': len(pred_and_snap),
        'only_in_pred': len(only_in_pred),
        'only_in_snap': len(only_in_snap)
    }

def check_v13_p_market_source():
    """
    Check if v13 predictions used final odds or snapshot odds for p_market
    """
    print("\n" + "=" * 60)
    print("V13 MODEL P_MARKET SOURCE CHECK")
    print("=" * 60)
    
    pred = load_predictions()
    
    # Check what columns exist in predictions
    print(f"\nPrediction columns: {pred.columns.tolist()}")
    
    # If 'odds' column exists in predictions, it's likely final odds
    if 'odds' in pred.columns:
        print("\n'odds' column exists in predictions.")
        print("This suggests p_market was calculated from odds at inference time.")
        print("Query: Was the inference done with 'final' or 'snapshot' odds?")
        
        # Sample odds values
        print(f"\nSample odds values: {pred['odds'].head(10).tolist()}")
    
    # Check for p_market column
    if 'p_market' in pred.columns:
        print("\n'p_market' column exists in predictions.")
        print(f"Sample p_market: {pred['p_market'].head(10).tolist()}")
    
    return pred.columns.tolist()

def main():
    pop_stats = analyze_population()
    pred_cols = check_v13_p_market_source()
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR PHASE 8 VALIDATION")
    print("=" * 60)
    print(f"Intersection (safe comparison set): {pop_stats['intersection_pred_snap']} races")
    print(f"Missing from snapshot: {pop_stats['only_in_pred']} races (cannot eval with snapshot)")
    print(f"Missing from predictions: {pop_stats['only_in_snap']} races (cannot eval at all)")
    
if __name__ == '__main__':
    main()
