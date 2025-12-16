"""
Phase 8: Recalculate prob_residual_softmax with Snapshot p_market

v13 market_residual model uses:
  score_logit = logit(p_market) + delta_logit
  prob_residual_softmax = softmax(score_logit) within race

If we replace p_market with snapshot-based p_market:
  score_logit_snap = logit(p_market_snap) + delta_logit
  prob_residual_softmax_snap = softmax(score_logit_snap)

This shows the "true" prediction if we had used pre-race odds.
"""
import pandas as pd
import numpy as np
from scipy.special import expit, logit as scipy_logit
import os

def load_data():
    pred = pd.read_parquet('data/predictions/v13_market_residual_2025_infer.parquet')
    snap = pd.read_parquet('data/odds_snapshots/2025_win_T-10m_jra_only.parquet')
    return pred, snap

def calculate_p_market_from_odds(odds_series):
    """Convert odds to normalized market probability"""
    raw_prob = 1.0 / odds_series.replace(0, np.nan)
    return raw_prob

def softmax_per_race(df, score_col, out_col):
    """Apply softmax normalization within each race"""
    def softmax(x):
        exp_x = np.exp(x - x.max())  # Numerical stability
        return exp_x / exp_x.sum()
    
    df[out_col] = df.groupby('race_id')[score_col].transform(softmax)
    return df

def main():
    print("Loading data...")
    pred, snap = load_data()
    
    # Merge snapshot odds with predictions
    # Need to align on race_id and horse_number/horse_id
    # Check what keys are available
    print(f"Predictions columns: {pred.columns.tolist()}")
    print(f"Snapshot columns: {snap.columns.tolist()}")
    
    # Predictions uses horse_id (ketto_toroku_bango), snapshot uses horse_number (umaban)
    # We need to join through preprocessed data or use a different key
    # Actually, let's check if predictions have horse_number
    
    # Load preprocessed to get horse_id -> horse_number mapping
    df_full = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
    df_2025 = df_full[df_full['race_id'].astype(str).str.startswith('2025')].copy()
    
    # Create mapping: race_id + horse_id -> horse_number
    id_map = df_2025[['race_id', 'horse_id', 'horse_number']].drop_duplicates()
    print(f"ID mapping rows: {len(id_map)}")
    
    # Merge snapshot with mapping
    snap_with_id = snap.merge(id_map, on=['race_id', 'horse_number'], how='left')
    print(f"Snapshot with horse_id: {snap_with_id['horse_id'].notna().sum()} / {len(snap_with_id)}")
    
    # Now merge with predictions
    pred_merged = pred.merge(
        snap_with_id[['race_id', 'horse_id', 'odds_snapshot']],
        on=['race_id', 'horse_id'],
        how='left'
    )
    
    coverage = pred_merged['odds_snapshot'].notna().sum() / len(pred_merged)
    print(f"Snapshot odds coverage in predictions: {coverage:.1%}")
    
    # Filter to rows with snapshot odds
    pred_snap = pred_merged.dropna(subset=['odds_snapshot']).copy()
    print(f"Rows with snapshot odds: {len(pred_snap)}")
    
    # Calculate p_market from snapshot odds
    pred_snap['p_market_snap_raw'] = calculate_p_market_from_odds(pred_snap['odds_snapshot'])
    
    # Normalize within race
    pred_snap['p_market_snap'] = pred_snap.groupby('race_id')['p_market_snap_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.nan
    )
    
    # Recalculate score_logit with snapshot p_market
    # Original: score_logit = logit(p_market) + delta_logit
    # New: score_logit_snap = logit(p_market_snap) + delta_logit
    
    # Clip to avoid logit of 0 or 1
    eps = 1e-6
    pred_snap['p_market_snap_clipped'] = pred_snap['p_market_snap'].clip(eps, 1 - eps)
    pred_snap['score_logit_snap'] = scipy_logit(pred_snap['p_market_snap_clipped']) + pred_snap['delta_logit']
    
    # Apply softmax per race
    pred_snap = softmax_per_race(pred_snap, 'score_logit_snap', 'prob_residual_softmax_snap')
    
    # Compare original vs snapshot-based predictions
    pred_snap['prob_diff'] = pred_snap['prob_residual_softmax_snap'] - pred_snap['prob_residual_softmax']
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPARISON: Final vs Snapshot p_market")
    print("=" * 60)
    
    print(f"\nMean absolute diff: {pred_snap['prob_diff'].abs().mean():.6f}")
    print(f"Max absolute diff: {pred_snap['prob_diff'].abs().max():.6f}")
    print(f"Std of diff: {pred_snap['prob_diff'].std():.6f}")
    
    # Check rank changes
    pred_snap['rank_final'] = pred_snap.groupby('race_id')['prob_residual_softmax'].rank(ascending=False)
    pred_snap['rank_snap'] = pred_snap.groupby('race_id')['prob_residual_softmax_snap'].rank(ascending=False)
    pred_snap['rank_changed'] = pred_snap['rank_final'] != pred_snap['rank_snap']
    
    rank_change_rate = pred_snap['rank_changed'].mean()
    print(f"\nRank change rate: {rank_change_rate:.1%}")
    
    # Top-N selection changes
    for n in [1, 3, 5]:
        final_topn = set(pred_snap[pred_snap['rank_final'] <= n][['race_id', 'horse_id']].apply(tuple, axis=1))
        snap_topn = set(pred_snap[pred_snap['rank_snap'] <= n][['race_id', 'horse_id']].apply(tuple, axis=1))
        overlap = len(final_topn & snap_topn) / len(final_topn) if final_topn else 0
        print(f"Top-{n} overlap: {overlap:.1%}")
    
    # Save for further analysis
    out_path = 'data/predictions/v13_market_residual_2025_snapshot_recalc.parquet'
    pred_snap.to_parquet(out_path)
    print(f"\nSaved to {out_path}")
    
    return pred_snap

if __name__ == '__main__':
    main()
