"""
Phase 6: Odds Bin Diagnostics
オッズbin別のLogLoss/ECE/ROI診断

Usage (in container):
    docker compose exec app python src/evaluation/odds_bin_diagnostics.py
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            ece += mask.sum() * np.abs(bin_accuracy - bin_confidence)
    
    return ece / len(y_true) if len(y_true) > 0 else 0.0


def analyze_odds_bins(df: pd.DataFrame, prob_col: str = 'prob') -> List[Dict]:
    """オッズbin別に分析"""
    
    # Intersection filter
    valid = df[(df['odds'].notna()) & (df['odds'] > 0) & 
               (df[prob_col].notna()) & (df['rank'].notna())].copy()
    
    valid['is_winner'] = (valid['rank'] == 1).astype(int)
    
    # Market probability
    valid['p_market_raw'] = 1.0 / valid['odds']
    valid['p_market'] = valid.groupby('race_id')['p_market_raw'].transform(lambda x: x / x.sum())
    
    # Odds bins
    odds_bins = [
        (1.0, 2.0, '1.0-2.0 (大本命)'),
        (2.0, 5.0, '2.0-5.0 (本命)'),
        (5.0, 10.0, '5.0-10.0 (人気)'),
        (10.0, 20.0, '10.0-20.0 (中穴)'),
        (20.0, 50.0, '20.0-50.0 (穴)'),
        (50.0, 100.0, '50.0-100.0 (大穴)'),
        (100.0, 9999.0, '100.0+ (超大穴)')
    ]
    
    results = []
    
    for min_odds, max_odds, label in odds_bins:
        mask = (valid['odds'] >= min_odds) & (valid['odds'] < max_odds)
        bin_df = valid[mask]
        
        if len(bin_df) < 100:
            continue
        
        y_true = bin_df['is_winner'].values
        y_model = np.clip(bin_df[prob_col].values, 1e-15, 1 - 1e-15)
        y_market = np.clip(bin_df['p_market'].values, 1e-15, 1 - 1e-15)
        
        # Metrics
        try:
            model_ll = log_loss(y_true, y_model)
            market_ll = log_loss(y_true, y_market)
            model_ece = compute_ece(y_true, y_model)
            market_ece = compute_ece(y_true, y_market)
        except Exception:
            continue
        
        # Win rate
        actual_win_rate = y_true.mean() * 100
        model_mean_prob = y_model.mean() * 100
        market_mean_prob = y_market.mean() * 100
        
        # Fair odds vs actual odds
        avg_odds = bin_df['odds'].mean()
        fair_odds = 100 / actual_win_rate if actual_win_rate > 0 else 0
        
        # ROI if betting all in this bin (単勝)
        total_bet = len(bin_df) * 100
        total_payout = (bin_df[bin_df['is_winner'] == 1]['odds'] * 100).sum()
        roi = (total_payout / total_bet * 100) if total_bet > 0 else 0
        
        results.append({
            'bin': label,
            'count': len(bin_df),
            'races': bin_df['race_id'].nunique(),
            'win_rate': actual_win_rate,
            'model_prob': model_mean_prob,
            'market_prob': market_mean_prob,
            'avg_odds': avg_odds,
            'fair_odds': fair_odds,
            'model_logloss': model_ll,
            'market_logloss': market_ll,
            'll_gap': model_ll - market_ll,
            'model_ece': model_ece,
            'market_ece': market_ece,
            'roi': roi
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Odds Bin Diagnostics")
    parser.add_argument('--input', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--year', type=int, default=2024)
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # JRA-only, specific year
    df = filter_jra_only(df)
    df = df[df['year'] == args.year]
    
    logger.info(f"JRA-only {args.year}: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Analyze
    results = analyze_odds_bins(df)
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'diagnostics_odds_bins.md')
    
    report = f"""# Odds Bin Diagnostics Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {args.year}
**Filter**: JRA-only (intersection)

## Odds Bin Analysis

| Bin | Count | Win% | Model% | Market% | Model LL | Market LL | Gap | ECE Model | ECE Market | ROI |
|-----|-------|------|--------|---------|----------|-----------|-----|-----------|------------|-----|
"""
    
    for r in results:
        report += f"| {r['bin']} | {r['count']:,} | {r['win_rate']:.1f}% | {r['model_prob']:.1f}% | {r['market_prob']:.1f}% | {r['model_logloss']:.4f} | {r['market_logloss']:.4f} | {r['ll_gap']:+.4f} | {r['model_ece']:.4f} | {r['market_ece']:.4f} | {r['roi']:.1f}% |\n"
    
    # Summary
    total_model_ll = sum(r['model_logloss'] * r['count'] for r in results) / sum(r['count'] for r in results)
    total_market_ll = sum(r['market_logloss'] * r['count'] for r in results) / sum(r['count'] for r in results)
    
    report += f"""
## Summary

- **Weighted Model LogLoss**: {total_model_ll:.5f}
- **Weighted Market LogLoss**: {total_market_ll:.5f}
- **Gap**: {total_model_ll - total_market_ll:+.5f}

## Observations

"""
    
    # Identify problematic bins
    for r in results:
        if r['ll_gap'] > 0.03:
            report += f"- ⚠️ **{r['bin']}**: モデルがMarketより+{r['ll_gap']:.4f}悪い\n"
        elif r['ll_gap'] < -0.01:
            report += f"- ✅ **{r['bin']}**: モデルがMarketより{r['ll_gap']:+.4f}良い\n"
    
    # Calibration issues
    report += "\n### Calibration Issues\n\n"
    for r in results:
        calibration_error = r['model_prob'] - r['win_rate']
        if abs(calibration_error) > 2:
            direction = "過大評価" if calibration_error > 0 else "過小評価"
            report += f"- {r['bin']}: Model probability {direction} ({r['model_prob']:.1f}% vs actual {r['win_rate']:.1f}%)\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
