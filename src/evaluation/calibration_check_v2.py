"""
Phase 6 (v2): Calibration Check with Same-Population Metrics
Model vs Market の calibration 比較（同一母集団で）

Usage (in container):
    docker compose exec app python src/evaluation/calibration_check_v2.py --period screening
    docker compose exec app python src/evaluation/calibration_check_v2.py --period screening --include_nar
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, PeriodConfig
from utils.race_filter import filter_races, add_race_filter_args, get_race_stats

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


def get_intersection_data(df: pd.DataFrame, prob_col: str = 'prob') -> pd.DataFrame:
    """
    Market と Model の両方が揃っている行のみ抽出
    
    条件:
    - odds が存在 (> 0)
    - prob (モデル予測) が存在
    - rank が存在 (勝馬判定用)
    """
    mask = (
        (df['odds'].notna()) & 
        (df['odds'] > 0) &
        (df[prob_col].notna()) &
        (df['rank'].notna())
    )
    
    result = df[mask].copy()
    logger.info(f"Intersection filter: {len(df):,} → {len(result):,} rows "
                f"({len(result)/len(df)*100:.1f}%)")
    
    return result


def calculate_metrics(df: pd.DataFrame, prob_col: str, source_name: str) -> Dict:
    """
    LogLoss / Brier / AUC / ECE を計算
    """
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    
    y_true = (df['rank'] == 1).astype(int).values
    y_prob = np.clip(df[prob_col].values, 1e-15, 1 - 1e-15)
    
    try:
        ll = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)
    except Exception as e:
        logger.error(f"Error calculating metrics for {source_name}: {e}")
        return {'error': str(e)}
    
    return {
        'source': source_name,
        'sample_count': len(df),
        'race_count': df['race_id'].nunique(),
        'winner_count': int(y_true.sum()),
        'logloss': ll,
        'brier': brier,
        'auc': auc,
        'ece': ece
    }


def generate_report(
    market_metrics: Dict,
    model_metrics: Dict,
    output_path: str,
    period: PeriodConfig,
    filter_type: str
):
    """レポート生成"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = f"""# Phase 6 (v2): Calibration Check Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {period.start_year}-{period.end_year}
**Filter**: {filter_type}

## Same-Population Metrics Comparison

> ✅ Market と Model は **同一のサンプル集合** で計算されています

| Metric | Market | Model | Winner? |
|--------|--------|-------|---------|
| Sample Count | {market_metrics['sample_count']:,} | {model_metrics['sample_count']:,} | {'✅ Same' if market_metrics['sample_count'] == model_metrics['sample_count'] else '❌ Diff'} |
| Race Count | {market_metrics['race_count']:,} | {model_metrics['race_count']:,} | {'✅ Same' if market_metrics['race_count'] == model_metrics['race_count'] else '❌ Diff'} |
| **LogLoss** | **{market_metrics['logloss']:.5f}** | **{model_metrics['logloss']:.5f}** | **{'🏆 Market' if market_metrics['logloss'] < model_metrics['logloss'] else '🏆 Model'}** |
| Brier Score | {market_metrics['brier']:.5f} | {model_metrics['brier']:.5f} | {'Market' if market_metrics['brier'] < model_metrics['brier'] else 'Model'} |
| AUC | {market_metrics['auc']:.5f} | {model_metrics['auc']:.5f} | {'Market' if market_metrics['auc'] > model_metrics['auc'] else 'Model'} |
| ECE | {market_metrics['ece']:.5f} | {model_metrics['ece']:.5f} | {'Market' if market_metrics['ece'] < model_metrics['ece'] else 'Model'} |

## LogLoss Difference

- Market LogLoss: {market_metrics['logloss']:.5f}
- Model LogLoss: {model_metrics['logloss']:.5f}
- **Delta**: {model_metrics['logloss'] - market_metrics['logloss']:.5f} ({'Model worse' if model_metrics['logloss'] > market_metrics['logloss'] else 'Model better'})

## Notes

- Market probability: `p_market = (1/odds) / sum(1/odds)`
- Model probability: `prob` column from predictions
- データ母集団: **完全一致** (intersection filter適用)
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 (v2): Calibration Check")
    add_period_args(parser)
    add_race_filter_args(parser)
    parser.add_argument('--input', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--prob_col', type=str, default='prob')
    parser.add_argument('--output_dir', type=str, default='reports')
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period error: {e}")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # Period filter
    df = df[(df['year'] >= period.start_year) & (df['year'] <= period.end_year)]
    logger.info(f"Period filtered: {len(df):,} rows")
    
    # Race filter (JRA-only by default)
    filter_type = "JRA-only" if not args.include_nar else "JRA+NAR"
    df = filter_races(df, include_nar=args.include_nar, include_overseas=args.include_overseas)
    
    # Intersection filter (same population for market and model)
    df = get_intersection_data(df, prob_col=args.prob_col)
    
    if len(df) == 0:
        logger.error("No data after intersection filter")
        return
    
    # Calculate p_market
    df['raw_prob'] = 1.0 / df['odds']
    df['p_market'] = df.groupby('race_id')['raw_prob'].transform(lambda x: x / x.sum())
    
    # Calculate metrics for both
    market_metrics = calculate_metrics(df, 'p_market', 'Market')
    model_metrics = calculate_metrics(df, args.prob_col, 'Model')
    
    logger.info(f"Market LogLoss: {market_metrics['logloss']:.5f}")
    logger.info(f"Model LogLoss: {model_metrics['logloss']:.5f}")
    
    # Generate report
    report_name = 'phase6_calibration_jra.md' if not args.include_nar else 'phase6_calibration_all.md'
    generate_report(
        market_metrics,
        model_metrics,
        os.path.join(args.output_dir, report_name),
        period,
        filter_type
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
