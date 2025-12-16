"""
Diagnostics: Sum(Prob) Diagnostic
レース内確率合計の診断・正規化

Usage (in container):
    docker compose exec app python src/diagnostics/sum_prob_diagnostic.py --year 2024
    docker compose exec app python src/diagnostics/sum_prob_diagnostic.py --input data/predictions/calibrated/v12_oof_unified.parquet
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Standard probability columns to diagnose
STANDARD_PROB_COLS = [
    ('prob_model_raw', 'build_predictions_table.py'),
    ('prob_model_norm', 'build_predictions_table.py'),
    ('prob_model_calib_temp', 'build_predictions_table.py'),
    ('prob_model_calib_isotonic', 'build_predictions_table.py'),
    ('prob_model_calib_beta_full', 'build_predictions_table.py'),
    ('p_market', 'build_predictions_table.py'),
    ('p_blend', 'market_blend_lambda_wf.py'),
    # v13 market-residual columns
    ('prob_residual_raw', 'train_market_residual_wf.py'),
    ('prob_residual_norm', 'train_market_residual_wf.py'),
    ('prob_residual_softmax', 'train_market_residual_wf.py'),
]


def log_input_schema(df: pd.DataFrame):
    """入力スキーマをログ出力"""
    logger.info("=== Input DataFrame Schema ===")
    logger.info(f"Rows: {len(df):,}, Columns: {len(df.columns)}")
    for col in sorted(df.columns):
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  {col}: {n_valid:,} ({pct:.1f}%)")
    logger.info("=" * 40)


def diagnose_prob_column(df: pd.DataFrame, prob_col: str, race_col: str = 'race_id') -> Dict:
    """
    指定確率列のレース内合計を診断
    
    Args:
        df: データ
        prob_col: 確率列名
        race_col: レースID列名
    
    Returns:
        診断結果Dict
    """
    if prob_col not in df.columns:
        return {'exists': False, 'column': prob_col}
    
    valid = df[df[prob_col].notna()].copy()
    
    if len(valid) == 0:
        return {'exists': True, 'column': prob_col, 'valid_rows': 0}
    
    # レース内合計を計算
    race_sum = valid.groupby(race_col)[prob_col].sum()
    
    # 分布統計
    stats = {
        'exists': True,
        'column': prob_col,
        'valid_rows': len(valid),
        'valid_races': len(race_sum),
        'sum_mean': float(race_sum.mean()),
        'sum_median': float(race_sum.median()),
        'sum_std': float(race_sum.std()),
        'sum_min': float(race_sum.min()),
        'sum_p01': float(race_sum.quantile(0.01)),
        'sum_p05': float(race_sum.quantile(0.05)),
        'sum_p95': float(race_sum.quantile(0.95)),
        'sum_p99': float(race_sum.quantile(0.99)),
        'sum_max': float(race_sum.max()),
    }
    
    # |sum - 1| <= threshold の割合
    stats['pct_within_001'] = float((abs(race_sum - 1) <= 0.01).mean() * 100)
    stats['pct_within_005'] = float((abs(race_sum - 1) <= 0.05).mean() * 100)
    
    # 異常検知
    stats['n_sum_zero'] = int((race_sum <= 0).sum())
    stats['n_prob_negative'] = int((valid[prob_col] < 0).sum())
    stats['n_prob_gt_one'] = int((valid[prob_col] > 1).sum())
    stats['n_sum_lt_05'] = int((race_sum < 0.5).sum())
    stats['n_sum_gt_15'] = int((race_sum > 1.5).sum())
    
    # 極端なレースのサンプル
    extreme_races = race_sum[(race_sum < 0.5) | (race_sum > 1.5)]
    stats['extreme_race_samples'] = extreme_races.head(5).to_dict()
    
    # Top1 probの診断（可能なら）
    if 'rank' in valid.columns:
        # 各レースでmax probの馬
        idx_max = valid.groupby(race_col)[prob_col].idxmax()
        top1_df = valid.loc[idx_max]
        
        stats['top1_avg_prob'] = float(top1_df[prob_col].mean())
        stats['top1_win_rate'] = float((top1_df['rank'] == 1).mean() * 100)
    
    return stats


def normalize_prob_column(
    df: pd.DataFrame, 
    prob_col: str, 
    out_col: str = 'prob_model_norm',
    race_col: str = 'race_id'
) -> pd.DataFrame:
    """
    確率列をレース内正規化
    
    Args:
        df: データ
        prob_col: 元の確率列
        out_col: 出力列名
        race_col: レースID列
    
    Returns:
        正規化列を追加したDataFrame
    """
    df = df.copy()
    
    # レース内合計を計算
    race_sum = df.groupby(race_col)[prob_col].transform('sum')
    
    # 正規化（sum=0のレースはNaN）
    df[out_col] = np.where(
        race_sum > 0,
        df[prob_col] / race_sum,
        np.nan
    )
    
    # sum=0のレース数をログ
    n_zero_sum_races = (df.groupby(race_col)[prob_col].sum() <= 0).sum()
    if n_zero_sum_races > 0:
        logger.warning(f"Found {n_zero_sum_races} races with sum(prob) <= 0, set {out_col} to NaN")
    
    return df


def generate_report(
    diagnostics: List[Dict],
    output_path: str,
    year: int,
    total_rows: int,
    total_races: int
):
    """診断レポート生成"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = f"""# Sum(Prob) Diagnostic Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {year}
**Filter**: JRA-only (intersection)
**Total Rows**: {total_rows:,}
**Total Races**: {total_races:,}

## Probability Column Diagnostics

"""
    
    for diag in diagnostics:
        col = diag.get('column', 'unknown')
        
        if not diag.get('exists', False):
            report += f"### {col}\n\n❌ Column not found\n\n"
            continue
        
        if diag.get('valid_rows', 0) == 0:
            report += f"### {col}\n\n⚠️ No valid data\n\n"
            continue
        
        report += f"### {col}\n\n"
        report += f"**Valid Rows**: {diag['valid_rows']:,} | **Valid Races**: {diag['valid_races']:,}\n\n"
        
        report += "#### Sum Distribution (per race)\n\n"
        report += "| Statistic | Value |\n"
        report += "|-----------|-------|\n"
        report += f"| Mean | {diag['sum_mean']:.4f} |\n"
        report += f"| Median | {diag['sum_median']:.4f} |\n"
        report += f"| Std | {diag['sum_std']:.4f} |\n"
        report += f"| Min | {diag['sum_min']:.4f} |\n"
        report += f"| P01 | {diag['sum_p01']:.4f} |\n"
        report += f"| P05 | {diag['sum_p05']:.4f} |\n"
        report += f"| P95 | {diag['sum_p95']:.4f} |\n"
        report += f"| P99 | {diag['sum_p99']:.4f} |\n"
        report += f"| Max | {diag['sum_max']:.4f} |\n\n"
        
        report += "#### Normality Check\n\n"
        report += f"- |sum - 1| ≤ 0.01: **{diag['pct_within_001']:.1f}%**\n"
        report += f"- |sum - 1| ≤ 0.05: **{diag['pct_within_005']:.1f}%**\n\n"
        
        report += "#### Anomaly Detection\n\n"
        report += f"- sum ≤ 0: {diag['n_sum_zero']} races\n"
        report += f"- prob < 0: {diag['n_prob_negative']} rows\n"
        report += f"- prob > 1: {diag['n_prob_gt_one']} rows\n"
        report += f"- sum < 0.5: {diag['n_sum_lt_05']} races\n"
        report += f"- sum > 1.5: {diag['n_sum_gt_15']} races\n\n"
        
        if 'top1_avg_prob' in diag:
            report += "#### Top1 Prediction Quality\n\n"
            report += f"- Avg Top1 Prob: {diag['top1_avg_prob']:.4f}\n"
            report += f"- Top1 Win Rate: {diag['top1_win_rate']:.1f}%\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sum(Prob) Diagnostic")
    parser.add_argument('--year', type=int, default=None, help='Filter by year (optional)')
    parser.add_argument('--input', type=str, default='data/predictions/calibrated/v12_oof_unified.parquet')
    parser.add_argument('--output_dir', type=str, default='reports/diagnostics')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Log input schema
    log_input_schema(df)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # Year filter (optional)
    if args.year is not None:
        df = df[df['year'] == args.year].copy()
        logger.info(f"Year filter applied: {args.year}")
    
    # JRA-only filter (if not already applied)
    if df['race_id'].astype(str).str[4:5].eq('0').mean() < 0.99:
        df = filter_jra_only(df)
    
    logger.info(f"Data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Run diagnostics for standard columns
    diagnostics = []
    for col_name, generator_script in STANDARD_PROB_COLS:
        diag = diagnose_prob_column(df, col_name)
        diag['generator_script'] = generator_script
        diagnostics.append(diag)
        
        if diag.get('exists') and diag.get('valid_rows', 0) > 0:
            logger.info(f"✅ {col_name}: sum mean={diag['sum_mean']:.4f}, "
                       f"within 1%: {diag['pct_within_001']:.1f}%")
        elif not diag.get('exists'):
            logger.warning(f"❌ {col_name}: Column not found (generate with: {generator_script})")
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    year_str = str(args.year) if args.year else 'all'
    report_path = os.path.join(args.output_dir, 'sum_prob_diagnostic.md')
    
    # Enhanced report with hints
    report = f"""# Sum(Prob) Diagnostic Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {year_str}
**Filter**: JRA-only
**Total Rows**: {len(df):,}
**Total Races**: {df['race_id'].nunique():,}

## Input Columns

Available columns in input file:
"""
    for col in sorted(df.columns):
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        report += f"- `{col}`: {n_valid:,} ({pct:.1f}%)\n"
    
    report += "\n## Probability Column Diagnostics\n\n"
    
    for diag in diagnostics:
        col = diag.get('column', 'unknown')
        gen_script = diag.get('generator_script', 'unknown')
        
        if not diag.get('exists', False):
            report += f"### {col}\n\n❌ Column not found\n\n"
            report += f"> **Hint**: Generate with `python src/phase6/{gen_script}`\n\n"
            continue
        
        if diag.get('valid_rows', 0) == 0:
            report += f"### {col}\n\n⚠️ No valid data\n\n"
            continue
        
        report += f"### {col}\n\n"
        report += f"**Valid Rows**: {diag['valid_rows']:,} | **Valid Races**: {diag['valid_races']:,}\n\n"
        
        report += "#### Sum Distribution (per race)\n\n"
        report += "| Statistic | Value |\n"
        report += "|-----------|-------|\n"
        report += f"| Mean | {diag['sum_mean']:.4f} |\n"
        report += f"| Median | {diag['sum_median']:.4f} |\n"
        report += f"| Std | {diag['sum_std']:.4f} |\n"
        report += f"| Min | {diag['sum_min']:.4f} |\n"
        report += f"| P01 | {diag['sum_p01']:.4f} |\n"
        report += f"| P05 | {diag['sum_p05']:.4f} |\n"
        report += f"| P95 | {diag['sum_p95']:.4f} |\n"
        report += f"| P99 | {diag['sum_p99']:.4f} |\n"
        report += f"| Max | {diag['sum_max']:.4f} |\n\n"
        
        report += "#### Normality Check\n\n"
        report += f"- |sum - 1| ≤ 0.01: **{diag['pct_within_001']:.1f}%**\n"
        report += f"- |sum - 1| ≤ 0.05: **{diag['pct_within_005']:.1f}%**\n\n"
        
        report += "#### Anomaly Detection\n\n"
        report += f"- sum ≤ 0: {diag['n_sum_zero']} races\n"
        report += f"- prob < 0: {diag['n_prob_negative']} rows\n"
        report += f"- prob > 1: {diag['n_prob_gt_one']} rows\n"
        report += f"- sum < 0.5: {diag['n_sum_lt_05']} races\n"
        report += f"- sum > 1.5: {diag['n_sum_gt_15']} races\n\n"
        
        if 'top1_avg_prob' in diag:
            report += "#### Top1 Prediction Quality\n\n"
            report += f"- Avg Top1 Prob: {diag['top1_avg_prob']:.4f}\n"
            report += f"- Top1 Win Rate: {diag['top1_win_rate']:.1f}%\n\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
