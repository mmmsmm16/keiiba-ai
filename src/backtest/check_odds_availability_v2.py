"""
Phase 5 (v2): Check Odds Availability with JRA-Only Filter
オッズデータの可用性確認（JRAのみデフォルト）

Usage (in container):
    docker compose exec app python src/backtest/check_odds_availability_v2.py --period screening
    docker compose exec app python src/backtest/check_odds_availability_v2.py --period screening --include_nar
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


def check_duplicates(df: pd.DataFrame) -> Dict:
    """
    (race_id, umaban)の重複を検出
    """
    if 'umaban' not in df.columns:
        if 'horse_number' in df.columns:
            key_col = 'horse_number'
        else:
            return {'has_duplicates': False, 'duplicate_count': 0}
    else:
        key_col = 'umaban'
    
    dup_mask = df.duplicated(subset=['race_id', key_col], keep=False)
    dup_count = dup_mask.sum()
    
    return {
        'has_duplicates': dup_count > 0,
        'duplicate_count': int(dup_count),
        'duplicate_rate': dup_count / len(df) * 100 if len(df) > 0 else 0,
        'key_column': key_col
    }


def deduplicate(df: pd.DataFrame, strategy: str = 'first') -> pd.DataFrame:
    """
    (race_id, umaban)の重複を削除
    
    Args:
        df: 対象DataFrame
        strategy: 'first'=最初を残す, 'last'=最後を残す
    
    Returns:
        重複削除後のDataFrame
    """
    key_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    
    before_count = len(df)
    result = df.drop_duplicates(subset=['race_id', key_col], keep=strategy)
    after_count = len(result)
    
    if before_count != after_count:
        logger.warning(f"Dedup: {before_count:,} → {after_count:,} rows "
                       f"(-{before_count - after_count:,} duplicates)")
    
    return result


def check_race_complete_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    全頭のoddsが揃っているレースにフラグを付ける
    """
    # レースごとにodds非nullの割合を計算
    race_stats = df.groupby('race_id').agg({
        'odds': lambda x: x.notna().all(),
        'race_id': 'count'
    }).rename(columns={'odds': 'odds_complete', 'race_id': 'horse_count'})
    
    # マージ
    result = df.merge(race_stats[['odds_complete']], left_on='race_id', right_index=True, how='left')
    
    complete_races = race_stats['odds_complete'].sum()
    total_races = len(race_stats)
    logger.info(f"Races with complete odds: {complete_races:,}/{total_races:,} "
                f"({complete_races/total_races*100:.2f}%)")
    
    return result


def calculate_market_metrics(df: pd.DataFrame) -> Dict:
    """
    市場ベースライン指標を計算（同一母集団で）
    """
    # 有効データのみ
    valid = df[(df['odds'].notna()) & (df['odds'] > 0)].copy()
    
    if len(valid) == 0:
        return {'error': 'No valid odds data'}
    
    # p_market = 1 / odds
    valid['p_market'] = 1.0 / valid['odds']
    
    # レースごとにoverround計算
    race_overround = valid.groupby('race_id')['p_market'].sum().reset_index()
    race_overround.columns = ['race_id', 'overround']
    
    # p_market を正規化（確率として使用可能に）
    valid = valid.merge(race_overround, on='race_id', how='left')
    valid['p_market_norm'] = valid['p_market'] / valid['overround']
    
    # 勝馬フラグ
    valid['is_winner'] = (valid['rank'] == 1).astype(int)
    
    # LogLoss
    from sklearn.metrics import log_loss, roc_auc_score
    
    y_true = valid['is_winner'].values
    y_prob = np.clip(valid['p_market_norm'].values, 1e-15, 1 - 1e-15)
    
    try:
        market_logloss = log_loss(y_true, y_prob)
        market_auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        market_logloss = None
        market_auc = None
        logger.warning(f"Metric calculation error: {e}")
    
    return {
        'sample_count': len(valid),
        'race_count': valid['race_id'].nunique(),
        'overround_mean': race_overround['overround'].mean(),
        'overround_std': race_overround['overround'].std(),
        'overround_min': race_overround['overround'].min(),
        'overround_max': race_overround['overround'].max(),
        'market_logloss': market_logloss,
        'market_auc': market_auc,
        'race_overround': race_overround  # Return for diagnosis
    }


def diagnose_extreme_overround(df: pd.DataFrame, race_overround: pd.DataFrame, threshold: float = 1.05) -> Dict:
    """
    overround < threshold のレースを診断
    
    Args:
        df: 元データ
        race_overround: レースごとのoverround DataFrame
        threshold: 異常判定閾値（default: 1.05）
    
    Returns:
        診断結果Dict
    """
    extreme_races = race_overround[race_overround['overround'] < threshold]
    
    if len(extreme_races) == 0:
        return {
            'extreme_count': 0,
            'extreme_rate': 0.0,
            'details': []
        }
    
    logger.warning(f"Found {len(extreme_races)} races with overround < {threshold}")
    
    # 詳細診断
    details = []
    for _, row in extreme_races.head(10).iterrows():  # Top 10
        race_id = row['race_id']
        race_df = df[df['race_id'] == race_id]
        
        detail = {
            'race_id': race_id,
            'overround': row['overround'],
            'runner_count': len(race_df),
            'min_odds': race_df['odds'].min(),
            'max_odds': race_df['odds'].max(),
            'median_odds': race_df['odds'].median(),
            'top3_odds': race_df.nsmallest(3, 'odds')['odds'].tolist() if len(race_df) >= 3 else []
        }
        details.append(detail)
        
        logger.info(f"  Race {race_id}: overround={row['overround']:.4f}, "
                    f"runners={len(race_df)}, min_odds={detail['min_odds']:.1f}")
    
    return {
        'extreme_count': len(extreme_races),
        'extreme_rate': len(extreme_races) / len(race_overround) * 100,
        'extreme_race_ids': extreme_races['race_id'].tolist(),
        'details': details
    }


def filter_valid_overround(df: pd.DataFrame, min_overround: float = 1.05, max_overround: float = 1.35) -> pd.DataFrame:
    """
    overround が妥当な範囲のレースのみ抽出
    
    Args:
        df: 元データ
        min_overround: 最小閾値
        max_overround: 最大閾値
    
    Returns:
        フィルタ後のDataFrame
    """
    # overround計算
    df = df.copy()
    df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
    race_overround = df.groupby('race_id')['raw_prob'].sum().reset_index()
    race_overround.columns = ['race_id', 'overround']
    
    valid_races = race_overround[
        (race_overround['overround'] >= min_overround) &
        (race_overround['overround'] <= max_overround)
    ]['race_id'].tolist()
    
    before_count = df['race_id'].nunique()
    result = df[df['race_id'].isin(valid_races)]
    after_count = result['race_id'].nunique()
    
    excluded = before_count - after_count
    if excluded > 0:
        logger.info(f"Overround filter: {before_count} → {after_count} races "
                    f"(excluded {excluded} races with overround < {min_overround} or > {max_overround})")
    
    return result


def generate_report(
    stats: Dict,
    dup_info: Dict,
    market_metrics: Dict,
    output_path: str,
    period: PeriodConfig,
    filter_type: str
):
    """レポート生成"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = f"""# Phase 5 (v2): Odds Availability Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {period.start_year}-{period.end_year}
**Filter**: {filter_type}

## Data Summary

| Metric | Value |
|--------|-------|
| Total Rows | {stats['total_rows']:,} |
| Total Races | {stats['total_races']:,} |
| JRA Rows | {stats['jra_rows']:,} |
| JRA Races | {stats['jra_races']:,} |
| NAR Rows | {stats['nar_rows']:,} |
| NAR Races | {stats['nar_races']:,} |

## Duplicate Check

| Metric | Value |
|--------|-------|
| Has Duplicates | {dup_info['has_duplicates']} |
| Duplicate Count | {dup_info['duplicate_count']:,} |
| Duplicate Rate | {dup_info['duplicate_rate']:.4f}% |

## Market Metrics ({filter_type})

| Metric | Value |
|--------|-------|
| Sample Count | {market_metrics.get('sample_count', 'N/A'):,} |
| Race Count | {market_metrics.get('race_count', 'N/A'):,} |
| Overround Mean | {market_metrics.get('overround_mean', 'N/A'):.4f} |
| Overround Std | {market_metrics.get('overround_std', 'N/A'):.4f} |
| Overround Min | {market_metrics.get('overround_min', 'N/A'):.4f} |
| Overround Max | {market_metrics.get('overround_max', 'N/A'):.4f} |
| **Market LogLoss** | **{market_metrics.get('market_logloss', 'N/A'):.5f}** |
| Market AUC | {market_metrics.get('market_auc', 'N/A'):.5f} |

## Odds Coverage

| Metric | Value |
|--------|-------|
| Rows with Odds | {market_metrics.get('sample_count', 0):,} |
| Coverage Rate | {market_metrics.get('sample_count', 0) / stats['total_rows'] * 100 if stats['total_rows'] > 0 else 0:.2f}% |

"""
    
    # 判定
    coverage = market_metrics.get('sample_count', 0) / stats['total_rows'] * 100 if stats['total_rows'] > 0 else 0
    overround_ok = 1.15 <= market_metrics.get('overround_mean', 0) <= 1.35
    
    report += f"""
## Validation

- **Coverage >95%**: {'✅ PASS' if coverage > 95 else f'❌ FAIL ({coverage:.2f}%)'}
- **Overround in range [1.15, 1.35]**: {'✅ PASS' if overround_ok else f'❌ FAIL ({market_metrics.get("overround_mean", 0):.4f})'}
- **No extreme overround min**: {'✅ PASS' if market_metrics.get('overround_min', 0) > 0.9 else f'⚠️ WARNING (min={market_metrics.get("overround_min", 0):.4f})'}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 (v2): Check Odds Availability")
    add_period_args(parser)
    add_race_filter_args(parser)
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data_v11.parquet')
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
    
    # Get stats before race filter
    stats_before = get_race_stats(df)
    
    # Race filter (JRA-only by default)
    filter_type = "JRA-only" if not args.include_nar else "JRA+NAR"
    df = filter_races(df, include_nar=args.include_nar, include_overseas=args.include_overseas)
    
    # Check duplicates
    dup_info = check_duplicates(df)
    if dup_info['has_duplicates']:
        df = deduplicate(df)
    
    # Check race complete odds
    df = check_race_complete_odds(df)
    
    # Calculate market metrics
    market_metrics = calculate_market_metrics(df)
    
    # Get updated stats
    stats = get_race_stats(df)
    
    # Generate report
    report_name = 'phase5_odds_availability_jra.md' if not args.include_nar else 'phase5_odds_availability_all.md'
    generate_report(
        stats,
        dup_info,
        market_metrics,
        os.path.join(args.output_dir, report_name),
        period,
        filter_type
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
