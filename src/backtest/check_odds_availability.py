"""
Phase 5: Odds Availability Check & Market Baseline
オッズデータの可用性確認と市場確率ベースライン評価

Usage (in container):
    docker compose exec app python src/backtest/check_odds_availability.py --period screening
    docker compose exec app python src/backtest/check_odds_availability.py --start_year 2021 --end_year 2024
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, filter_dataframe_by_period
from betting.purchase_model import PurchaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OddsAvailabilityChecker:
    """オッズデータの可用性を確認し、カバー率を算出する"""
    
    def __init__(self, data_path: str, period_config):
        self.data_path = data_path
        self.period = period_config
        self.df = None
        self.results = {}
    
    def load_data(self):
        """データをロードし、期間でフィルタリング"""
        logger.info(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # 年カラムを確認・追加
        if 'year' not in self.df.columns and 'race_date' in self.df.columns:
            self.df['year'] = pd.to_datetime(self.df['race_date']).dt.year
        elif 'year' not in self.df.columns and 'race_id' in self.df.columns:
            # race_id から年を抽出（例: 202401010101 → 2024）
            self.df['year'] = self.df['race_id'].astype(str).str[:4].astype(int)
        
        # 期間フィルタ
        self.df = filter_dataframe_by_period(self.df, self.period)
        logger.info(f"Data loaded: {len(self.df):,} rows for period {self.period.start_year}-{self.period.end_year}")
    
    def check_odds_coverage(self):
        """オッズの欠損率・カバー率を計算"""
        logger.info("Checking odds coverage...")
        
        # オッズカラムを検出
        odds_col = 'odds' if 'odds' in self.df.columns else 'win_odds'
        if odds_col not in self.df.columns:
            logger.error("No odds column found (tried 'odds', 'win_odds')")
            return
        
        # 有効なオッズの条件
        valid_odds = self.df[odds_col].notna() & (self.df[odds_col] > 0)
        
        # Row-level coverage
        total_rows = len(self.df)
        valid_rows = valid_odds.sum()
        row_coverage = valid_rows / total_rows * 100 if total_rows > 0 else 0
        
        # Race-level coverage
        race_id_col = 'race_id'
        total_races = self.df[race_id_col].nunique()
        
        # レース内に1頭でも有効なオッズがあればカウント
        races_with_odds = self.df[valid_odds].groupby(race_id_col).size()
        races_with_valid_odds = len(races_with_odds)
        race_coverage = races_with_valid_odds / total_races * 100 if total_races > 0 else 0
        
        # 年別カバー率
        yearly_coverage = []
        for year in self.period.years:
            year_df = self.df[self.df['year'] == year]
            if len(year_df) > 0:
                year_valid = year_df[odds_col].notna() & (year_df[odds_col] > 0)
                yearly_coverage.append({
                    'year': year,
                    'total_rows': len(year_df),
                    'valid_rows': year_valid.sum(),
                    'row_coverage': year_valid.sum() / len(year_df) * 100,
                    'total_races': year_df[race_id_col].nunique(),
                    'races_with_odds': year_df[year_valid].groupby(race_id_col).ngroups
                })
        
        self.results['odds_coverage'] = {
            'total_rows': total_rows,
            'valid_rows': int(valid_rows),
            'row_coverage': row_coverage,
            'total_races': total_races,
            'races_with_valid_odds': races_with_valid_odds,
            'race_coverage': race_coverage,
            'yearly': yearly_coverage,
            'odds_column': odds_col
        }
        
        logger.info(f"Row coverage: {row_coverage:.2f}% ({valid_rows:,}/{total_rows:,})")
        logger.info(f"Race coverage: {race_coverage:.2f}% ({races_with_valid_odds:,}/{total_races:,})")
        
        return self.results['odds_coverage']
    
    def generate_report(self, output_path: str):
        """カバー率レポートを生成"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cov = self.results.get('odds_coverage', {})
        
        report = f"""# Phase 5: Odds Coverage Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {self.period.start_year}-{self.period.end_year}
**Odds Column**: `{cov.get('odds_column', 'N/A')}`

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Row Coverage** | {cov.get('row_coverage', 0):.2f}% | > 95% | {'✅ PASS' if cov.get('row_coverage', 0) > 95 else '⚠️ WARNING'} |
| **Race Coverage** | {cov.get('race_coverage', 0):.2f}% | > 95% | {'✅ PASS' if cov.get('race_coverage', 0) > 95 else '⚠️ WARNING'} |

## Details

- Total rows: {cov.get('total_rows', 0):,}
- Valid rows (odds > 0): {cov.get('valid_rows', 0):,}
- Total races: {cov.get('total_races', 0):,}
- Races with valid odds: {cov.get('races_with_valid_odds', 0):,}

## Yearly Breakdown

| Year | Total Rows | Valid Rows | Row Coverage | Races | Races w/ Odds |
|------|------------|------------|--------------|-------|---------------|
"""
        for y in cov.get('yearly', []):
            report += f"| {y['year']} | {y['total_rows']:,} | {y['valid_rows']:,} | {y['row_coverage']:.2f}% | {y['total_races']:,} | {y['races_with_odds']:,} |\n"
        
        report += """
## Missing Data Handling

> [!WARNING]
> レースでオッズが欠損している行は、LogLoss/Brier/ROI評価から除外します。
> これにより、評価対象の母集団が統一されます。

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
        return output_path


class MarketBaselineEvaluator:
    """市場確率（p_market）のベースライン評価"""
    
    def __init__(self, data_path: str, period_config):
        self.data_path = data_path
        self.period = period_config
        self.df = None
        self.results = {}
        self.pm = PurchaseModel()
    
    def load_data(self):
        """データロード"""
        logger.info(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # 年カラム追加
        if 'year' not in self.df.columns and 'race_id' in self.df.columns:
            self.df['year'] = self.df['race_id'].astype(str).str[:4].astype(int)
        
        # 期間フィルタ
        self.df = filter_dataframe_by_period(self.df, self.period)
        
        # オッズ有効行のみ
        odds_col = 'odds' if 'odds' in self.df.columns else 'win_odds'
        self.odds_col = odds_col
        valid_mask = self.df[odds_col].notna() & (self.df[odds_col] > 0)
        self.df = self.df[valid_mask].copy()
        
        logger.info(f"Data loaded: {len(self.df):,} valid rows")
    
    def calculate_market_probability(self):
        """市場確率とoverround算出"""
        logger.info("Calculating market probability and overround...")
        
        # PurchaseModel を使用
        self.df = self.pm.calculate_market_probability(
            self.df, 
            odds_col=self.odds_col, 
            race_id_col='race_id'
        )
        
        # Overround統計
        overround_stats = self.df.groupby('race_id')['overround'].first()
        self.results['overround'] = {
            'mean': float(overround_stats.mean()),
            'std': float(overround_stats.std()),
            'min': float(overround_stats.min()),
            'max': float(overround_stats.max()),
            'median': float(overround_stats.median())
        }
        
        logger.info(f"Overround: mean={self.results['overround']['mean']:.3f}, "
                   f"std={self.results['overround']['std']:.3f}")
    
    def evaluate_market_accuracy(self):
        """市場予測の精度評価（LogLoss, Brier, AUC）"""
        from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
        
        logger.info("Evaluating market prediction accuracy...")
        
        # ターゲット（1着か否か）
        target_col = 'rank' if 'rank' in self.df.columns else 'finish_position'
        if target_col not in self.df.columns:
            # rankがない場合、オッズ最小を1位とする代替
            logger.warning(f"No {target_col} column found. Using odds-based proxy.")
            self.df['is_winner'] = (self.df.groupby('race_id')[self.odds_col]
                                    .transform('min') == self.df[self.odds_col]).astype(int)
        else:
            self.df['is_winner'] = (self.df[target_col] == 1).astype(int)
        
        # 有効なp_marketのみ
        valid_mask = self.df['p_market'].notna() & (self.df['p_market'] > 0) & (self.df['p_market'] < 1)
        eval_df = self.df[valid_mask].copy()
        
        y_true = eval_df['is_winner'].values
        p_market = eval_df['p_market'].values
        
        # Clip for numerical stability
        p_market = np.clip(p_market, 1e-7, 1 - 1e-7)
        
        # Metrics
        try:
            logloss = log_loss(y_true, p_market)
            brier = brier_score_loss(y_true, p_market)
            auc = roc_auc_score(y_true, p_market)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logloss, brier, auc = np.nan, np.nan, np.nan
        
        self.results['market_accuracy'] = {
            'logloss': float(logloss),
            'brier': float(brier),
            'auc': float(auc),
            'n_samples': len(eval_df)
        }
        
        logger.info(f"Market LogLoss: {logloss:.5f}")
        logger.info(f"Market Brier: {brier:.5f}")
        logger.info(f"Market AUC: {auc:.5f}")
    
    def calculate_yearly_overround(self):
        """年別overround"""
        yearly = []
        for year in self.period.years:
            year_races = self.df[self.df['year'] == year].groupby('race_id')['overround'].first()
            if len(year_races) > 0:
                yearly.append({
                    'year': year,
                    'mean_overround': float(year_races.mean()),
                    'std_overround': float(year_races.std()),
                    'n_races': len(year_races)
                })
        self.results['yearly_overround'] = yearly
    
    def generate_report(self, output_path: str):
        """マーケットベースラインレポート生成"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        ov = self.results.get('overround', {})
        acc = self.results.get('market_accuracy', {})
        
        report = f"""# Phase 5: Market Baseline Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {self.period.start_year}-{self.period.end_year}

## 1. Market Probability Definition

```
p_market = (1 / odds) / sum(1 / odds)
overround = sum(1 / odds)
```

## 2. Overround Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean** | {ov.get('mean', 0):.4f} | {'標準的（15-20%控除）' if 1.15 <= ov.get('mean', 0) <= 1.25 else '異常値の可能性'} |
| Std | {ov.get('std', 0):.4f} | |
| Min | {ov.get('min', 0):.4f} | |
| Max | {ov.get('max', 0):.4f} | |
| Median | {ov.get('median', 0):.4f} | |

> [!NOTE]
> overround ≈ 1.15-1.20 が標準的な控除率（15-20%）を示す

## 3. Market Prediction Accuracy (p_market as predictor)

| Metric | Value | Description |
|--------|-------|-------------|
| **LogLoss** | {acc.get('logloss', 0):.5f} | 市場確率の対数損失 |
| **Brier Score** | {acc.get('brier', 0):.5f} | 予測確率の二乗誤差 |
| **AUC** | {acc.get('auc', 0):.5f} | 識別能力 |
| Samples | {acc.get('n_samples', 0):,} | 評価対象行数 |

> [!IMPORTANT]
> モデル（p_model）がこの市場ベースラインを上回ることが価値提供の条件

## 4. Yearly Overround

| Year | Mean Overround | Std | N Races |
|------|----------------|-----|---------|
"""
        for y in self.results.get('yearly_overround', []):
            report += f"| {y['year']} | {y['mean_overround']:.4f} | {y['std_overround']:.4f} | {y['n_races']:,} |\n"
        
        report += """
## 5. Interpretation

- **overroundが高い**: 市場の控除が大きく、ROI達成が難しい
- **overroundが低い**: 市場に隙がある可能性（稀）
- **市場AUCが高い**: 市場は人気馬を正しく識別している
- **市場LogLossが低い**: 市場確率のCalibrationが良好

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Odds Availability & Market Baseline Check"
    )
    add_period_args(parser)
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed/preprocessed_data_v11.parquet',
        help='Path to preprocessed data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['coverage', 'baseline', 'all'],
        default='all',
        help='What to run'
    )
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period validation failed: {e}")
        sys.exit(1)
    
    logger.info(f"Period: {period.start_year}-{period.end_year}")
    
    # 1. Odds Coverage Check
    if args.mode in ['coverage', 'all']:
        checker = OddsAvailabilityChecker(args.data_path, period)
        checker.load_data()
        checker.check_odds_coverage()
        checker.generate_report(
            os.path.join(args.output_dir, 'phase5_odds_coverage.md')
        )
    
    # 2. Market Baseline
    if args.mode in ['baseline', 'all']:
        evaluator = MarketBaselineEvaluator(args.data_path, period)
        evaluator.load_data()
        evaluator.calculate_market_probability()
        evaluator.evaluate_market_accuracy()
        evaluator.calculate_yearly_overround()
        evaluator.generate_report(
            os.path.join(args.output_dir, 'phase5_market_baseline.md')
        )
    
    logger.info("Phase 5 check completed!")


if __name__ == "__main__":
    main()
