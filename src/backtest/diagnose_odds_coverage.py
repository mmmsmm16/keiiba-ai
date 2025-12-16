"""
Phase 5 Prerequisite: Odds Coverage Diagnostic
オッズカバー率の欠損原因を診断

Usage (in container):
    docker compose exec app python src/backtest/diagnose_odds_coverage.py --data_path data/processed/preprocessed_data_v11.parquet

診断項目:
- 年別/場別欠損率
- キー不一致検出 (race_id, umaban)
- JOINキーの型差分チェック
- 重複行チェック
- 欠損サンプル出力
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OddsCoverageDiagnostic:
    """オッズカバー率の診断ツール"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.odds_col = None
        self.diagnostics = {}
    
    def load_data(self):
        """データロード"""
        logger.info(f"Loading {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # odds列検出
        self.odds_col = 'odds' if 'odds' in self.df.columns else 'win_odds'
        if self.odds_col not in self.df.columns:
            raise ValueError("No odds column found")
        
        # year列追加
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['race_id'].astype(str).str[:4].astype(int)
        
        # venue (競馬場) を抽出
        if 'venue' not in self.df.columns and 'race_id' in self.df.columns:
            # race_id形式: YYYYJJDDRRXX (J=場、D=日目、R=レース)
            self.df['venue_code'] = self.df['race_id'].astype(str).str[4:6]
        
        logger.info(f"Loaded {len(self.df):,} rows")
    
    def run_diagnostics(self):
        """全診断を実行"""
        self._check_basic_coverage()
        self._check_yearly_breakdown()
        self._check_venue_breakdown()
        self._check_key_types()
        self._check_duplicates()
        self._sample_missing_records()
        self._check_odds_values()
    
    def _check_basic_coverage(self):
        """基本カバー率"""
        logger.info("\n=== Basic Coverage ===")
        
        valid_mask = self.df[self.odds_col].notna() & (self.df[self.odds_col] > 0)
        
        total = len(self.df)
        valid = valid_mask.sum()
        missing = total - valid
        
        # Null vs Zero vs Negative
        null_count = self.df[self.odds_col].isna().sum()
        zero_count = (self.df[self.odds_col] == 0).sum()
        negative_count = (self.df[self.odds_col] < 0).sum()
        
        self.diagnostics['basic'] = {
            'total_rows': total,
            'valid_rows': int(valid),
            'row_coverage': valid / total * 100,
            'null_count': int(null_count),
            'zero_count': int(zero_count),
            'negative_count': int(negative_count)
        }
        
        logger.info(f"Total rows: {total:,}")
        logger.info(f"Valid odds (>0): {valid:,} ({valid/total*100:.2f}%)")
        logger.info(f"  - Null: {null_count:,}")
        logger.info(f"  - Zero: {zero_count:,}")
        logger.info(f"  - Negative: {negative_count:,}")
    
    def _check_yearly_breakdown(self):
        """年別内訳"""
        logger.info("\n=== Yearly Breakdown ===")
        
        valid_mask = self.df[self.odds_col].notna() & (self.df[self.odds_col] > 0)
        
        yearly = []
        for year in sorted(self.df['year'].unique()):
            year_df = self.df[self.df['year'] == year]
            year_valid = year_df[self.odds_col].notna() & (year_df[self.odds_col] > 0)
            
            coverage = year_valid.sum() / len(year_df) * 100 if len(year_df) > 0 else 0
            yearly.append({
                'year': year,
                'total': len(year_df),
                'valid': int(year_valid.sum()),
                'coverage': coverage
            })
            
            status = "✅" if coverage > 95 else "⚠️" if coverage > 50 else "❌"
            logger.info(f"  {year}: {coverage:.1f}% ({year_valid.sum():,}/{len(year_df):,}) {status}")
        
        self.diagnostics['yearly'] = yearly
    
    def _check_venue_breakdown(self):
        """場別内訳（上位10件）"""
        logger.info("\n=== Venue Breakdown (Top 10 worst) ===")
        
        if 'venue_code' not in self.df.columns:
            logger.info("  No venue_code available")
            return
        
        valid_mask = self.df[self.odds_col].notna() & (self.df[self.odds_col] > 0)
        
        venue_stats = []
        for venue in self.df['venue_code'].unique():
            venue_df = self.df[self.df['venue_code'] == venue]
            venue_valid = venue_df[self.odds_col].notna() & (venue_df[self.odds_col] > 0)
            
            if len(venue_df) > 0:
                venue_stats.append({
                    'venue': venue,
                    'total': len(venue_df),
                    'valid': int(venue_valid.sum()),
                    'coverage': venue_valid.sum() / len(venue_df) * 100
                })
        
        # Worst coverage first
        venue_stats.sort(key=lambda x: x['coverage'])
        
        for v in venue_stats[:10]:
            logger.info(f"  Venue {v['venue']}: {v['coverage']:.1f}% ({v['valid']:,}/{v['total']:,})")
        
        self.diagnostics['venue'] = venue_stats
    
    def _check_key_types(self):
        """キーの型チェック"""
        logger.info("\n=== Key Type Check ===")
        
        key_cols = ['race_id', 'horse_id', 'umaban']
        
        for col in key_cols:
            if col in self.df.columns:
                dtype = str(self.df[col].dtype)
                sample = self.df[col].dropna().head(3).tolist()
                logger.info(f"  {col}: dtype={dtype}, sample={sample}")
                
                # 文字列の場合、パディングをチェック
                if 'object' in dtype or 'str' in dtype:
                    str_lengths = self.df[col].astype(str).str.len().value_counts().head(3)
                    logger.info(f"    Length distribution: {dict(str_lengths)}")
    
    def _check_duplicates(self):
        """重複行チェック"""
        logger.info("\n=== Duplicate Check ===")
        
        key_cols = ['race_id']
        if 'horse_id' in self.df.columns:
            key_cols.append('horse_id')
        elif 'umaban' in self.df.columns:
            key_cols.append('umaban')
        
        if len(key_cols) > 1:
            dup_mask = self.df.duplicated(subset=key_cols, keep=False)
            dup_count = dup_mask.sum()
            
            if dup_count > 0:
                logger.warning(f"  ⚠️ Found {dup_count:,} duplicate rows on {key_cols}")
                # Sample duplicates
                dup_sample = self.df[dup_mask].head(10)[key_cols + [self.odds_col]]
                logger.info(f"  Sample duplicates:\n{dup_sample.to_string()}")
            else:
                logger.info(f"  ✅ No duplicates on {key_cols}")
            
            self.diagnostics['duplicates'] = {
                'keys': key_cols,
                'count': int(dup_count)
            }
    
    def _sample_missing_records(self):
        """欠損レコードのサンプル"""
        logger.info("\n=== Missing Odds Sample ===")
        
        missing_mask = self.df[self.odds_col].isna() | (self.df[self.odds_col] <= 0)
        
        if missing_mask.sum() > 0:
            sample_cols = ['race_id', 'year', self.odds_col]
            if 'horse_id' in self.df.columns:
                sample_cols.insert(1, 'horse_id')
            if 'umaban' in self.df.columns:
                sample_cols.insert(2, 'umaban')
            if 'venue_code' in self.df.columns:
                sample_cols.insert(3, 'venue_code')
            
            sample = self.df[missing_mask][sample_cols].head(10)
            logger.info(f"  Sample missing records:\n{sample.to_string()}")
            
            # Missing by year
            missing_by_year = self.df[missing_mask].groupby('year').size()
            logger.info(f"\n  Missing by year:\n{missing_by_year.to_string()}")
    
    def _check_odds_values(self):
        """オッズ値の分布"""
        logger.info("\n=== Odds Value Distribution ===")
        
        valid_mask = self.df[self.odds_col].notna() & (self.df[self.odds_col] > 0)
        valid_odds = self.df.loc[valid_mask, self.odds_col]
        
        if len(valid_odds) > 0:
            stats = valid_odds.describe()
            logger.info(f"  Stats:\n{stats.to_string()}")
            
            # 異常値チェック
            very_low = (valid_odds < 1.0).sum()
            very_high = (valid_odds > 1000).sum()
            if very_low > 0:
                logger.warning(f"  ⚠️ Odds < 1.0: {very_low:,}")
            if very_high > 0:
                logger.warning(f"  ⚠️ Odds > 1000: {very_high:,}")
    
    def generate_report(self, output_path: str):
        """診断レポート生成"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        basic = self.diagnostics.get('basic', {})
        
        report = f"""# Odds Coverage Diagnostic Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Data**: {self.data_path}

## 1. Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Row Coverage** | {basic.get('row_coverage', 0):.2f}% | {'✅' if basic.get('row_coverage', 0) > 95 else '⚠️'} |
| Total Rows | {basic.get('total_rows', 0):,} | |
| Valid Odds | {basic.get('valid_rows', 0):,} | |
| Null Values | {basic.get('null_count', 0):,} | |
| Zero Values | {basic.get('zero_count', 0):,} | |

## 2. Yearly Breakdown

| Year | Total | Valid | Coverage | Status |
|------|-------|-------|----------|--------|
"""
        for y in self.diagnostics.get('yearly', []):
            status = '✅' if y['coverage'] > 95 else '⚠️' if y['coverage'] > 50 else '❌'
            report += f"| {y['year']} | {y['total']:,} | {y['valid']:,} | {y['coverage']:.1f}% | {status} |\n"
        
        report += """
## 3. Venue Analysis (Worst 10)

| Venue | Total | Valid | Coverage |
|-------|-------|-------|----------|
"""
        for v in self.diagnostics.get('venue', [])[:10]:
            report += f"| {v['venue']} | {v['total']:,} | {v['valid']:,} | {v['coverage']:.1f}% |\n"
        
        dup = self.diagnostics.get('duplicates', {})
        report += f"""
## 4. Duplicate Check

- Keys: {dup.get('keys', 'N/A')}
- Duplicate Rows: {dup.get('count', 0):,}

## 5. Recommendations

"""
        if basic.get('row_coverage', 0) < 95:
            report += "- ⚠️ Coverage below 95%, investigate missing odds source\n"
        if basic.get('null_count', 0) > basic.get('total_rows', 1) * 0.1:
            report += "- ⚠️ High null rate, check JOIN keys or data source\n"
        if dup.get('count', 0) > 0:
            report += "- ⚠️ Duplicates found, may cause JOIN explosion\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Odds Coverage Issues")
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed/preprocessed_data_v11.parquet'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/phase5_odds_diagnostic.md'
    )
    
    args = parser.parse_args()
    
    diag = OddsCoverageDiagnostic(args.data_path)
    diag.load_data()
    diag.run_diagnostics()
    diag.generate_report(args.output)


if __name__ == "__main__":
    main()
