"""
Phase 7: Sanrentan (三連単) Strategy
三連単の確率・期待値ベース買い目選択

Usage (in container):
    docker compose exec app python src/backtest/sanrentan_strategy.py --period screening
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.period_guard import add_period_args, parse_period_args, filter_dataframe_by_period, PeriodConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SanrentanConfig:
    """三連単戦略設定"""
    top_n: int = 5  # 上位N頭からP(N,3)点
    ev_threshold: float = 0.0
    min_odds: float = 5.0
    max_odds: float = 1000.0
    max_bets_per_race: int = 20
    bet_amount: int = 100
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SanrentanStrategy:
    """三連単戦略"""
    
    def __init__(self, config: SanrentanConfig = None):
        self.config = config or SanrentanConfig()
    
    def select_bets(
        self,
        race_id: str,
        trio_probs: pd.DataFrame,
        odds_data: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """レースの買い目を選択"""
        race_trios = trio_probs[trio_probs['race_id'] == race_id].copy()
        
        if len(race_trios) == 0:
            return []
        
        # オッズがある場合はEV計算
        if odds_data is not None:
            race_odds = odds_data[odds_data['race_id'] == race_id]
            if len(race_odds) > 0:
                race_trios = self._merge_odds(race_trios, race_odds)
                race_trios['ev'] = race_trios['probability'] * race_trios['odds'] - 1
            else:
                race_trios['ev'] = race_trios['probability'] - 0.275
                race_trios['odds'] = np.nan
        else:
            race_trios['ev'] = race_trios['probability'] - 0.275
            race_trios['odds'] = np.nan
        
        # フィルタ
        mask = race_trios['ev'] > self.config.ev_threshold
        if 'odds' in race_trios.columns and race_trios['odds'].notna().any():
            mask &= race_trios['odds'] >= self.config.min_odds
            mask &= race_trios['odds'] <= self.config.max_odds
        
        candidates = race_trios[mask].copy()
        
        if len(candidates) == 0:
            return []
        
        candidates = candidates.sort_values('ev', ascending=False).head(self.config.max_bets_per_race)
        
        bets = []
        for _, row in candidates.iterrows():
            bet = {
                'race_id': race_id,
                'ticket_type': 'sanrentan',
                'first': row['first'],
                'second': row['second'],
                'third': row['third'],
                'probability': row['probability'],
                'ev': row['ev'],
                'odds': row.get('odds', np.nan),
                'bet_amount': self.config.bet_amount
            }
            bets.append(bet)
        
        return bets
    
    def _merge_odds(self, trios: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """オッズデータをマージ（順あり）"""
        trios['trio_key'] = trios.apply(
            lambda r: (r['first'], r['second'], r['third']), axis=1
        )
        
        if all(c in odds.columns for c in ['first', 'second', 'third']):
            odds = odds.copy()
            odds['trio_key'] = odds.apply(
                lambda r: (r['first'], r['second'], r['third']), axis=1
            )
            merged = trios.merge(odds[['trio_key', 'odds']], on='trio_key', how='left')
        else:
            merged = trios.copy()
            merged['odds'] = np.nan
        
        return merged


class SanrentanBacktest:
    """三連単バックテスト"""
    
    def __init__(
        self,
        trio_probs_path: str,
        results_path: str,
        odds_path: Optional[str] = None,
        config: SanrentanConfig = None
    ):
        self.trio_probs_path = trio_probs_path
        self.results_path = results_path
        self.odds_path = odds_path
        self.config = config or SanrentanConfig()
        self.strategy = SanrentanStrategy(config)
        
        self.trio_probs = None
        self.results = None
        self.odds = None
    
    def load_data(self, period: PeriodConfig):
        """データロード"""
        logger.info("Loading trio probability data (ordered)...")
        self.trio_probs = pd.read_parquet(self.trio_probs_path)
        
        self.trio_probs['year'] = self.trio_probs['race_id'].astype(str).str[:4].astype(int)
        self.trio_probs = self.trio_probs[
            (self.trio_probs['year'] >= period.start_year) &
            (self.trio_probs['year'] <= period.end_year)
        ]
        
        logger.info(f"Trio probs (ordered): {len(self.trio_probs):,} rows")
        
        logger.info("Loading race results...")
        if os.path.exists(self.results_path):
            self.results = pd.read_parquet(self.results_path)
            self.results['year'] = self.results['race_id'].astype(str).str[:4].astype(int)
            self.results = self.results[
                (self.results['year'] >= period.start_year) &
                (self.results['year'] <= period.end_year)
            ]
        
        if self.odds_path and os.path.exists(self.odds_path):
            logger.info("Loading odds data...")
            self.odds = pd.read_parquet(self.odds_path)
    
    def evaluate_bets(self, bets: List[Dict], results: pd.DataFrame) -> List[Dict]:
        """買い目を評価（順あり的中判定）"""
        evaluated = []
        
        for bet in bets:
            race_id = bet['race_id']
            first, second, third = bet['first'], bet['second'], bet['third']
            
            race_result = results[results['race_id'] == race_id]
            
            if len(race_result) == 0:
                bet['hit'] = 0
                bet['payout'] = 0
                evaluated.append(bet)
                continue
            
            # 1-2-3着を取得
            top3 = race_result[race_result['rank'] <= 3].sort_values('rank')
            if len(top3) < 3:
                bet['hit'] = 0
                bet['payout'] = 0
                evaluated.append(bet)
                continue
            
            if 'umaban' in top3.columns:
                actual = top3['umaban'].tolist()[:3]
            else:
                actual = top3.index.tolist()[:3]
            
            # 順あり的中判定
            bet_order = [first, second, third]
            if bet_order == actual:
                bet['hit'] = 1
                if not np.isnan(bet.get('odds', np.nan)):
                    bet['payout'] = bet['odds'] * bet['bet_amount']
                else:
                    bet['payout'] = 0
            else:
                bet['hit'] = 0
                bet['payout'] = 0
            
            evaluated.append(bet)
        
        return evaluated
    
    def run_backtest(self) -> pd.DataFrame:
        """バックテスト実行"""
        race_ids = self.trio_probs['race_id'].unique()
        logger.info(f"Running backtest on {len(race_ids)} races...")
        
        all_bets = []
        
        for race_id in race_ids:
            bets = self.strategy.select_bets(race_id, self.trio_probs, self.odds)
            
            if bets and self.results is not None:
                bets = self.evaluate_bets(bets, self.results)
            
            all_bets.extend(bets)
        
        results_df = pd.DataFrame(all_bets)
        
        if len(results_df) > 0:
            results_df['year'] = results_df['race_id'].astype(str).str[:4].astype(int)
        
        return results_df
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """指標計算"""
        if len(results_df) == 0:
            return {'roi': 0, 'hit_rate': 0, 'total_bets': 0}
        
        total_bet = results_df['bet_amount'].sum()
        total_payout = results_df['payout'].sum()
        total_hits = results_df['hit'].sum()
        
        return {
            'total_bets': len(results_df),
            'total_bet_amount': total_bet,
            'total_payout': total_payout,
            'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
            'profit': total_payout - total_bet,
            'hit_rate': (total_hits / len(results_df) * 100) if len(results_df) > 0 else 0,
            'hits': int(total_hits),
            'bets_per_race': len(results_df) / results_df['race_id'].nunique() if results_df['race_id'].nunique() > 0 else 0
        }
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        metrics: Dict,
        output_path: str,
        period: PeriodConfig
    ):
        """レポート生成"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = f"""# Phase 7: Sanrentan (三連単) Strategy Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {period.start_year}-{period.end_year}

## Configuration

```yaml
"""
        for k, v in self.config.to_dict().items():
            report += f"{k}: {v}\n"
        
        report += f"""```

## Overall Metrics

| Metric | Value |
|--------|-------|
| Total Bets | {metrics['total_bets']:,} |
| Total Bet Amount | ¥{metrics['total_bet_amount']:,.0f} |
| Total Payout | ¥{metrics['total_payout']:,.0f} |
| **ROI** | **{metrics['roi']:.2f}%** |
| Profit | ¥{metrics['profit']:,.0f} |
| Hit Rate | {metrics['hit_rate']:.2f}% |
| Hits | {metrics['hits']:,} |
| Bets/Race | {metrics['bets_per_race']:.2f} |

"""
        
        if len(results_df) > 0 and 'year' in results_df.columns:
            report += "## Yearly Breakdown\n\n"
            report += "| Year | Bets | Bet Amount | Payout | ROI | Hits |\n"
            report += "|------|------|------------|--------|-----|------|\n"
            
            for year in sorted(results_df['year'].unique()):
                year_df = results_df[results_df['year'] == year]
                year_bet = year_df['bet_amount'].sum()
                year_payout = year_df['payout'].sum()
                year_roi = (year_payout / year_bet * 100) if year_bet > 0 else 0
                year_hits = year_df['hit'].sum()
                report += f"| {year} | {len(year_df):,} | ¥{year_bet:,.0f} | ¥{year_payout:,.0f} | {year_roi:.2f}% | {year_hits:,} |\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Sanrentan Strategy Backtest")
    add_period_args(parser)
    parser.add_argument('--trio_probs', type=str, default='data/probabilities/trio_ordered_probs.parquet')
    parser.add_argument('--results', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--odds', type=str, default=None)
    parser.add_argument('--top_n', type=int, default=5)
    parser.add_argument('--ev_threshold', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default='reports')
    
    args = parser.parse_args()
    
    try:
        period = parse_period_args(args)
    except ValueError as e:
        logger.error(f"Period error: {e}")
        sys.exit(1)
    
    config = SanrentanConfig(
        top_n=args.top_n,
        ev_threshold=args.ev_threshold
    )
    
    backtest = SanrentanBacktest(
        trio_probs_path=args.trio_probs,
        results_path=args.results,
        odds_path=args.odds,
        config=config
    )
    
    backtest.load_data(period)
    results_df = backtest.run_backtest()
    metrics = backtest.calculate_metrics(results_df)
    
    logger.info(f"Results: {metrics}")
    
    backtest.generate_report(
        results_df,
        metrics,
        os.path.join(args.output_dir, 'phase7_sanrentan.md'),
        period
    )
    
    if len(results_df) > 0:
        os.makedirs('data/backtest', exist_ok=True)
        results_df.to_parquet('data/backtest/sanrentan_results.parquet', index=False)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
