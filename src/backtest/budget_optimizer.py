"""
Phase 7: Budget Optimizer (Multi-unit Allocation)
複数口対応の予算配分最適化

Usage (in container):
    docker compose exec app python src/backtest/budget_optimizer.py --input data/backtest/sanrentan_results.parquet
"""

import sys
import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """予算配分設定"""
    race_budget: int = 3000  # 1レースあたり上限（円）
    min_unit: int = 100  # 最小購入単位（JRA規定）
    max_per_bet: int = 1000  # 1点あたり上限
    kelly_fraction: float = 0.1
    ev_threshold: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BudgetOptimizer:
    """
    複数口対応の予算配分最適化
    
    手法: 連続配分 → 100円丸め (MVP推奨)
    """
    
    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig()
    
    def allocate_kelly(
        self,
        bets: pd.DataFrame,
        bankroll: float = None
    ) -> pd.DataFrame:
        """
        Kelly基準で配分し100円単位に丸める
        
        Kelly: f* = (p * odds - 1) / (odds - 1) = ev / (odds - 1)
        """
        result = bets.copy()
        
        if bankroll is None:
            bankroll = self.config.race_budget
        
        # EV計算済みの場合
        if 'ev' not in result.columns:
            if 'probability' in result.columns and 'odds' in result.columns:
                result['ev'] = result['probability'] * result['odds'] - 1
            else:
                result['ev'] = 0
        
        # EVフィルタ
        result = result[result['ev'] > self.config.ev_threshold].copy()
        
        if len(result) == 0:
            return pd.DataFrame()
        
        # Kelly fraction計算
        def kelly_fraction(row):
            if pd.isna(row['odds']) or row['odds'] <= 1:
                return 0
            f = row['ev'] / (row['odds'] - 1)
            return max(0, f * self.config.kelly_fraction)
        
        result['kelly_f'] = result.apply(kelly_fraction, axis=1)
        
        # 合計が1を超える場合は正規化
        total_f = result['kelly_f'].sum()
        if total_f > 1:
            result['kelly_f'] = result['kelly_f'] / total_f
        
        # 配分金額（連続）
        result['raw_amount'] = result['kelly_f'] * bankroll
        
        # 100円単位に丸め
        result['bet_amount'] = (result['raw_amount'] / self.config.min_unit).apply(np.floor) * self.config.min_unit
        
        # 上限適用
        result['bet_amount'] = result['bet_amount'].clip(0, self.config.max_per_bet)
        
        # 0円は除外
        result = result[result['bet_amount'] > 0].copy()
        
        # 合計が予算上限を超える場合は調整
        while result['bet_amount'].sum() > self.config.race_budget and len(result) > 0:
            # 最小EVのベットを減額
            min_ev_idx = result['ev'].idxmin()
            current = result.loc[min_ev_idx, 'bet_amount']
            result.loc[min_ev_idx, 'bet_amount'] = max(0, current - self.config.min_unit)
            if result.loc[min_ev_idx, 'bet_amount'] == 0:
                result = result.drop(min_ev_idx)
        
        return result
    
    def allocate_greedy(
        self,
        bets: pd.DataFrame,
        bankroll: float = None
    ) -> pd.DataFrame:
        """Greedy配分: EV順に均等配分"""
        result = bets.copy()
        
        if bankroll is None:
            bankroll = self.config.race_budget
        
        if 'ev' not in result.columns:
            result['ev'] = 0
        
        # EVフィルタ & ソート
        result = result[result['ev'] > self.config.ev_threshold].copy()
        result = result.sort_values('ev', ascending=False)
        
        if len(result) == 0:
            return pd.DataFrame()
        
        # 予算内で均等配分
        n_bets = len(result)
        per_bet = int(bankroll / n_bets / self.config.min_unit) * self.config.min_unit
        per_bet = min(per_bet, self.config.max_per_bet)
        
        if per_bet < self.config.min_unit:
            # 予算不足、上位に絞る
            max_bets = int(bankroll / self.config.min_unit)
            result = result.head(max_bets)
            per_bet = self.config.min_unit
        
        result['bet_amount'] = per_bet
        
        return result


class BetRecommendationGenerator:
    """購入推奨リスト生成"""
    
    def __init__(self, optimizer: BudgetOptimizer = None):
        self.optimizer = optimizer or BudgetOptimizer()
    
    def process_races(
        self,
        bets_df: pd.DataFrame,
        method: str = 'kelly'
    ) -> pd.DataFrame:
        """全レースを処理"""
        results = []
        
        race_ids = bets_df['race_id'].unique()
        logger.info(f"Processing {len(race_ids)} races with {method} allocation...")
        
        for race_id in race_ids:
            race_bets = bets_df[bets_df['race_id'] == race_id].copy()
            
            if method == 'kelly':
                allocated = self.optimizer.allocate_kelly(race_bets)
            else:
                allocated = self.optimizer.allocate_greedy(race_bets)
            
            if len(allocated) > 0:
                results.append(allocated)
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def to_json_format(self, bets_df: pd.DataFrame) -> List[Dict]:
        """JRA購入可能形式のJSONに変換"""
        records = []
        
        for _, row in bets_df.iterrows():
            record = {
                'race_id': row['race_id'],
                'ticket_type': row.get('ticket_type', 'unknown'),
                'bet_amount': int(row['bet_amount'])
            }
            
            # 券種別の組み合わせ
            if 'first' in row:
                # 三連単
                record['combination'] = [row['first'], row['second'], row['third']]
                record['ordered'] = True
            elif 'horse_1' in row and 'horse_3' in row:
                # 三連複
                record['combination'] = sorted([row['horse_1'], row['horse_2'], row['horse_3']])
                record['ordered'] = False
            elif 'horse_1' in row:
                # 馬連
                record['combination'] = sorted([row['horse_1'], row['horse_2']])
                record['ordered'] = False
            
            if 'ev' in row:
                record['ev'] = round(float(row['ev']), 4)
            if 'probability' in row:
                record['probability'] = round(float(row['probability']), 6)
            
            records.append(record)
        
        return records
    
    def to_csv_format(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """CSV出力用に整形"""
        output_cols = ['race_id', 'ticket_type', 'bet_amount']
        
        # 組み合わせカラム追加
        if 'first' in bets_df.columns:
            output_cols.extend(['first', 'second', 'third'])
        elif 'horse_1' in bets_df.columns:
            output_cols.extend(['horse_1', 'horse_2'])
            if 'horse_3' in bets_df.columns:
                output_cols.append('horse_3')
        
        if 'ev' in bets_df.columns:
            output_cols.append('ev')
        if 'probability' in bets_df.columns:
            output_cols.append('probability')
        
        available = [c for c in output_cols if c in bets_df.columns]
        return bets_df[available].copy()


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Budget Optimizer")
    parser.add_argument('--input', type=str, required=True, help='Input bets parquet file')
    parser.add_argument('--method', type=str, default='kelly', choices=['kelly', 'greedy'])
    parser.add_argument('--race_budget', type=int, default=3000)
    parser.add_argument('--max_per_bet', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    # Load bets
    logger.info(f"Loading bets from {args.input}...")
    bets_df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(bets_df):,} bets")
    
    # Configure
    config = BudgetConfig(
        race_budget=args.race_budget,
        max_per_bet=args.max_per_bet
    )
    
    optimizer = BudgetOptimizer(config)
    generator = BetRecommendationGenerator(optimizer)
    
    # Process
    allocated = generator.process_races(bets_df, method=args.method)
    
    if len(allocated) == 0:
        logger.warning("No bets allocated")
        return
    
    logger.info(f"Allocated: {len(allocated):,} bets, ¥{allocated['bet_amount'].sum():,.0f} total")
    
    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # JSON
    json_data = generator.to_json_format(allocated)
    json_path = os.path.join(args.output_dir, 'bet_recommendations.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON saved to {json_path}")
    
    # CSV
    csv_data = generator.to_csv_format(allocated)
    csv_path = os.path.join(args.output_dir, 'bet_recommendations.csv')
    csv_data.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to {csv_path}")
    
    # Summary
    logger.info("\n=== Allocation Summary ===")
    logger.info(f"Total bets: {len(allocated):,}")
    logger.info(f"Total amount: ¥{allocated['bet_amount'].sum():,.0f}")
    logger.info(f"Avg bet/race: {len(allocated) / allocated['race_id'].nunique():.2f}")
    logger.info(f"Avg amount/race: ¥{allocated['bet_amount'].sum() / allocated['race_id'].nunique():,.0f}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
