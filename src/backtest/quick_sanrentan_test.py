"""
Phase 7: Quick Sanrentan Backtest with Payout Data
三連単バックテスト（払戻データ使用）

Usage (in container):
    docker compose exec app python src/backtest/quick_sanrentan_test.py --year 2024
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.payout_loader import PayoutLoader, format_combination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_backtest(
    predictions_df: pd.DataFrame,
    payout_map: Dict,
    top_n: int = 5
) -> pd.DataFrame:
    """
    三連単バックテスト
    
    Args:
        predictions_df: 予測データ（race_id, horse_number/umaban, prob/score）
        payout_map: 払戻マップ
        top_n: 上位N頭からのBOX買い
    
    Returns:
        レース毎の結果DataFrame
    """
    results = []
    
    # Try to identify horse number column
    if 'umaban' in predictions_df.columns:
        horse_col = 'umaban'
    elif 'horse_number' in predictions_df.columns:
        horse_col = 'horse_number'
    else:
        raise ValueError("No horse number column found (umaban or horse_number)")
    
    # Score column
    if 'prob' in predictions_df.columns:
        score_col = 'prob'
    elif 'score' in predictions_df.columns:
        score_col = 'score'
    else:
        raise ValueError("No score column found (prob or score)")
    
    race_ids = predictions_df['race_id'].unique()
    logger.info(f"Running backtest on {len(race_ids)} races...")
    
    for race_id in race_ids:
        if race_id not in payout_map:
            continue
        
        race_df = predictions_df[predictions_df['race_id'] == race_id].copy()
        
        if len(race_df) < top_n:
            continue
        
        # Get top N by score
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        # Generate all permutations (BOX)
        from itertools import permutations
        tickets = list(permutations(horse_numbers, 3))
        
        bet_amount = len(tickets) * 100  # 100円/点
        payout = 0
        hit = 0
        hit_comb = None
        
        # Check each ticket
        sanrentan_payouts = payout_map[race_id].get('sanrentan', {})
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=True)
            if comb_str in sanrentan_payouts:
                payout = sanrentan_payouts[comb_str]  # 100円あたりの払戻
                hit = 1
                hit_comb = t
                break  # 三連単は1つしか当たらない
        
        results.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit,
            'hit_combination': hit_comb,
            'roi': (payout / bet_amount * 100) if bet_amount > 0 else 0
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Quick Sanrentan Backtest")
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--top_n', type=int, default=5, help='Top N horses for BOX')
    parser.add_argument('--predictions', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    
    args = parser.parse_args()
    
    # Load predictions
    logger.info(f"Loading predictions from {args.predictions}...")
    df = pd.read_parquet(args.predictions)
    
    # Add year column and filter
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df = df[df['year'] == args.year]
    df = df[df['prob'].notna()]  # Only rows with predictions
    
    logger.info(f"Filtered to {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load payout data
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([args.year])
    logger.info(f"Loaded {len(payout_map):,} races with payout data")
    
    # Run backtest
    results = run_backtest(df, payout_map, top_n=args.top_n)
    
    if results.empty:
        logger.warning("No results generated")
        return
    
    # Calculate metrics
    total_bet = results['bet_amount'].sum()
    total_payout = results['payout'].sum()
    total_hits = results['hit'].sum()
    
    roi = (total_payout / total_bet * 100) if total_bet > 0 else 0
    hit_rate = (total_hits / len(results) * 100) if len(results) > 0 else 0
    
    logger.info("=" * 60)
    logger.info(f"三連単 BOX{args.top_n} Backtest Results ({args.year})")
    logger.info("=" * 60)
    logger.info(f"Races: {len(results):,}")
    logger.info(f"Total Tickets: {results['tickets'].sum():,}")
    logger.info(f"Total Bet: ¥{total_bet:,.0f}")
    logger.info(f"Total Payout: ¥{total_payout:,.0f}")
    logger.info(f"Profit: ¥{total_payout - total_bet:,.0f}")
    logger.info(f"ROI: {roi:.2f}%")
    logger.info(f"Hit Rate: {hit_rate:.2f}% ({total_hits:,} hits)")
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'phase7_sanrentan_backtest.md')
    
    report = f"""# Phase 7: Sanrentan (三連単) Backtest Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {args.year}

## Configuration

- **Top N (BOX)**: {args.top_n}
- **Tickets/Race**: {args.top_n}P3 = {args.top_n * (args.top_n - 1) * (args.top_n - 2)} 点

## Results

| Metric | Value |
|--------|-------|
| Races | {len(results):,} |
| Total Tickets | {results['tickets'].sum():,} |
| Total Bet | ¥{total_bet:,.0f} |
| Total Payout | ¥{total_payout:,.0f} |
| Profit | ¥{total_payout - total_bet:,.0f} |
| **ROI** | **{roi:.2f}%** |
| Hit Rate | {hit_rate:.2f}% |
| Hits | {total_hits:,} |

"""
    
    # Top payouts
    top_payouts = results[results['hit'] == 1].nlargest(10, 'payout')
    if len(top_payouts) > 0:
        report += "## Top 10 Payouts\n\n"
        report += "| Race ID | Payout | Combination |\n"
        report += "|---------|--------|-------------|\n"
        for _, row in top_payouts.iterrows():
            report += f"| {row['race_id']} | ¥{row['payout']:,.0f} | {row['hit_combination']} |\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
