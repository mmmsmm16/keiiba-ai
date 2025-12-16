"""
Phase 8: Holdout 2025 Backtest
2025年ホールドアウトデータでのバックテスト

Usage (in container):
    docker compose exec app python src/backtest/holdout_2025_backtest.py
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations, permutations
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.payout_loader import PayoutLoader, format_combination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_box_backtest(
    df: pd.DataFrame,
    payout_map: Dict,
    ticket_type: str,
    top_n: int = 5
) -> Dict:
    """BOX買いバックテスト"""
    results = []
    
    horse_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in df.columns else 'score'
    
    # score列がなければraw_scoreを使用
    if score_col not in df.columns or df[score_col].isna().all():
        # オッズからp_marketを計算して代用
        if 'odds' in df.columns:
            df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
            df['p_market'] = df.groupby('race_id')['raw_prob'].transform(lambda x: x / x.sum())
            score_col = 'p_market'
        else:
            logger.error("No score/probability column available")
            return {'roi': 0}
    
    for race_id in df['race_id'].unique():
        if race_id not in payout_map:
            continue
        
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df[race_df[score_col].notna()]
        
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        if ticket_type == 'umaren':
            tickets = list(combinations(horse_numbers, 2))
            payouts = payout_map[race_id].get('umaren', {})
            ordered = False
        elif ticket_type == 'sanrenpuku':
            tickets = list(combinations(horse_numbers, 3))
            payouts = payout_map[race_id].get('sanrenpuku', {})
            ordered = False
        elif ticket_type == 'sanrentan':
            tickets = list(permutations(horse_numbers, 3))
            payouts = payout_map[race_id].get('sanrentan', {})
            ordered = True
        else:
            continue
        
        bet_amount = len(tickets) * 100
        payout = 0
        hit = 0
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=ordered)
            if comb_str in payouts:
                payout = payouts[comb_str]
                hit = 1
                break
        
        results.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit
        })
    
    if not results:
        return {'roi': 0}
    
    res_df = pd.DataFrame(results)
    total_bet = res_df['bet_amount'].sum()
    total_payout = res_df['payout'].sum()
    
    return {
        'ticket_type': ticket_type,
        'top_n': top_n,
        'races': len(res_df),
        'total_bet': total_bet,
        'total_payout': total_payout,
        'profit': total_payout - total_bet,
        'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
        'hit_rate': (res_df['hit'].sum() / len(res_df) * 100) if len(res_df) > 0 else 0,
        'hits': int(res_df['hit'].sum())
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Holdout 2025 Backtest")
    parser.add_argument('--data', type=str, default='data/processed/preprocessed_data_v11.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    
    args = parser.parse_args()
    
    # Load 2025 data
    logger.info(f"Loading 2025 data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df = df[df['year'] == 2025].copy()
    logger.info(f"2025 data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load payout data
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([2025])
    logger.info(f"2025 payout data: {len(payout_map):,} races")
    
    # Run backtests using market probability (no model predictions for 2025)
    all_results = []
    
    # Test different ticket types and top_n
    configs = [
        ('umaren', 3), ('umaren', 4), ('umaren', 5),
        ('sanrenpuku', 4), ('sanrenpuku', 5), ('sanrenpuku', 6),
        ('sanrentan', 4), ('sanrentan', 5), ('sanrentan', 6),
    ]
    
    for ticket_type, top_n in configs:
        result = run_box_backtest(df, payout_map, ticket_type, top_n)
        all_results.append(result)
        logger.info(f"{ticket_type} BOX{top_n}: ROI={result['roi']:.2f}%, Hit={result['hit_rate']:.2f}%")
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'phase8_holdout_2025.md')
    
    report = f"""# Phase 8: Holdout 2025 Backtest Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: 2025 (Holdout)

## Results (Market Probability Based)

| Ticket | TopN | Races | Hits | Hit Rate | Total Bet | Payout | Profit | **ROI** |
|--------|------|-------|------|----------|-----------|--------|--------|---------|
"""
    
    for r in all_results:
        if r.get('races', 0) > 0:
            report += f"| {r['ticket_type']} | {r['top_n']} | {r['races']:,} | {r['hits']:,} | {r['hit_rate']:.1f}% | ¥{r['total_bet']:,.0f} | ¥{r['total_payout']:,.0f} | ¥{r['profit']:,.0f} | **{r['roi']:.1f}%** |\n"
    
    # Comparison with 2024
    report += """
## Comparison: 2024 WF vs 2025 Holdout

| Ticket | TopN | 2024 ROI | 2025 ROI | Delta |
|--------|------|----------|----------|-------|
| umaren | 3 | 83.1% | See above | TBD |
| sanrentan | 5 | 64.1% | See above | TBD |

## Notes

- 2025データはホールドアウト期間（学習に未使用）
- 市場確率（p_market）ベースでの評価
- モデル予測を使用した評価は別途実施予定

"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
