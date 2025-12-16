"""
Phase 7: Multi-Ticket Backtest
馬連/三連複/三連単のバックテスト（払戻データ使用）

Usage (in container):
    docker compose exec app python src/backtest/multi_ticket_backtest.py --year 2024
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


def run_umaren_backtest(
    predictions_df: pd.DataFrame,
    payout_map: Dict,
    top_n: int = 5
) -> Dict:
    """馬連BOX バックテスト"""
    results = []
    
    horse_col = 'umaban' if 'umaban' in predictions_df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in predictions_df.columns else 'score'
    
    for race_id in predictions_df['race_id'].unique():
        if race_id not in payout_map:
            continue
        
        race_df = predictions_df[predictions_df['race_id'] == race_id].copy()
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        tickets = list(combinations(horse_numbers, 2))
        bet_amount = len(tickets) * 100
        payout = 0
        hit = 0
        hit_comb = None
        
        umaren_payouts = payout_map[race_id].get('umaren', {})
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=False)
            if comb_str in umaren_payouts:
                payout = umaren_payouts[comb_str]
                hit = 1
                hit_comb = t
                break
        
        results.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit
        })
    
    df = pd.DataFrame(results)
    if df.empty:
        return {'roi': 0, 'hit_rate': 0, 'total_bet': 0, 'total_payout': 0, 'races': 0}
    
    total_bet = df['bet_amount'].sum()
    total_payout = df['payout'].sum()
    
    return {
        'ticket_type': 'umaren',
        'top_n': top_n,
        'races': len(df),
        'total_tickets': df['tickets'].sum(),
        'total_bet': total_bet,
        'total_payout': total_payout,
        'profit': total_payout - total_bet,
        'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
        'hit_rate': (df['hit'].sum() / len(df) * 100) if len(df) > 0 else 0,
        'hits': int(df['hit'].sum())
    }


def run_sanrenpuku_backtest(
    predictions_df: pd.DataFrame,
    payout_map: Dict,
    top_n: int = 6
) -> Dict:
    """三連複BOX バックテスト"""
    results = []
    
    horse_col = 'umaban' if 'umaban' in predictions_df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in predictions_df.columns else 'score'
    
    for race_id in predictions_df['race_id'].unique():
        if race_id not in payout_map:
            continue
        
        race_df = predictions_df[predictions_df['race_id'] == race_id].copy()
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        tickets = list(combinations(horse_numbers, 3))
        bet_amount = len(tickets) * 100
        payout = 0
        hit = 0
        
        sanrenpuku_payouts = payout_map[race_id].get('sanrenpuku', {})
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=False)
            if comb_str in sanrenpuku_payouts:
                payout = sanrenpuku_payouts[comb_str]
                hit = 1
                break
        
        results.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit
        })
    
    df = pd.DataFrame(results)
    if df.empty:
        return {'roi': 0, 'hit_rate': 0, 'total_bet': 0, 'total_payout': 0, 'races': 0}
    
    total_bet = df['bet_amount'].sum()
    total_payout = df['payout'].sum()
    
    return {
        'ticket_type': 'sanrenpuku',
        'top_n': top_n,
        'races': len(df),
        'total_tickets': df['tickets'].sum(),
        'total_bet': total_bet,
        'total_payout': total_payout,
        'profit': total_payout - total_bet,
        'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
        'hit_rate': (df['hit'].sum() / len(df) * 100) if len(df) > 0 else 0,
        'hits': int(df['hit'].sum())
    }


def run_sanrentan_backtest(
    predictions_df: pd.DataFrame,
    payout_map: Dict,
    top_n: int = 5
) -> Dict:
    """三連単BOX バックテスト"""
    results = []
    
    horse_col = 'umaban' if 'umaban' in predictions_df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in predictions_df.columns else 'score'
    
    for race_id in predictions_df['race_id'].unique():
        if race_id not in payout_map:
            continue
        
        race_df = predictions_df[predictions_df['race_id'] == race_id].copy()
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        tickets = list(permutations(horse_numbers, 3))
        bet_amount = len(tickets) * 100
        payout = 0
        hit = 0
        
        sanrentan_payouts = payout_map[race_id].get('sanrentan', {})
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=True)
            if comb_str in sanrentan_payouts:
                payout = sanrentan_payouts[comb_str]
                hit = 1
                break
        
        results.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit
        })
    
    df = pd.DataFrame(results)
    if df.empty:
        return {'roi': 0, 'hit_rate': 0, 'total_bet': 0, 'total_payout': 0, 'races': 0}
    
    total_bet = df['bet_amount'].sum()
    total_payout = df['payout'].sum()
    
    return {
        'ticket_type': 'sanrentan',
        'top_n': top_n,
        'races': len(df),
        'total_tickets': df['tickets'].sum(),
        'total_bet': total_bet,
        'total_payout': total_payout,
        'profit': total_payout - total_bet,
        'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
        'hit_rate': (df['hit'].sum() / len(df) * 100) if len(df) > 0 else 0,
        'hits': int(df['hit'].sum())
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Ticket Backtest")
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--predictions', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    
    args = parser.parse_args()
    
    # Load predictions
    logger.info(f"Loading predictions from {args.predictions}...")
    df = pd.read_parquet(args.predictions)
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df = df[df['year'] == args.year]
    df = df[df['prob'].notna()]
    
    logger.info(f"Filtered to {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load payout data
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([args.year])
    logger.info(f"Loaded {len(payout_map):,} races with payout data")
    
    # Run backtests
    all_results = []
    
    # Umaren BOX3, BOX4, BOX5
    for top_n in [3, 4, 5]:
        result = run_umaren_backtest(df, payout_map, top_n=top_n)
        all_results.append(result)
        logger.info(f"馬連 BOX{top_n}: ROI={result['roi']:.2f}%, Hit={result['hit_rate']:.2f}%")
    
    # Sanrenpuku BOX4, BOX5, BOX6
    for top_n in [4, 5, 6]:
        result = run_sanrenpuku_backtest(df, payout_map, top_n=top_n)
        all_results.append(result)
        logger.info(f"三連複 BOX{top_n}: ROI={result['roi']:.2f}%, Hit={result['hit_rate']:.2f}%")
    
    # Sanrentan BOX4, BOX5, BOX6
    for top_n in [4, 5, 6]:
        result = run_sanrentan_backtest(df, payout_map, top_n=top_n)
        all_results.append(result)
        logger.info(f"三連単 BOX{top_n}: ROI={result['roi']:.2f}%, Hit={result['hit_rate']:.2f}%")
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'phase7_multi_ticket_backtest.md')
    
    report = f"""# Phase 7: Multi-Ticket Backtest Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {args.year}

## Summary

| 券種 | TopN | Tickets/Race | Races | Hits | Hit Rate | Total Bet | Total Payout | Profit | **ROI** |
|------|------|--------------|-------|------|----------|-----------|--------------|--------|---------|
"""
    
    for r in all_results:
        tpr = r['total_tickets'] // r['races'] if r['races'] > 0 else 0
        report += f"| {r['ticket_type']} | {r['top_n']} | {tpr} | {r['races']:,} | {r['hits']:,} | {r['hit_rate']:.1f}% | ¥{r['total_bet']:,.0f} | ¥{r['total_payout']:,.0f} | ¥{r['profit']:,.0f} | **{r['roi']:.1f}%** |\n"
    
    # Best result
    best = max(all_results, key=lambda x: x['roi'])
    report += f"""
## Best Strategy

**{best['ticket_type']} BOX{best['top_n']}**: ROI **{best['roi']:.2f}%**, Hit Rate {best['hit_rate']:.2f}%

## Notes

- 全てBOX買い（確率上位N頭から全組み合わせ）
- 控除率: 単勝/複勝25%, 馬連/ワイド25%, 三連系27.5%
- ROI 100%超には選別（EV/確率フィルタ）が必要

"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
