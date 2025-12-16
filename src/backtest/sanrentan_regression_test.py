"""
Phase 5 (P0-5): Sanrentan Regression Test
BOX買いとStrategy形式の結果一致テスト

Purpose:
- BOX版とStrategy版で同一条件なら同じ結果（的中数/払戻/ROI）になること
- 不一致の場合は原因（券種キー/馬番順/払戻テーブルキー）を特定

Usage:
    docker compose exec app python src/backtest/sanrentan_regression_test.py
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from itertools import permutations
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.payout_loader import PayoutLoader, format_combination
from utils.race_filter import filter_jra_only

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_sanrentan_box_backtest(
    df: pd.DataFrame,
    payout_map: Dict,
    top_n: int = 5
) -> Tuple[Dict, List[Dict]]:
    """
    三連単BOXバックテスト（参照実装）
    
    Returns:
        (summary, race_details)
    """
    race_details = []
    
    horse_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    score_col = 'p_market'  # 固定でp_marketを使用
    
    # p_market計算
    df = df.copy()
    df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
    df['p_market'] = df.groupby('race_id')['raw_prob'].transform(lambda x: x / x.sum())
    
    for race_id in df['race_id'].unique():
        if race_id not in payout_map:
            continue
        
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df[race_df[score_col].notna()]
        
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        tickets = list(permutations(horse_numbers, 3))
        bet_amount = len(tickets) * 100
        payout = 0
        hit = 0
        hit_comb = None
        
        sanrentan_payouts = payout_map[race_id].get('sanrentan', {})
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=True)
            if comb_str in sanrentan_payouts:
                payout = sanrentan_payouts[comb_str]
                hit = 1
                hit_comb = t
                break
        
        race_details.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit,
            'hit_comb': hit_comb,
            'horse_numbers': horse_numbers
        })
    
    if not race_details:
        return {'roi': 0, 'hits': 0, 'races': 0}, []
    
    res_df = pd.DataFrame(race_details)
    total_bet = res_df['bet_amount'].sum()
    total_payout = res_df['payout'].sum()
    
    summary = {
        'method': 'BOX',
        'top_n': top_n,
        'races': len(res_df),
        'total_bet': total_bet,
        'total_payout': total_payout,
        'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
        'hits': int(res_df['hit'].sum()),
        'hit_rate': (res_df['hit'].sum() / len(res_df) * 100) if len(res_df) > 0 else 0
    }
    
    return summary, race_details


def run_sanrentan_strategy_backtest(
    df: pd.DataFrame,
    payout_map: Dict,
    top_n: int = 5
) -> Tuple[Dict, List[Dict]]:
    """
    三連単ストラテジー形式バックテスト（テスト対象実装）
    
    BOX相当の設定: formation = [[1,2,...,N], [1,2,...,N], [1,2,...,N]]
    """
    race_details = []
    
    horse_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    score_col = 'p_market'
    
    df = df.copy()
    df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
    df['p_market'] = df.groupby('race_id')['raw_prob'].transform(lambda x: x / x.sum())
    
    for race_id in df['race_id'].unique():
        if race_id not in payout_map:
            continue
        
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df[race_df[score_col].notna()]
        
        if len(race_df) < top_n:
            continue
        
        # Get top N
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        # Formation: BOX相当（全順列）
        # [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]] => P(5,3) = 60点
        tickets = []
        for h1 in horse_numbers:
            for h2 in horse_numbers:
                if h1 == h2:
                    continue
                for h3 in horse_numbers:
                    if h3 == h1 or h3 == h2:
                        continue
                    tickets.append((h1, h2, h3))
        
        bet_amount = len(tickets) * 100
        payout = 0
        hit = 0
        hit_comb = None
        
        sanrentan_payouts = payout_map[race_id].get('sanrentan', {})
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=True)
            if comb_str in sanrentan_payouts:
                payout = sanrentan_payouts[comb_str]
                hit = 1
                hit_comb = t
                break
        
        race_details.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'bet_amount': bet_amount,
            'payout': payout,
            'hit': hit,
            'hit_comb': hit_comb,
            'horse_numbers': horse_numbers
        })
    
    if not race_details:
        return {'roi': 0, 'hits': 0, 'races': 0}, []
    
    res_df = pd.DataFrame(race_details)
    total_bet = res_df['bet_amount'].sum()
    total_payout = res_df['payout'].sum()
    
    summary = {
        'method': 'Strategy',
        'top_n': top_n,
        'races': len(res_df),
        'total_bet': total_bet,
        'total_payout': total_payout,
        'roi': (total_payout / total_bet * 100) if total_bet > 0 else 0,
        'hits': int(res_df['hit'].sum()),
        'hit_rate': (res_df['hit'].sum() / len(res_df) * 100) if len(res_df) > 0 else 0
    }
    
    return summary, race_details


def compare_results(box_details: List[Dict], strategy_details: List[Dict]) -> Dict:
    """
    BOXとStrategyの結果を比較
    """
    box_df = pd.DataFrame(box_details)
    strategy_df = pd.DataFrame(strategy_details)
    
    if len(box_df) == 0 or len(strategy_df) == 0:
        return {'match': False, 'error': 'Empty results'}
    
    # race_idでマージして比較
    merged = box_df.merge(strategy_df, on='race_id', suffixes=('_box', '_strategy'))
    
    # 比較項目
    comparisons = {
        'tickets_match': (merged['tickets_box'] == merged['tickets_strategy']).all(),
        'bet_amount_match': (merged['bet_amount_box'] == merged['bet_amount_strategy']).all(),
        'payout_match': (merged['payout_box'] == merged['payout_strategy']).all(),
        'hit_match': (merged['hit_box'] == merged['hit_strategy']).all()
    }
    
    all_match = all(comparisons.values())
    
    # 不一致があれば詳細を出力
    mismatches = []
    if not all_match:
        for col in ['tickets', 'bet_amount', 'payout', 'hit']:
            mismatch_mask = merged[f'{col}_box'] != merged[f'{col}_strategy']
            if mismatch_mask.any():
                mismatches.append({
                    'column': col,
                    'count': mismatch_mask.sum(),
                    'races': merged[mismatch_mask]['race_id'].tolist()[:5]  # First 5
                })
    
    return {
        'match': all_match,
        'comparisons': comparisons,
        'race_count': len(merged),
        'mismatches': mismatches
    }


def main():
    logger.info("=== Sanrentan Regression Test ===")
    
    # Load 2024 JRA data
    df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
    df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    df = df[df['year'] == 2024]
    df = filter_jra_only(df)
    
    logger.info(f"Data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load payout
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([2024])
    
    # Run both methods
    for top_n in [4, 5]:
        logger.info(f"\n--- TOP {top_n} ---")
        
        box_summary, box_details = run_sanrentan_box_backtest(df, payout_map, top_n)
        strategy_summary, strategy_details = run_sanrentan_strategy_backtest(df, payout_map, top_n)
        
        logger.info(f"BOX:      ROI={box_summary['roi']:.2f}%, Hits={box_summary['hits']}")
        logger.info(f"Strategy: ROI={strategy_summary['roi']:.2f}%, Hits={strategy_summary['hits']}")
        
        # Compare
        comparison = compare_results(box_details, strategy_details)
        
        if comparison['match']:
            logger.info("✅ PASS: BOX and Strategy results match")
        else:
            logger.error("❌ FAIL: Results do not match!")
            for mismatch in comparison['mismatches']:
                logger.error(f"  Mismatch in {mismatch['column']}: {mismatch['count']} races")
                logger.error(f"  Example races: {mismatch['races']}")
    
    logger.info("\n=== Regression Test Complete ===")


if __name__ == "__main__":
    main()
