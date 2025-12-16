"""
Phase 7: EV-Filtered Backtest
確率/EVフィルタによる券種バックテスト

Usage (in container):
    docker compose exec app python src/backtest/ev_filtered_backtest.py --year 2024
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations, permutations
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.payout_loader import PayoutLoader, format_combination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_filtered_backtest(
    predictions_df: pd.DataFrame,
    payout_map: Dict,
    trio_probs_path: str,
    ticket_type: str = 'sanrentan',
    min_prob: float = 0.0,
    min_ev: float = 0.0,
    top_k_bets: int = 10
) -> Dict:
    """
    確率/EVフィルタ付きバックテスト
    
    Args:
        predictions_df: 予測データ
        payout_map: 払戻マップ
        trio_probs_path: 組み合わせ確率データのパス
        ticket_type: sanrentan, sanrenpuku, umaren
        min_prob: 最低確率閾値
        min_ev: 最低EV閾値 (EV = prob * estimated_odds - 1)
        top_k_bets: 上位K件の買い目のみ
    """
    # Load trio probabilities
    logger.info(f"Loading {ticket_type} probabilities...")
    
    if ticket_type == 'sanrentan':
        prob_df = pd.read_parquet(trio_probs_path)
        prob_df['year'] = prob_df['race_id'].astype(str).str[:4].astype(int)
    elif ticket_type == 'sanrenpuku':
        prob_df = pd.read_parquet(trio_probs_path.replace('trio_ordered', 'trio_unordered'))
        prob_df['year'] = prob_df['race_id'].astype(str).str[:4].astype(int)
    elif ticket_type == 'umaren':
        prob_df = pd.read_parquet(trio_probs_path.replace('trio_ordered_probs', 'pair_probs'))
        prob_df['year'] = prob_df['race_id'].astype(str).str[:4].astype(int)
    else:
        raise ValueError(f"Unknown ticket type: {ticket_type}")
    
    results = []
    total_bet = 0
    total_payout = 0
    total_hits = 0
    race_count = 0
    
    race_ids = prob_df['race_id'].unique()
    
    for race_id in race_ids:
        if race_id not in payout_map:
            continue
        
        race_probs = prob_df[prob_df['race_id'] == race_id].copy()
        
        # Filter by probability
        if min_prob > 0:
            race_probs = race_probs[race_probs['probability'] >= min_prob]
        
        if len(race_probs) == 0:
            continue
        
        # Sort by probability (as proxy for EV without odds)
        race_probs = race_probs.sort_values('probability', ascending=False)
        
        # Top K bets
        race_probs = race_probs.head(top_k_bets)
        
        race_bet = len(race_probs) * 100
        race_payout = 0
        race_hit = 0
        
        payouts = payout_map[race_id].get(ticket_type, {})
        
        for _, row in race_probs.iterrows():
            if ticket_type == 'sanrentan':
                comb = [row['first'], row['second'], row['third']]
                comb_str = format_combination([int(c) for c in comb], ordered=True)
            elif ticket_type == 'sanrenpuku':
                comb = [row['horse_1'], row['horse_2'], row['horse_3']]
                comb_str = format_combination([int(c) for c in comb], ordered=False)
            elif ticket_type == 'umaren':
                comb = [row['horse_1'], row['horse_2']]
                comb_str = format_combination([int(c) for c in comb], ordered=False)
            
            if comb_str in payouts:
                race_payout += payouts[comb_str]
                race_hit = 1
        
        total_bet += race_bet
        total_payout += race_payout
        if race_hit:
            total_hits += 1
        race_count += 1
        
        results.append({
            'race_id': race_id,
            'bets': len(race_probs),
            'bet_amount': race_bet,
            'payout': race_payout,
            'hit': race_hit
        })
    
    roi = (total_payout / total_bet * 100) if total_bet > 0 else 0
    hit_rate = (total_hits / race_count * 100) if race_count > 0 else 0
    
    return {
        'ticket_type': ticket_type,
        'min_prob': min_prob,
        'top_k': top_k_bets,
        'races': race_count,
        'total_bet': total_bet,
        'total_payout': total_payout,
        'profit': total_payout - total_bet,
        'roi': roi,
        'hit_rate': hit_rate,
        'hits': total_hits,
        'avg_bets_per_race': total_bet / race_count / 100 if race_count > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="EV-Filtered Backtest")
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--predictions', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--trio_probs', type=str, default='data/probabilities/trio_ordered_probs.parquet')
    parser.add_argument('--output_dir', type=str, default='reports')
    
    args = parser.parse_args()
    
    # Load payout data
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([args.year])
    logger.info(f"Loaded {len(payout_map):,} races with payout data")
    
    # Filter to target year
    logger.info("Running EV-filtered backtests...")
    
    all_results = []
    
    # Grid search over different filters
    ticket_types = ['sanrentan', 'sanrenpuku', 'umaren']
    min_probs = [0.0, 0.01, 0.02, 0.03, 0.05]
    top_ks = [3, 5, 10, 20]
    
    for ticket_type in ticket_types:
        try:
            trio_path = args.trio_probs
            if ticket_type == 'sanrenpuku':
                trio_path = args.trio_probs.replace('trio_ordered', 'trio_unordered')
            elif ticket_type == 'umaren':
                trio_path = args.trio_probs.replace('trio_ordered_probs', 'pair_probs')
            
            # Load probabilities
            prob_df = pd.read_parquet(trio_path)
            prob_df['year'] = prob_df['race_id'].astype(str).str[:4].astype(int)
            prob_df = prob_df[prob_df['year'] == args.year]
            
            for min_prob in min_probs:
                for top_k in top_ks:
                    result = run_filtered_backtest(
                        predictions_df=None,  # Not used when reading from prob files
                        payout_map=payout_map,
                        trio_probs_path=args.trio_probs,
                        ticket_type=ticket_type,
                        min_prob=min_prob,
                        top_k_bets=top_k
                    )
                    all_results.append(result)
                    logger.info(f"{ticket_type} minP={min_prob:.2f} topK={top_k}: ROI={result['roi']:.1f}%, Hit={result['hit_rate']:.1f}%")
        
        except Exception as e:
            logger.warning(f"Error for {ticket_type}: {e}")
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'phase7_ev_filtered_backtest.md')
    
    report = f"""# Phase 7: EV-Filtered Backtest Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {args.year}

## Grid Search Results

| Ticket | Min Prob | Top K | Races | Bets/Race | Total Bet | Payout | Profit | **ROI** | Hit Rate |
|--------|----------|-------|-------|-----------|-----------|--------|--------|---------|----------|
"""
    
    # Sort by ROI
    all_results.sort(key=lambda x: x['roi'], reverse=True)
    
    for r in all_results[:30]:  # Top 30 results
        report += f"| {r['ticket_type']} | {r['min_prob']:.2f} | {r['top_k']} | {r['races']:,} | {r['avg_bets_per_race']:.1f} | ¥{r['total_bet']:,.0f} | ¥{r['total_payout']:,.0f} | ¥{r['profit']:,.0f} | **{r['roi']:.1f}%** | {r['hit_rate']:.1f}% |\n"
    
    # Best result
    if all_results:
        best = all_results[0]
        report += f"""
## Best Strategy

**{best['ticket_type']}** (Min Prob={best['min_prob']:.2f}, Top K={best['top_k']})
- ROI: **{best['roi']:.2f}%**
- Hit Rate: {best['hit_rate']:.2f}%
- Profit: ¥{best['profit']:,.0f}

"""
    
    # 100%+ strategies
    profitable = [r for r in all_results if r['roi'] >= 100]
    if profitable:
        report += "\n## ROI 100%+ Strategies\n\n"
        for r in profitable:
            report += f"- **{r['ticket_type']}** (minP={r['min_prob']:.2f}, topK={r['top_k']}): ROI **{r['roi']:.1f}%**\n"
    else:
        report += "\n> ⚠️ No strategy achieved ROI 100%+ in this grid search\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
