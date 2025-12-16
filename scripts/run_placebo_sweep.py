"""
Placebo Sweep: Multiple seed comparison for model validation
複数シードでplaceboを実行し、本番との統計的比較を行う

Usage:
    docker compose exec app python scripts/run_placebo_sweep.py \
        --year 2025 --ticket sanrenpuku --topn 4 --n_seeds 20
"""

import sys
import os
import argparse
import logging
import subprocess
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_backtest(
    year: int,
    predictions_input: str,
    prob_col: str,
    ticket: str,
    topn: int,
    placebo: str,
    placebo_seed: int,
    bankroll: float,
    max_bet_frac: float,
    slippage_factor: float,
    odds_source: str = 'final',
    allow_final_odds: bool = True
) -> Dict[str, Any]:
    """Run backtest and parse output for key metrics"""
    
    cmd = [
        'python', 'src/backtest/multi_ticket_backtest_v2.py',
        '--year', str(year),
        '--predictions_input', predictions_input,
        '--prob_col', prob_col,
        '--odds_source', odds_source,
        '--slippage_factor', str(slippage_factor),
        '--bankroll', str(bankroll),
        '--max_bet_frac', str(max_bet_frac),
        '--placebo', placebo,
        '--placebo_seed', str(placebo_seed),
        '--output_dir', '/tmp/placebo_sweep'
    ]
    
    if allow_final_odds:
        cmd.append('--allow_final_odds')
    
    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace')
    
    # Parse output for metrics
    metrics = {
        'seed': placebo_seed,
        'placebo': placebo,
        'roi': None,
        'max_dd': None,
        'rescale': None,
        'races': None,
        'hits': None,
        'hit_rate': None,
        'ticket_hit_rate': None,
        'total_bet': None,
        'total_payout': None,
        'skip_count': None
    }
    
    for line in result.stdout.split('\n'):
        if f'{ticket} BOX{topn}' in line:
            # Parse: "sanrenpuku BOX4: ROI=612.47%, MaxDD=1.05%, Rescale=0"
            try:
                parts = line.split(':')[1]
                for part in parts.split(','):
                    part = part.strip()
                    if part.startswith('ROI='):
                        metrics['roi'] = float(part.split('=')[1].replace('%', ''))
                    elif part.startswith('MaxDD='):
                        metrics['max_dd'] = float(part.split('=')[1].replace('%', ''))
                    elif part.startswith('Rescale='):
                        metrics['rescale'] = int(part.split('=')[1])
            except:
                pass
    
    # Check stderr for errors
    if result.returncode != 0:
        logger.warning(f"Seed {placebo_seed}: non-zero return code")
        logger.warning(result.stderr[:500] if result.stderr else "No stderr")
    
    return metrics


def run_backtest_direct(
    df: pd.DataFrame,
    payout_map: Dict,
    ticket_type: str,
    top_n: int,
    placebo: str,
    placebo_seed: int,
    bankroll: float,
    max_bet_frac: float
) -> Dict[str, Any]:
    """Run backtest directly in-process for speed"""
    from backtest.multi_ticket_backtest_v2 import run_box_backtest
    from utils.payout_loader import format_combination
    
    df_run = df.copy()
    
    # Apply placebo shuffle
    if placebo == 'race_shuffle':
        np.random.seed(placebo_seed)
        df_run['prob'] = df_run.groupby('race_id')['prob'].transform(
            lambda x: x.sample(frac=1.0, random_state=placebo_seed).values
        )
    elif placebo == 'global_shuffle':
        np.random.seed(placebo_seed)
        df_run['prob'] = df_run['prob'].sample(frac=1.0, random_state=placebo_seed).values
    
    # Run backtest
    result = run_box_backtest(
        df_run, payout_map, ticket_type, top_n,
        bankroll=bankroll,
        max_bet_frac=max_bet_frac,
        min_equity_threshold=100,
        rescale_mode='scale',
        stop_if_bankrupt=False
    )
    
    return {
        'seed': placebo_seed,
        'placebo': placebo,
        'races': result.get('races', 0),
        'hits': result.get('hits', 0),
        'hit_rate': result.get('hit_rate', 0),
        'total_tickets': result.get('total_tickets', 0),
        'total_hit_tickets': result.get('total_hit_tickets', 0),
        'ticket_hit_rate': result.get('ticket_hit_rate', 0),
        'roi': result.get('roi', 0),
        'max_dd': result.get('max_dd_pct', 0),
        'total_bet': result.get('total_executed_bet', 0),
        'total_payout': result.get('total_payout', 0),
        'skip_count': result.get('skip_count', 0),
        'rescale_count': result.get('rescale_count', 0)
    }


def main():
    parser = argparse.ArgumentParser(description="Placebo Sweep for Model Validation")
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--predictions_input', type=str, 
                        default='data/predictions/v13_market_residual_2025_infer.parquet')
    parser.add_argument('--prob_col', type=str, default='prob_residual_softmax')
    parser.add_argument('--ticket', type=str, default='sanrenpuku')
    parser.add_argument('--topn', type=int, default=4)
    parser.add_argument('--odds_source', type=str, default='final')
    parser.add_argument('--slippage_factor', type=float, default=0.90)
    parser.add_argument('--bankroll', type=float, default=100_000_000,
                        help='Initial bankroll (high for full completion)')
    parser.add_argument('--max_bet_frac', type=float, default=0.0001,
                        help='Max bet fraction (low for stability)')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--out_md', type=str, 
                        default='reports/phase8/phase8_placebo_sweep.md')
    parser.add_argument('--out_csv', type=str, 
                        default='reports/phase8/phase8_placebo_sweep.csv')
    
    args = parser.parse_args()
    
    logger.info(f"=== Placebo Sweep: {args.ticket} BOX{args.topn}, {args.n_seeds} seeds ===")
    logger.info(f"Bankroll: ¥{args.bankroll:,.0f}, Max Bet Frac: {args.max_bet_frac}")
    
    # Load data once
    from utils.payout_loader import PayoutLoader
    from utils.race_filter import filter_jra_only
    
    logger.info(f"Loading data...")
    df = pd.read_parquet(args.predictions_input)
    
    # Load main data for horse_number
    main_df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
    main_df['year'] = main_df['race_id'].astype(str).str[:4].astype(int)
    main_df = main_df[main_df['year'] == args.year]
    main_df = filter_jra_only(main_df)
    
    # Merge to get horse_number
    merge_cols = ['race_id', 'horse_id']
    df = df.merge(main_df[['race_id', 'horse_id', 'horse_number']], 
                  on=merge_cols, how='left')
    df = df.dropna(subset=['horse_number', args.prob_col])
    df['prob'] = df[args.prob_col]
    
    logger.info(f"Data: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Load payouts
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([args.year])
    logger.info(f"Payout data: {len(payout_map):,} races")
    
    # 1) Run normal (placebo=none)
    logger.info("Running NORMAL (placebo=none)...")
    normal_result = run_backtest_direct(
        df, payout_map, args.ticket, args.topn,
        placebo='none',
        placebo_seed=0,
        bankroll=args.bankroll,
        max_bet_frac=args.max_bet_frac
    )
    logger.info(f"Normal: ROI={normal_result['roi']:.1f}%, Races={normal_result['races']}, Hits={normal_result['hits']}")
    
    # 2) Run placebo sweep
    logger.info(f"Running PLACEBO (race_shuffle) x {args.n_seeds} seeds...")
    placebo_results = []
    
    for seed in range(args.n_seeds):
        result = run_backtest_direct(
            df, payout_map, args.ticket, args.topn,
            placebo='race_shuffle',
            placebo_seed=seed,
            bankroll=args.bankroll,
            max_bet_frac=args.max_bet_frac
        )
        placebo_results.append(result)
        
        if (seed + 1) % 5 == 0:
            logger.info(f"  Seed {seed}: ROI={result['roi']:.1f}%, Races={result['races']}")
    
    # 3) Aggregate statistics
    placebo_df = pd.DataFrame(placebo_results)
    
    stats = {
        'roi_mean': placebo_df['roi'].mean(),
        'roi_std': placebo_df['roi'].std(),
        'roi_p5': placebo_df['roi'].quantile(0.05),
        'roi_p95': placebo_df['roi'].quantile(0.95),
        'hit_rate_mean': placebo_df['hit_rate'].mean(),
        'hit_rate_std': placebo_df['hit_rate'].std(),
        'ticket_hit_rate_mean': placebo_df['ticket_hit_rate'].mean(),
        'ticket_hit_rate_std': placebo_df['ticket_hit_rate'].std(),
        'max_dd_mean': placebo_df['max_dd'].mean(),
        'max_dd_std': placebo_df['max_dd'].std(),
        'races_mean': placebo_df['races'].mean(),
        'skip_count_mean': placebo_df['skip_count'].mean()
    }
    
    logger.info(f"Placebo stats: ROI={stats['roi_mean']:.1f}% ± {stats['roi_std']:.1f}%")
    
    # 4) Generate report
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    
    report = f"""# Phase 8: Placebo Sweep Comparison

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {args.year} (Holdout)
**Model**: v13_market_residual ({args.prob_col})
**Ticket**: {args.ticket} BOX{args.topn}
**Odds**: {args.odds_source}, slippage={args.slippage_factor}

## Bankroll Settings (Full Completion Mode)

| Parameter | Value |
|-----------|-------|
| Initial Bankroll | ¥{args.bankroll:,.0f} |
| Max Bet Fraction | {args.max_bet_frac*100:.4f}% |
| Placebo Seeds | {args.n_seeds} |

---

## Normal (placebo=none)

| Metric | Value |
|--------|-------|
| Races | {normal_result['races']:,} |
| Hits | {normal_result['hits']:,} |
| Race Hit Rate | {normal_result['hit_rate']:.2f}% |
| Total Tickets | {normal_result['total_tickets']:,} |
| Hit Tickets | {normal_result['total_hit_tickets']:,} |
| Ticket Hit Rate | {normal_result['ticket_hit_rate']:.2f}% |
| **ROI** | **{normal_result['roi']:.2f}%** |
| Max DD | {normal_result['max_dd']:.2f}% |
| Total Bet | ¥{normal_result['total_bet']:,.0f} |
| Total Payout | ¥{normal_result['total_payout']:,.0f} |
| Skip Count | {normal_result['skip_count']} |

---

## Placebo (race_shuffle) Statistics ({args.n_seeds} seeds)

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|----|
| ROI | {stats['roi_mean']:.2f}% | {stats['roi_std']:.2f}% | {stats['roi_p5']:.2f}% | {stats['roi_p95']:.2f}% |
| Race Hit Rate | {stats['hit_rate_mean']:.2f}% | {stats['hit_rate_std']:.2f}% | - | - |
| Ticket Hit Rate | {stats['ticket_hit_rate_mean']:.2f}% | {stats['ticket_hit_rate_std']:.2f}% | - | - |
| Max DD | {stats['max_dd_mean']:.2f}% | {stats['max_dd_std']:.2f}% | - | - |
| Races | {stats['races_mean']:.0f} | - | - | - |
| Skip Count | {stats['skip_count_mean']:.1f} | - | - | - |

---

## Comparison Summary

| Metric | Normal | Placebo (mean) | Delta | Judgment |
|--------|--------|----------------|-------|----------|
| **ROI** | **{normal_result['roi']:.1f}%** | {stats['roi_mean']:.1f}% | **{normal_result['roi'] - stats['roi_mean']:+.1f}%** | {'✅' if normal_result['roi'] > stats['roi_p95'] else '⚠️'} |
| Race Hit Rate | {normal_result['hit_rate']:.1f}% | {stats['hit_rate_mean']:.1f}% | {normal_result['hit_rate'] - stats['hit_rate_mean']:+.1f}% | {'✅' if normal_result['hit_rate'] > stats['hit_rate_mean'] else '⚠️'} |
| Ticket Hit Rate | {normal_result['ticket_hit_rate']:.2f}% | {stats['ticket_hit_rate_mean']:.2f}% | {normal_result['ticket_hit_rate'] - stats['ticket_hit_rate_mean']:+.2f}% | {'✅' if normal_result['ticket_hit_rate'] > stats['ticket_hit_rate_mean'] else '⚠️'} |
| Max DD | {normal_result['max_dd']:.1f}% | {stats['max_dd_mean']:.1f}% | {normal_result['max_dd'] - stats['max_dd_mean']:+.1f}% | {'✅' if normal_result['max_dd'] < stats['max_dd_mean'] else '⚠️'} |
| Races | {normal_result['races']} | {stats['races_mean']:.0f} | {normal_result['races'] - stats['races_mean']:.0f} | {'✅' if abs(normal_result['races'] - stats['races_mean']) < 10 else '⚠️'} |

---

## Conclusion

> **{'✅ Model prediction is VALID' if normal_result['roi'] > stats['roi_p95'] else '⚠️ Review needed'}**

"""
    
    if normal_result['roi'] > stats['roi_p95']:
        report += f"""
Normal ROI ({normal_result['roi']:.1f}%) > Placebo P95 ({stats['roi_p95']:.1f}%)

This means the v13 model prediction is statistically significant with >95% confidence.
The model's ranking ability is not due to random chance.
"""
    else:
        report += f"""
**WARNING**: Normal ROI is within placebo distribution range.
This may indicate:
1. Potential data leakage
2. Model not learning meaningful patterns
3. Implementation issue

Investigation required.
"""
    
    report += f"""
---

## Re-run Commands

```bash
# This sweep
docker compose exec app python scripts/run_placebo_sweep.py \\
    --year {args.year} \\
    --ticket {args.ticket} --topn {args.topn} \\
    --predictions_input {args.predictions_input} \\
    --prob_col {args.prob_col} \\
    --bankroll {args.bankroll} --max_bet_frac {args.max_bet_frac} \\
    --n_seeds {args.n_seeds} \\
    --out_md {args.out_md} --out_csv {args.out_csv}
```
"""
    
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved: {args.out_md}")
    
    # Save CSV
    placebo_df['type'] = 'placebo'
    normal_df = pd.DataFrame([normal_result])
    normal_df['type'] = 'normal'
    
    all_df = pd.concat([normal_df, placebo_df], ignore_index=True)
    all_df.to_csv(args.out_csv, index=False)
    logger.info(f"CSV saved: {args.out_csv}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
