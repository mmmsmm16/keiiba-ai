"""
Paper Trade Aggregate: Period summary report generation
Phase 9: Real-world operation simulation

Usage:
    docker compose exec app python scripts/paper_trade_aggregate.py \
        --start 2025-12-01 --end 2025-12-16 \
        --config config/runtime/paper_trade.yaml
"""

import sys
import os
import argparse
import logging
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_date_range(start_str: str, end_str: str) -> List[str]:
    """Generate list of dates in range"""
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def aggregate_period(start_str: str, end_str: str, config: Dict) -> Dict[str, Any]:
    """Aggregate paper trade results for period"""
    
    data_dir = Path(config['report']['data_dir'])
    report_dir = Path(config['report']['out_dir'])
    
    dates = generate_date_range(start_str, end_str)
    logger.info(f"=== Paper Trade Aggregate: {start_str} to {end_str} ({len(dates)} days) ===")
    
    # Collect daily summaries
    daily_data = []
    all_ledger = []
    
    for date_str in dates:
        summary_path = report_dir / date_str / 'daily_summary.json'
        if not summary_path.exists():
            continue
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        if summary.get('status') != 'settled':
            logger.warning(f"{date_str}: not settled, skipping")
            continue
        
        m = summary['metrics']
        daily_data.append({
            'date': date_str,
            'races': m['total_races'],
            'race_hits': m['race_hits'],
            'race_hit_rate': m['race_hit_rate'],
            'tickets': m['total_tickets'],
            'ticket_hits': m['ticket_hits'],
            'ticket_hit_rate': m['ticket_hit_rate'],
            'stake': m['total_stake'],
            'payout': m['total_payout'],
            'profit': m['profit'],
            'roi': m['roi'],
            'max_dd': m['max_dd']
        })
        
        # Load ledger
        ledger_path = data_dir / date_str / 'ledger.parquet'
        if ledger_path.exists():
            ledger_df = pd.read_parquet(ledger_path)
            ledger_df['date'] = date_str
            all_ledger.append(ledger_df)
    
    if not daily_data:
        logger.warning("No settled days found in range")
        return {'status': 'no_data'}
    
    daily_df = pd.DataFrame(daily_data)
    
    # Aggregate metrics
    total_stake = daily_df['stake'].sum()
    total_payout = daily_df['payout'].sum()
    total_profit = daily_df['profit'].sum()
    total_races = daily_df['races'].sum()
    total_race_hits = daily_df['race_hits'].sum()
    total_tickets = daily_df['tickets'].sum()
    total_ticket_hits = daily_df['ticket_hits'].sum()
    
    roi = (total_payout / total_stake * 100) if total_stake > 0 else 0
    race_hit_rate = (total_race_hits / total_races * 100) if total_races > 0 else 0
    ticket_hit_rate = (total_ticket_hits / total_tickets * 100) if total_tickets > 0 else 0
    
    # Calculate period max drawdown
    if all_ledger:
        full_ledger = pd.concat(all_ledger, ignore_index=True)
        full_ledger = full_ledger.sort_values(['date', 'race_id'])
        full_ledger['cumulative_profit'] = (full_ledger['payout'].fillna(0) - full_ledger['stake']).cumsum()
        
        initial_bankroll = config['risk']['initial_bankroll']
        full_ledger['equity'] = initial_bankroll + full_ledger['cumulative_profit']
        
        max_equity = full_ledger['equity'].cummax()
        drawdown = (max_equity - full_ledger['equity']) / max_equity * 100
        max_dd = drawdown.max()
        final_equity = full_ledger['equity'].iloc[-1] if len(full_ledger) > 0 else initial_bankroll
    else:
        max_dd = 0
        final_equity = config['risk']['initial_bankroll']
    
    # Summary
    summary = {
        'period': {'start': start_str, 'end': end_str},
        'days_covered': len(daily_data),
        'metrics': {
            'total_races': int(total_races),
            'total_race_hits': int(total_race_hits),
            'race_hit_rate': round(race_hit_rate, 2),
            'total_tickets': int(total_tickets),
            'total_ticket_hits': int(total_ticket_hits),
            'ticket_hit_rate': round(ticket_hit_rate, 2),
            'total_stake': round(total_stake, 0),
            'total_payout': round(total_payout, 0),
            'total_profit': round(total_profit, 0),
            'roi': round(roi, 2),
            'max_dd': round(max_dd, 2),
            'final_equity': round(final_equity, 0)
        },
        'daily_breakdown': daily_data,
        'config_snapshot': {
            'ticket': config['strategy']['ticket'],
            'topn': config['strategy']['topn'],
            'odds_source': config['data']['odds_source'],
            'slippage_factor': config['data']['slippage_factor'],
            'initial_bankroll': config['risk']['initial_bankroll']
        }
    }
    
    # Save outputs
    out_prefix = f"summary_{start_str}_{end_str}"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    summary = convert_types(summary)
    
    with open(report_dir / f'{out_prefix}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    daily_df.to_csv(report_dir / f'{out_prefix}.csv', index=False)
    
    # Generate report
    report = generate_aggregate_report(summary, config)
    with open(report_dir / f'{out_prefix}.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Aggregate complete: {len(daily_data)} days")
    logger.info(f"ROI: {roi:.1f}%, Race Hit Rate: {race_hit_rate:.1f}%")
    logger.info(f"Total Profit: ¥{total_profit:,.0f}")
    
    return summary


def generate_aggregate_report(summary: Dict, config: Dict) -> str:
    """Generate aggregate report"""
    
    m = summary['metrics']
    cfg = summary['config_snapshot']
    daily = summary['daily_breakdown']
    
    report = f"""# Paper Trade Period Summary

**Period**: {summary['period']['start']} to {summary['period']['end']}
**Days Covered**: {summary['days_covered']}

## Configuration

| Parameter | Value |
|-----------|-------|
| Ticket Type | {cfg['ticket']} BOX{cfg['topn']} |
| Odds Source | {cfg['odds_source']} |
| Slippage Factor | {cfg['slippage_factor']} |
| Initial Bankroll | ¥{cfg['initial_bankroll']:,} |

> **Note**: odds_source=final は確定オッズです。実運用では事前オッズとの乖離に注意。

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| **Total Races** | {m['total_races']:,} |
| **Total Race Hits** | {m['total_race_hits']:,} |
| **Race Hit Rate** | {m['race_hit_rate']:.1f}% |
| **Total Tickets** | {m['total_tickets']:,} |
| **Total Ticket Hits** | {m['total_ticket_hits']:,} |
| **Ticket Hit Rate** | {m['ticket_hit_rate']:.2f}% |

---

## Financial Summary

| Metric | Value |
|--------|-------|
| **Total Stake** | ¥{m['total_stake']:,.0f} |
| **Total Payout** | ¥{m['total_payout']:,.0f} |
| **Total Profit** | ¥{m['total_profit']:+,.0f} |
| **ROI** | **{m['roi']:.1f}%** |
| **Max Drawdown** | {m['max_dd']:.1f}% |
| **Final Equity** | ¥{m['final_equity']:,.0f} |

---

## Daily Breakdown

| Date | Races | Hits | Hit Rate | Stake | Payout | Profit | ROI |
|------|-------|------|----------|-------|--------|--------|-----|
"""
    
    for d in daily:
        report += f"| {d['date']} | {d['races']} | {d['race_hits']} | {d['race_hit_rate']:.1f}% | ¥{d['stake']:,.0f} | ¥{d['payout']:,.0f} | ¥{d['profit']:+,.0f} | {d['roi']:.1f}% |\n"
    
    report += f"""
---

## Integrity Check

| Check | Status |
|-------|--------|
| ROI = Payout/Stake | {'✅' if abs(m['roi'] - m['total_payout']/m['total_stake']*100) < 0.1 else '❌'} |
| Profit = Payout - Stake | {'✅' if abs(m['total_profit'] - (m['total_payout'] - m['total_stake'])) < 1 else '❌'} |

---

## Backtest Comparison

| Metric | Paper Trade | Backtest (Phase8) | Delta |
|--------|-------------|-------------------|-------|
| ROI | {m['roi']:.1f}% | 612.5% | {m['roi'] - 612.5:+.1f}% |
| Race Hit Rate | {m['race_hit_rate']:.1f}% | 45.9% | {m['race_hit_rate'] - 45.9:+.1f}% |
| Ticket Hit Rate | {m['ticket_hit_rate']:.2f}% | 11.5% | {m['ticket_hit_rate'] - 11.5:+.2f}% |

> **Note**: 差異が大きい場合は調査が必要です。
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Paper Trade Aggregate")
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str,
                        default='config/runtime/paper_trade.yaml',
                        help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Aggregate
    result = aggregate_period(args.start, args.end, config)
    
    logger.info("Done!")
    return result


if __name__ == "__main__":
    main()
