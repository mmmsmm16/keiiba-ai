"""
Paper Trade Settle: Settlement and daily report generation
Phase 9: Real-world operation simulation

Usage:
    docker compose exec app python scripts/paper_trade_settle.py \
        --date 2025-12-16 \
        --config configs/runtime/paper_trade.yaml
"""

import sys
import os
import argparse
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

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


def settle_daily(date_str: str, config: Dict) -> Dict[str, Any]:
    """Settle daily paper trade with actual results"""
    
    from utils.payout_loader import PayoutLoader
    
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    
    logger.info(f"=== Paper Trade Settle: {date_str} ===")
    
    # Paths
    data_dir = Path(config['report']['data_dir']) / date_str
    report_dir = Path(config['report']['out_dir']) / date_str
    
    # Check if run was completed
    ledger_path = data_dir / 'ledger.parquet'
    if not ledger_path.exists():
        logger.error(f"Ledger not found: {ledger_path}")
        logger.error("Run paper_trade_run.py first")
        return {'status': 'error', 'message': 'Ledger not found'}
    
    # Load pending ledger
    ledger_df = pd.read_parquet(ledger_path)
    logger.info(f"Loaded ledger: {len(ledger_df)} tickets")
    
    # Load payout data
    logger.info("Loading payout data...")
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([year])
    
    ticket_type = config['strategy']['ticket']
    slippage = config['data']['slippage_factor']
    
    # Settle each ticket
    total_payout = 0
    total_stake = 0
    hits = 0
    processed_races = set()
    missing_payout_races = []
    
    for idx, row in ledger_df.iterrows():
        race_id = row['race_id']
        ticket_key = row['ticket_key']
        stake = row['stake']
        
        total_stake += stake
        processed_races.add(race_id)
        
        # Check payout
        if race_id not in payout_map:
            missing_payout_races.append(race_id)
            ledger_df.at[idx, 'payout'] = 0
            ledger_df.at[idx, 'is_hit'] = False
            ledger_df.at[idx, 'skip_reason'] = 'missing_payouts'
            continue
        
        race_payouts = payout_map[race_id].get(ticket_type, {})
        
        if ticket_key in race_payouts:
            # Hit! Apply slippage to payout
            base_payout = race_payouts[ticket_key]
            payout = base_payout * slippage * (stake / 100)  # Scale by actual stake
            ledger_df.at[idx, 'payout'] = payout
            ledger_df.at[idx, 'is_hit'] = True
            total_payout += payout
            hits += 1
        else:
            ledger_df.at[idx, 'payout'] = 0
            ledger_df.at[idx, 'is_hit'] = False
    
    # Calculate metrics
    n_tickets = len(ledger_df)
    n_races = len(processed_races)
    
    # Count race-level hits (any ticket in race hit)
    race_hits = ledger_df.groupby('race_id')['is_hit'].any().sum()
    
    profit = total_payout - total_stake
    roi = (total_payout / total_stake * 100) if total_stake > 0 else 0
    race_hit_rate = (race_hits / n_races * 100) if n_races > 0 else 0
    ticket_hit_rate = (hits / n_tickets * 100) if n_tickets > 0 else 0
    
    # Calculate drawdown
    ledger_df['cumulative_profit'] = (ledger_df['payout'].fillna(0) - ledger_df['stake']).cumsum()
    initial_bankroll = config['risk']['initial_bankroll']
    ledger_df['equity'] = initial_bankroll + ledger_df['cumulative_profit']
    
    max_equity = ledger_df['equity'].cummax()
    drawdown = (max_equity - ledger_df['equity']) / max_equity * 100
    max_dd = drawdown.max()
    
    final_equity = initial_bankroll + profit
    
    # Save updated ledger
    ledger_df.to_parquet(ledger_path, index=False)
    logger.info(f"Updated ledger saved")
    
    # Load race/horse info for detailed report
    base_path = 'data/processed/preprocessed_data_v11.parquet'
    base_df = pd.read_parquet(base_path)
    
    # Filter to races in ledger (more reliable than date filtering)
    ledger_race_ids = ledger_df['race_id'].unique().tolist()
    base_df = base_df[base_df['race_id'].isin(ledger_race_ids)]
    
    # Load predictions for probability display
    predictions_path = data_dir / 'predictions.parquet'
    predictions_df = None
    if predictions_path.exists():
        predictions_df = pd.read_parquet(predictions_path)
        logger.info(f"Loaded predictions: {len(predictions_df)} rows")
    
    # Extract race info
    race_info = {}
    for race_id in processed_races:
        race_df = base_df[base_df['race_id'] == race_id]
        if len(race_df) > 0:
            row = race_df.iloc[0]
            # Parse race_id: YYYYJJKKNNNR (JRA format)
            # YY=year, JJ=venue, KK=kaisai, NNN=day, R=race_num (last 2 digits)
            rid_str = str(race_id)
            if len(rid_str) >= 12:
                venue_code = rid_str[4:6]   # Position 5-6: venue
                kai = rid_str[6:8]          # Position 7-8: 開催回次
                day = rid_str[8:10]         # Position 9-10: 日目
                race_num = rid_str[10:12]   # Position 11-12: レース番号
            else:
                venue_code = ''
                race_num = ''
            
            # JRA Course code mapping
            course_map = {
                '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
                '05': '東京', '06': '中山', '07': '中京', '08': '京都',
                '09': '阪神', '10': '小倉'
            }
            venue = course_map.get(venue_code, f'場{venue_code}')
            
            race_info[race_id] = {
                'venue': venue,
                'race_num': int(race_num) if race_num.isdigit() else race_num,
                'race_name': row.get('race_name', '') if 'race_name' in row.index else ''
            }
    
    # Get horse names mapping
    horse_names = {}
    for race_id in processed_races:
        race_df = base_df[base_df['race_id'] == race_id]
        for _, row in race_df.iterrows():
            h_num = int(row['horse_number'])
            h_name = row.get('horse_name', f'馬{h_num}') if 'horse_name' in row.index else f'馬{h_num}'
            horse_names[(race_id, h_num)] = h_name
    
    # Create summary
    summary = {
        'date': date_str,
        'status': 'settled',
        'metrics': {
            'total_races': n_races,
            'race_hits': int(race_hits),
            'race_hit_rate': round(race_hit_rate, 2),
            'total_tickets': n_tickets,
            'ticket_hits': int(hits),
            'ticket_hit_rate': round(ticket_hit_rate, 2),
            'total_stake': round(total_stake, 0),
            'total_payout': round(total_payout, 0),
            'profit': round(profit, 0),
            'roi': round(roi, 2),
            'max_dd': round(max_dd, 2),
            'final_equity': round(final_equity, 0)
        },
        'issues': {
            'missing_payout_races': len(missing_payout_races),
            'missing_payout_samples': missing_payout_races[:10]
        },
        'config_snapshot': {
            'ticket': config['strategy']['ticket'],
            'topn': config['strategy']['topn'],
            'odds_source': config['data']['odds_source'],
            'slippage_factor': config['data']['slippage_factor'],
            'initial_bankroll': config['risk']['initial_bankroll']
        }
    }
    
    # Save summary
    with open(report_dir / 'daily_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Generate daily report with detailed race info
    report = generate_daily_report(summary, config, ledger_df, race_info, horse_names, predictions_df)
    with open(report_dir / 'daily_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Settlement complete!")
    logger.info(f"ROI: {roi:.1f}%, Race Hit Rate: {race_hit_rate:.1f}%")
    logger.info(f"Profit: ¥{profit:,.0f}")
    
    return summary


def generate_daily_report(summary: Dict, config: Dict, ledger_df: pd.DataFrame = None, 
                          race_info: Dict = None, horse_names: Dict = None,
                          predictions_df: pd.DataFrame = None) -> str:
    """Generate daily report with detailed race breakdown and horse probabilities"""
    
    m = summary['metrics']
    issues = summary['issues']
    cfg = summary['config_snapshot']
    
    report = f"""# Paper Trade Daily Report

**Date**: {summary['date']}
**Status**: ✅ Settled

## Configuration

| Parameter | Value |
|-----------|-------|
| Ticket Type | {cfg['ticket']} BOX{cfg['topn']} |
| Odds Source | {cfg['odds_source']} |
| Slippage Factor | {cfg['slippage_factor']} |
| Initial Bankroll | ¥{cfg['initial_bankroll']:,} |

> **Note**: odds_source=final は確定オッズです。実運用では事前オッズとの乖離に注意。

---

## Results Summary

| Metric | Value |
|--------|-------|
| **Total Races** | {m['total_races']} |
| **Race Hits** | {m['race_hits']} |
| **Race Hit Rate** | {m['race_hit_rate']:.1f}% |
| **Total Tickets** | {m['total_tickets']} |
| **Ticket Hits** | {m['ticket_hits']} |
| **Ticket Hit Rate** | {m['ticket_hit_rate']:.2f}% |

---

## Financial Summary

| Metric | Value |
|--------|-------|
| **Total Stake** | ¥{m['total_stake']:,.0f} |
| **Total Payout** | ¥{m['total_payout']:,.0f} |
| **Profit** | ¥{m['profit']:+,.0f} |
| **ROI** | **{m['roi']:.1f}%** |
| **Max Drawdown** | {m['max_dd']:.1f}% |
| **Final Equity** | ¥{m['final_equity']:,.0f} |

---

## Integrity Check

| Check | Status |
|-------|--------|
| ROI = Payout/Stake | {'✅' if abs(m['roi'] - m['total_payout']/m['total_stake']*100) < 0.1 else '❌'} |
| Final Equity = Initial + Profit | {'✅' if abs(m['final_equity'] - (cfg['initial_bankroll'] + m['profit'])) < 1 else '❌'} |

---

## Race-by-Race Detail

"""
    
    # Add detailed race breakdown if ledger provided
    if ledger_df is not None and race_info is not None:
        # Group by race
        for race_id in sorted(ledger_df['race_id'].unique()):
            race_ledger = ledger_df[ledger_df['race_id'] == race_id]
            
            # Get race info
            info = race_info.get(race_id, {'venue': '?', 'race_num': '?', 'race_name': ''})
            venue = info['venue']
            race_num = info['race_num']
            race_name = info.get('race_name', '')
            
            # Race header
            race_hit = race_ledger['is_hit'].any()
            race_payout = race_ledger['payout'].sum()
            race_stake = race_ledger['stake'].sum()
            
            hit_mark = "✅ 的中" if race_hit else "❌ ハズレ"
            
            report += f"### {venue} {race_num}R {hit_mark}\n\n"
            if race_name:
                report += f"**{race_name}**\n\n"
            
            # Add probability table if predictions available
            if predictions_df is not None:
                race_preds = predictions_df[predictions_df['race_id'] == race_id].copy()
                if len(race_preds) > 0:
                    # Determine probability column
                    prob_col = None
                    for col in ['prob_residual_softmax', 'prob', 'prob_raw']:
                        if col in race_preds.columns:
                            prob_col = col
                            break
                    
                    if prob_col:
                        race_preds = race_preds.sort_values(prob_col, ascending=False)
                        
                        # Check available columns
                        has_market = 'p_market' in race_preds.columns and race_preds['p_market'].notna().any()
                        has_odds = 'odds' in race_preds.columns and race_preds['odds'].notna().any()
                        has_raw = 'raw_score' in race_preds.columns and race_preds['raw_score'].notna().any()
                        
                        report += "**📊 予測スコア比較**\n\n"
                        
                        if has_market and has_odds and has_raw:
                            report += "| 順位 | 馬番 | 馬名 | オッズ | 市場確率 | RawScore | Softmax確率 | 差分 | 着順 |\n"
                            report += "|------|------|------|--------|----------|----------|-------------|------|------|\n"
                        elif has_market and has_odds:
                            report += "| 順位 | 馬番 | 馬名 | オッズ | 市場確率 | モデル確率 | 差分 | 着順 |\n"
                            report += "|------|------|------|--------|----------|------------|------|------|\n"
                        else:
                            report += "| 順位 | 馬番 | 馬名 | モデル確率 | 着順 |\n"
                            report += "|------|------|------|------------|------|\n"
                        
                        for i, (_, pred_row) in enumerate(race_preds.iterrows()):
                            h_num = int(pred_row.get('horse_number', 0))
                            prob = pred_row[prob_col] * 100
                            h_name = horse_names.get((race_id, h_num), f'馬{h_num}') if horse_names else f'馬{h_num}'
                            
                            # Get actual rank if available
                            rank_str = ""
                            if 'rank' in pred_row.index and pd.notna(pred_row['rank']):
                                rank = int(pred_row['rank'])
                                if rank == 1:
                                    rank_str = "🥇"
                                elif rank == 2:
                                    rank_str = "🥈"
                                elif rank == 3:
                                    rank_str = "🥉"
                                else:
                                    rank_str = f"{rank}着"
                            
                            # Mark top 4
                            top_mark = "⭐" if i < 4 else ""
                            
                            if has_market and has_odds and has_raw:
                                odds = pred_row.get('odds', 0)
                                p_mkt = pred_row.get('p_market', 0) * 100 if pd.notna(pred_row.get('p_market')) else 0
                                raw = pred_row.get('raw_score', 0) if pd.notna(pred_row.get('raw_score')) else 0
                                edge = pred_row.get('edge', 0) * 100 if pd.notna(pred_row.get('edge')) else 0
                                
                                edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"
                                
                                report += f"| {top_mark}{i+1} | {h_num} | {h_name[:6]} | {odds:.1f} | {p_mkt:.1f}% | {raw:+.2f} | {prob:.1f}% | {edge_str} | {rank_str} |\n"
                            elif has_market and has_odds:
                                odds = pred_row.get('odds', 0)
                                p_mkt = pred_row.get('p_market', 0) * 100 if pd.notna(pred_row.get('p_market')) else 0
                                edge = pred_row.get('edge', 0) * 100 if pd.notna(pred_row.get('edge')) else 0
                                
                                edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"
                                edge_color = edge_str if edge != 0 else "-"
                                
                                report += f"| {top_mark}{i+1} | {h_num} | {h_name[:8]} | {odds:.1f} | {p_mkt:.1f}% | {prob:.1f}% | {edge_color} | {rank_str} |\n"
                            else:
                                report += f"| {top_mark}{i+1} | {h_num} | {h_name[:10]} | {prob:.1f}% | {rank_str} |\n"
                        
                        report += "\n"
            
            # Ticket table
            report += "**📝 購入馬券**\n\n"
            report += "| 買い目 | 馬名 | Stake | Payout | Hit |\n"
            report += "|--------|------|-------|--------|-----|\n"
            
            for _, row in race_ledger.iterrows():
                ticket_display = row['display']
                stake = row['stake']
                payout = row['payout'] if row['payout'] else 0
                is_hit = row['is_hit']
                
                # Get horse names
                horses = eval(row['horses']) if isinstance(row['horses'], str) else row['horses']
                h_names = []
                for h in horses:
                    name = horse_names.get((race_id, h), f'馬{h}') if horse_names else f'馬{h}'
                    h_names.append(name)
                horse_str = '-'.join(h_names)
                
                hit_str = "✅" if is_hit else ""
                payout_str = f"¥{payout:,.0f}" if payout > 0 else "-"
                
                report += f"| {ticket_display} | {horse_str} | ¥{stake:,} | {payout_str} | {hit_str} |\n"
            
            # Race summary
            profit = race_payout - race_stake
            report += f"\n**小計**: Stake ¥{race_stake:,} → Payout ¥{race_payout:,.0f} (Profit ¥{profit:+,.0f})\n\n"
            report += "---\n\n"
    
    # Issues section
    report += """## Issues

| Issue | Count |
|-------|-------|
"""
    report += f"| Missing Payout Data | {issues['missing_payout_races']} |\n"
    
    if issues['missing_payout_samples']:
        report += "\n### Missing Payout Samples\n"
        for rid in issues['missing_payout_samples'][:5]:
            report += f"- {rid}\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Paper Trade Settle")
    parser.add_argument('--date', type=str, required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str,
                        default='configs/runtime/paper_trade.yaml',
                        help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Settle
    result = settle_daily(args.date, config)
    
    logger.info("Done!")
    return result


if __name__ == "__main__":
    main()
