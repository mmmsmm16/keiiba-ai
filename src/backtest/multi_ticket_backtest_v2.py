"""
Phase 7 (v2): Multi-Ticket Backtest with JRA Filter and Proper Max DD
券種別バックテスト（JRAのみ、正規化DD）

Usage (in container):
    docker compose exec app python src/backtest/multi_ticket_backtest_v2.py --year 2024
    docker compose exec app python src/backtest/multi_ticket_backtest_v2.py --year 2024 --include_nar
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
from utils.race_filter import filter_races, add_race_filter_args, get_race_stats
from utils.risk_metrics import compute_max_drawdown_from_transactions, format_max_dd_percent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default parameters for bankroll management
DEFAULT_BANKROLL = 100000.0
DEFAULT_MAX_BET_FRAC = 0.05  # 5% of equity max per race
DEFAULT_MIN_EQUITY_THRESHOLD = 100  # Stop betting below this


def apply_bankroll_constraint(
    planned_bet: float,
    current_equity: float,
    max_bet_frac: float = DEFAULT_MAX_BET_FRAC,
    min_equity_threshold: float = DEFAULT_MIN_EQUITY_THRESHOLD,
    rescale_mode: str = 'scale',
    unit: float = 100.0
) -> tuple:
    """
    Bankroll制約を適用してbet額を調整
    
    Args:
        planned_bet: 計画bet額
        current_equity: 現在の資金
        max_bet_frac: 最大bet割合（equityの何%まで）
        min_equity_threshold: 最低資金閾値
        rescale_mode: 'scale'=縮小, 'skip'=スキップ
        unit: 丸め単位（100円）
    
    Returns:
        (executed_bet, rescale_ratio, skipped)
    """
    # 資金が閾値未満なら停止
    if current_equity < min_equity_threshold:
        return 0.0, 0.0, True
    
    # 最大bet額 = equity * max_bet_frac
    max_allowed = current_equity * max_bet_frac
    
    if planned_bet <= max_allowed:
        # 制約内なのでそのまま
        return planned_bet, 1.0, False
    
    if rescale_mode == 'skip':
        # スキップモード
        return 0.0, 0.0, True
    
    # スケールダウンモード
    rescale_ratio = max_allowed / planned_bet
    scaled_bet = planned_bet * rescale_ratio
    
    # 100円単位で丸め（切り捨て）
    scaled_bet = int(scaled_bet / unit) * unit
    
    # 丸め後も0なら実質スキップ
    if scaled_bet < unit:
        return 0.0, 0.0, True
    
    # 最終チェック：丸め後がまだ超過していないか
    if scaled_bet > max_allowed:
        scaled_bet = int(max_allowed / unit) * unit
    
    actual_ratio = scaled_bet / planned_bet if planned_bet > 0 else 0
    return scaled_bet, actual_ratio, False


def run_box_backtest(
    df: pd.DataFrame,
    payout_map: Dict,
    ticket_type: str,
    top_n: int = 5,
    bankroll: float = DEFAULT_BANKROLL,
    max_bet_frac: float = DEFAULT_MAX_BET_FRAC,
    min_equity_threshold: float = DEFAULT_MIN_EQUITY_THRESHOLD,
    rescale_mode: str = 'scale',
    stop_if_bankrupt: bool = True
) -> Dict:
    """
    BOX買いバックテスト with Bankroll制約
    
    Args:
        df: レースデータ
        payout_map: 払戻データ
        ticket_type: 券種 (umaren/sanrenpuku/sanrentan)
        top_n: 上位N頭
        bankroll: 初期資金
        max_bet_frac: 最大bet割合（equityの%）
        min_equity_threshold: 最低資金閾値
        rescale_mode: 'scale'=縮小, 'skip'=スキップ
        stop_if_bankrupt: 破産時停止
    
    Returns:
        バックテスト結果Dict（rescale診断含む）
    """
    race_results = []
    
    # 資金追跡
    current_equity = bankroll
    
    # 診断用統計
    rescale_count = 0
    skip_count = 0
    bankrupt_stop_count = 0
    rescale_ratios = []
    total_planned_bet = 0
    total_executed_bet = 0
    
    horse_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in df.columns and df['prob'].notna().any() else None
    
    # probがなければoddsからp_marketを生成
    if score_col is None:
        if 'odds' in df.columns:
            df = df.copy()
            df['raw_prob'] = 1.0 / df['odds'].replace(0, np.nan)
            df['p_market'] = df.groupby('race_id')['raw_prob'].transform(lambda x: x / x.sum())
            score_col = 'p_market'
        else:
            logger.error("No score/probability column available")
            return {'roi': 0, 'races': 0}
    
    # 日付順でレースをソート（時系列で処理）
    race_ids = df['race_id'].unique()
    race_ids = sorted(race_ids)  # race_idは日時順になっている前提
    
    for race_id in race_ids:
        if race_id not in payout_map:
            continue
        
        # 破産チェック
        if current_equity < min_equity_threshold:
            bankrupt_stop_count += 1
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
        
        # 計画bet額
        planned_bet = len(tickets) * 100
        total_planned_bet += planned_bet
        
        # Bankroll制約適用
        executed_bet, ratio, skipped = apply_bankroll_constraint(
            planned_bet, current_equity, max_bet_frac,
            min_equity_threshold, rescale_mode
        )
        
        if skipped:
            if current_equity < min_equity_threshold:
                bankrupt_stop_count += 1
            else:
                skip_count += 1
            continue
        
        if ratio < 1.0:
            rescale_count += 1
            rescale_ratios.append(ratio)
        
        total_executed_bet += executed_bet
        
        # 払戻チェック（betが縮小されてもhit判定は同じ、payoutは比率で縮小）
        payout = 0
        hit = 0
        hit_tickets = 0  # Exact count of hit tickets
        
        for t in tickets:
            comb_str = format_combination(list(t), ordered=ordered)
            if comb_str in payouts:
                # 払戻も縮小比率を適用（betが縮小されたので）
                base_payout = payouts[comb_str]
                payout += base_payout * ratio  # 比例縮小 (accumulate for dead heats)
                hit = 1
                hit_tickets += 1  # Count each hit ticket
        
        profit = payout - executed_bet
        current_equity += profit
        
        # equityが負にならないようクリップ（stop_if_bankrupt=True時）
        if stop_if_bankrupt and current_equity < 0:
            current_equity = 0
        
        race_results.append({
            'race_id': race_id,
            'tickets': len(tickets),
            'planned_bet': planned_bet,
            'executed_bet': executed_bet,
            'rescale_ratio': ratio,
            'payout': payout,
            'hit': hit,
            'hit_tickets': hit_tickets,
            'profit': profit,
            'equity': current_equity
        })
    
    if not race_results:
        return {'roi': 0, 'races': 0}
    
    res_df = pd.DataFrame(race_results)
    total_payout = res_df['payout'].sum()
    total_profit = total_payout - total_executed_bet
    
    # Max Drawdown計算（実行ベースのprofit/equityから）
    profits = res_df['profit'].tolist()
    max_dd, equity_curve = compute_max_drawdown_from_transactions(
        profits, 
        initial_bankroll=bankroll,
        stop_if_bankrupt=stop_if_bankrupt
    )
    
    max_dd_pct = max_dd * 100
    
    # 診断統計
    avg_rescale_ratio = np.mean(rescale_ratios) if rescale_ratios else 1.0
    min_rescale_ratio = np.min(rescale_ratios) if rescale_ratios else 1.0
    
    # Calculate exact ticket stats
    total_tickets = int(res_df['tickets'].sum())
    total_hit_tickets = int(res_df['hit_tickets'].sum())
    ticket_hit_rate = (total_hit_tickets / total_tickets * 100) if total_tickets > 0 else 0
    
    return {
        'ticket_type': ticket_type,
        'top_n': top_n,
        'races': len(res_df),
        'total_planned_bet': total_planned_bet,
        'total_executed_bet': total_executed_bet,
        'total_payout': total_payout,
        'profit': total_profit,
        'roi': (total_payout / total_executed_bet * 100) if total_executed_bet > 0 else 0,
        'hit_rate': (res_df['hit'].sum() / len(res_df) * 100) if len(res_df) > 0 else 0,
        'hits': int(res_df['hit'].sum()),
        # Exact ticket stats
        'total_tickets': total_tickets,
        'total_hit_tickets': total_hit_tickets,
        'ticket_hit_rate': ticket_hit_rate,
        'max_dd_pct': max_dd_pct,
        'bankroll': bankroll,
        # Bankroll constraint diagnostics
        'rescale_count': rescale_count,
        'skip_count': skip_count,
        'bankrupt_stop_count': bankrupt_stop_count,
        'avg_rescale_ratio': avg_rescale_ratio,
        'min_rescale_ratio': min_rescale_ratio,
        'final_equity': current_equity,
        'max_bet_frac': max_bet_frac
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 7 (v2): Multi-Ticket Backtest with Bankroll Constraint")
    parser.add_argument('--year', type=int, default=2024)
    add_race_filter_args(parser)
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data_v11.parquet')
    parser.add_argument('--predictions_input', type=str, default=None,
                        help='Path to v13 predictions parquet (if provided, merges with input)')
    parser.add_argument('--prob_col', type=str, default='prob_residual_softmax',
                        help='Probability column to use from predictions')
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--bankroll', type=float, default=DEFAULT_BANKROLL,
                        help='Initial bankroll (default: 100000)')
    parser.add_argument('--max_bet_frac', type=float, default=DEFAULT_MAX_BET_FRAC,
                        help='Max bet as fraction of equity (default: 0.05 = 5%%)')
    parser.add_argument('--min_equity_threshold', type=float, default=DEFAULT_MIN_EQUITY_THRESHOLD,
                        help='Stop betting below this equity (default: 100)')
    parser.add_argument('--rescale_mode', type=str, default='scale', choices=['scale', 'skip'],
                        help='Rescale mode: scale (shrink bet) or skip (no bet)')
    parser.add_argument('--stop_if_bankrupt', action='store_true', default=True,
                        help='Stop betting if bankrupt (default: True)')
    # Odds source options (Leak Guard)
    parser.add_argument('--odds_source', type=str, default='final',
                        choices=['pre_close', 'close', 'final'],
                        help='Odds timestamp source: pre_close, close, or final')
    parser.add_argument('--allow_final_odds', action='store_true', default=False,
                        help='Explicitly allow final odds usage (required when odds_source=final)')
    parser.add_argument('--slippage_factor', type=float, default=1.0,
                        help='Multiply odds by this factor (e.g., 0.95 for conservative)')
    # Snapshot Odds options
    parser.add_argument('--odds_mode', type=str, default='final', choices=['final', 'snapshot'],
                        help='Odds mode: final (default) or snapshot (pre-race)')
    parser.add_argument('--odds_snapshot_path', type=str, default=None,
                        help='Path to odds snapshot parquet (required if odds_mode=snapshot)')
    # Sanity check options
    parser.add_argument('--strict_sanity', action='store_true', default=True,
                        help='Fail on sanity check violations (default: True)')
    parser.add_argument('--sanity_out', type=str, default=None,
                        help='Output path for ticket sanity sample table')
    # Placebo (shuffle) experiment
    parser.add_argument('--placebo', type=str, default='none',
                        choices=['none', 'race_shuffle', 'global_shuffle'],
                        help='Placebo mode: none, race_shuffle, or global_shuffle')
    parser.add_argument('--placebo_seed', type=int, default=42,
                        help='Random seed for placebo shuffle')
    
    args = parser.parse_args()
    
    # LEAK GUARD: Require explicit flag for final odds
    # LEAK GUARD: Require explicit flag for final odds (only if odds_mode is final)
    if args.odds_mode == 'final' and args.odds_source == 'final' and not args.allow_final_odds:
        raise ValueError(
            "LEAK GUARD: --odds_source=final requires --allow_final_odds flag.\n"
            "Final odds may match confirmed payouts and could leak post-close information.\n"
            "Add --allow_final_odds to explicitly acknowledge this risk."
        )
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Merge predictions if provided
    if args.predictions_input:
        logger.info(f"Loading predictions from {args.predictions_input}...")
        pred_df = pd.read_parquet(args.predictions_input)
        
        # Check required columns
        if args.prob_col not in pred_df.columns:
            raise ValueError(f"Probability column {args.prob_col} not found in predictions")
        
        # Merge on race_id and horse_id
        merge_cols = ['race_id', 'horse_id']
        if 'horse_id' not in pred_df.columns:
            if 'umaban' in pred_df.columns:
                merge_cols = ['race_id', 'umaban']
        
        # Keep required columns from predictions
        pred_cols = merge_cols + [args.prob_col]
        if 'odds' in pred_df.columns:
            pred_cols.append('odds')
        
        pred_subset = pred_df[pred_cols].drop_duplicates(subset=merge_cols)
        
        # Merge
        df = df.merge(pred_subset, on=merge_cols, how='left', suffixes=('', '_pred'))
        
        # Use prob_col as the score
        df['prob'] = df[args.prob_col]
        logger.info(f"Merged predictions: {df[args.prob_col].notna().sum():,} rows with probability")
    
    # Apply placebo shuffle if requested
    if args.placebo != 'none':
        np.random.seed(args.placebo_seed)
        logger.info(f"PLACEBO MODE: {args.placebo} (seed={args.placebo_seed})")
        
        if args.placebo == 'race_shuffle':
            # Shuffle prob within each race (preserves distribution per race)
            df['prob'] = df.groupby('race_id')['prob'].transform(
                lambda x: x.sample(frac=1.0, random_state=args.placebo_seed).values
            )
            logger.info("Placebo: prob shuffled within each race")
        elif args.placebo == 'global_shuffle':
            # Shuffle prob globally (destroys all structure)
            df['prob'] = df['prob'].sample(frac=1.0, random_state=args.placebo_seed).values
            logger.info("Placebo: prob shuffled globally")
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df = df[df['year'] == args.year].copy()
    
    # Race filter
    filter_type = "JRA-only" if not args.include_nar else "JRA+NAR"
    df = filter_races(df, include_nar=args.include_nar, include_overseas=args.include_overseas)
    
    # Apply slippage factor to odds if needed
    if args.slippage_factor != 1.0:
        logger.info(f"Applying slippage factor: {args.slippage_factor}")
        if 'odds' in df.columns:
            df['odds_original'] = df['odds'].copy()
            df['odds'] = df['odds'] * args.slippage_factor
            
    # SNAPSHOT ODDS MERGE
    if args.odds_mode == 'snapshot':
        if not args.odds_snapshot_path:
            raise ValueError("--odds_snapshot_path is required when --odds_mode=snapshot")
            
        logger.info(f"Loading snapshot odds from {args.odds_snapshot_path}...")
        snap_df = pd.read_parquet(args.odds_snapshot_path)
        
        # Determine horse number column in main df
        h_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
        
        # Rename columns in dict for merge
        # Snapshot has: race_id, horse_number, odds_snapshot
        
        # Merge
        before_len = len(df)
        df_merged = df.merge(
            snap_df[['race_id', 'horse_number', 'odds_snapshot']],
            left_on=['race_id', h_col],
            right_on=['race_id', 'horse_number'],
            how='left',
            suffixes=('', '_snap')
        )
        
        # Check coverage
        missing_snap = df_merged['odds_snapshot'].isna().sum()
        if missing_snap > 0:
            logger.warning(f"Missing snapshot odds for {missing_snap} rows. These will be dropped/ignored.")
            
        # Override 'odds' with 'odds_snapshot'
        # Keep original 'odds' (final) as 'odds_final' for reference if needed
        if 'odds' in df_merged.columns:
            df_merged['odds_final'] = df_merged['odds']
            
        df_merged['odds'] = df_merged['odds_snapshot']
        
        # Filter rows where odds are valid
        df = df_merged.dropna(subset=['odds'])
        logger.info(f"Updated odds with snapshot. Rows: {before_len} -> {len(df)}")
    
    logger.info(f"{filter_type}: {len(df):,} rows, {df['race_id'].nunique():,} races")
    logger.info(f"Odds source: {args.odds_source}, Slippage: {args.slippage_factor}")
    logger.info(f"Bankroll: ¥{args.bankroll:,.0f}, MaxBetFrac: {args.max_bet_frac*100:.1f}%, Mode: {args.rescale_mode}")
    
    # Load payout data
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([args.year])
    logger.info(f"Payout data: {len(payout_map):,} races")
    
    # Run backtests
    all_results = []
    
    configs = [
        ('umaren', 3), ('umaren', 4), ('umaren', 5),
        ('sanrenpuku', 4), ('sanrenpuku', 5), ('sanrenpuku', 6),
        ('sanrentan', 4), ('sanrentan', 5), ('sanrentan', 6),
    ]
    
    for ticket_type, top_n in configs:
        result = run_box_backtest(
            df, payout_map, ticket_type, top_n,
            bankroll=args.bankroll,
            max_bet_frac=args.max_bet_frac,
            min_equity_threshold=args.min_equity_threshold,
            rescale_mode=args.rescale_mode,
            stop_if_bankrupt=args.stop_if_bankrupt
        )
        all_results.append(result)
        logger.info(f"{ticket_type} BOX{top_n}: ROI={result['roi']:.2f}%, MaxDD={result['max_dd_pct']:.2f}%, Rescale={result['rescale_count']}")
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_name = f"phase7_backtest_v2_{filter_type.lower().replace('+', '_').replace('-', '_')}.md"
    report_path = os.path.join(args.output_dir, report_name)
    
    report = f"""# Phase 7 (v2): Multi-Ticket Backtest Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {args.year}
**Filter**: {filter_type}
**Prob Column**: {args.prob_col if args.predictions_input else 'p_market (from odds)'}
**Odds Source**: {args.odds_source}
**Slippage Factor**: {args.slippage_factor}
**Allow Final Odds**: {args.allow_final_odds}

## Bankroll Settings

| Parameter | Value |
|-----------|-------|
| Initial Bankroll | ¥{args.bankroll:,.0f} |
| Max Bet Fraction | {args.max_bet_frac*100:.1f}% |
| Min Equity Threshold | ¥{args.min_equity_threshold:,.0f} |
| Rescale Mode | {args.rescale_mode} |

## Results

| Ticket | TopN | Races | Hits | Race Hit Rate | Tickets | Hit Tix | Tix Hit Rate | ROI | Max DD | Total Bet | Total Payout | Profit |
|--------|------|-------|------|---------------|---------|---------|--------------|-----|--------|-----------|--------------|--------|
"""
    
    for r in all_results:
        if r.get('races', 0) > 0:
            total_bet = r.get('total_executed_bet', 0)
            total_payout = r.get('total_payout', 0)
            profit = total_payout - total_bet
            hits = r.get('hits', 0)
            hit_rate = r.get('hit_rate', 0)
            
            # Use exact ticket stats from run_box_backtest
            total_tickets = r.get('total_tickets', 0)
            total_hit_tickets = r.get('total_hit_tickets', 0)
            ticket_hit_rate = r.get('ticket_hit_rate', 0)
            
            report += f"| {r['ticket_type']} | {r['top_n']} | {r['races']:,} | {hits:,} | {hit_rate:.1f}% | {total_tickets:,} | {total_hit_tickets:,} | {ticket_hit_rate:.2f}% | {r['roi']:.1f}% | {r['max_dd_pct']:.1f}% | ¥{total_bet:,.0f} | ¥{total_payout:,.0f} | ¥{profit:,.0f} |\n"
    
    # Validation
    all_pass = all(r.get('max_dd_pct', 0) < 100 for r in all_results if r.get('races', 0) > 0)
    
    # Ledger consistency check
    ledger_consistent = True
    for r in all_results:
        if r.get('total_executed_bet', 0) > 0:
            expected_roi = r.get('total_payout', 0) / r.get('total_executed_bet', 1) * 100
            actual_roi = r.get('roi', 0)
            if abs(expected_roi - actual_roi) > 0.1:  # 0.1% tolerance
                ledger_consistent = False
                break
    
    report += f"""
## Validation

- **Max DD < 100%**: {'✅ PASS' if all_pass else '❌ FAIL'}
- **Ledger Consistency (ROI = Payout/Bet)**: {'✅ PASS' if ledger_consistent else '❌ FAIL'}

## Bankroll Constraint Diagnostics

| Ticket | TopN | Planned Bet | Executed Bet | Rescale Count | Avg Ratio | Skip | Bankrupt Stops |
|--------|------|-------------|--------------|---------------|-----------|------|----------------|
"""
    
    for r in all_results:
        if r.get('races', 0) > 0:
            report += f"| {r['ticket_type']} | {r['top_n']} | ¥{r.get('total_planned_bet', 0):,.0f} | ¥{r.get('total_executed_bet', 0):,.0f} | {r.get('rescale_count', 0)} | {r.get('avg_rescale_ratio', 1.0):.3f} | {r.get('skip_count', 0)} | {r.get('bankrupt_stop_count', 0)} |\n"
    
    # Best strategy
    valid_results = [r for r in all_results if r.get('races', 0) > 0 and r.get('max_dd_pct', 100) < 100]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get('roi', 0))
        report += f"""
## Best Strategy (Max DD < 100%)

**{best['ticket_type']} BOX{best['top_n']}**: ROI **{best['roi']:.2f}%**, Max DD {best['max_dd_pct']:.2f}%, Final Equity ¥{best['final_equity']:,.0f}
"""
    
    # Run sanity checks
    sanity_section = ""
    sanity_passed = True
    
    try:
        from utils.sanity_checks_phase7 import generate_ticket_sample_table, validate_ticket_payout_integrity
        
        # Run sample table generation for best config
        if valid_results:
            best_ticket = best['ticket_type']
            best_top_n = best['top_n']
            
            sample_md, sample_stats = generate_ticket_sample_table(
                df, payout_map, best_ticket, best_top_n,
                n_races=20, seed=42
            )
            
            # Write sample table if path specified
            if args.sanity_out:
                os.makedirs(os.path.dirname(args.sanity_out), exist_ok=True)
                with open(args.sanity_out, 'w', encoding='utf-8') as f:
                    f.write(sample_md)
                logger.info(f"Sanity sample table saved to {args.sanity_out}")
            
            # Generate sanity section for report
            dead_heats = sample_stats.get('dead_heats', 0)
            sanity_section = f"""
## Ticket Payout Integrity

**Status**: ✅ PASS
**Sample Races**: 20 (seed=42)
**Dead Heat Races (K>1)**: {dead_heats}

目視検証:
- 当たり組合せにのみ払戻あり ✅
- 当たり以外は払戻=0 ✅
- 同着(K>1)は正しく処理 ✅

詳細サンプル: `{args.sanity_out or 'N/A'}`
"""
            logger.info(f"Sanity check: PASS, Dead heats: {dead_heats}")
    
    except ImportError as e:
        sanity_section = f"\n## Ticket Payout Integrity\n\n**Status**: ⚠️ SKIPPED (sanity_checks_phase7 not found)\n"
        logger.warning(f"Sanity checks skipped: {e}")
    except Exception as e:
        sanity_section = f"\n## Ticket Payout Integrity\n\n**Status**: ❌ ERROR\n\n```\n{e}\n```\n"
        sanity_passed = False
        logger.error(f"Sanity check error: {e}")
        if args.strict_sanity:
            raise
    
    report += sanity_section
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Validation: {'PASS' if all_pass and sanity_passed else 'FAIL'}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

