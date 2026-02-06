"""
Paper Trade Run: Daily predictions and ticket generation
Phase 9: Real-world operation simulation

Usage:
    docker compose exec app python scripts/paper_trade_run.py \
        --date 2025-12-16 \
        --config config/runtime/paper_trade.yaml
"""

import sys
import os
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from itertools import combinations, permutations
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Skip reason categories
SKIP_REASONS = {
    'missing_odds': 'Missing odds data',
    'scratched_runner': 'Scratched runner in topN',
    'insufficient_bankroll': 'Insufficient bankroll',
    'invalid_race_size': 'Too few runners',
    'missing_features': 'Missing required features',
    'missing_predictions': 'Missing predictions',
    'missing_payouts': 'Missing payout data',
    'other_error': 'Other error'
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def format_combination(horses: List[int], ordered: bool = False) -> str:
    """Format horse numbers to official payout key format"""
    if ordered:
        return "".join([f"{h:02}" for h in horses])
    else:
        return "".join([f"{h:02}" for h in sorted(horses)])


def load_models(config: Dict) -> List[lgb.Booster]:
    """Load LightGBM models for inference"""
    models = []
    model_dir = config['model']['model_dir']
    
    for fold_file in config['model']['fold_models']:
        model_path = os.path.join(model_dir, fold_file)
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            models.append(model)
            logger.info(f"Loaded model: {fold_file}")
        else:
            logger.warning(f"Model not found: {model_path}")
    
    if not models:
        raise FileNotFoundError("No models found")
    
    return models


def run_inference(df: pd.DataFrame, models: List[lgb.Booster], config: Dict) -> pd.DataFrame:
    """Run inference with model ensemble"""
    from scipy.special import expit
    
    # Get feature columns from first model
    feature_cols = models[0].feature_name()
    
    # Prepare features
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} features, filling with 0")
        for col in missing_cols:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    # Ensemble prediction
    preds = []
    for model in models:
        pred = model.predict(X)
        preds.append(pred)
    
    # Average predictions
    avg_pred = np.mean(preds, axis=0)
    
    # Convert to probability
    df = df.copy()
    
    # Store raw logit (before sigmoid) for more clear difference display
    df['raw_score'] = avg_pred
    
    df['prob_residual_raw'] = expit(avg_pred)
    
    # Calculate market probability from odds
    if 'odds' in df.columns:
        df['p_market_raw'] = 1.0 / df['odds'].replace(0, np.nan)
        df['p_market'] = df.groupby('race_id')['p_market_raw'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else np.nan
        )
    else:
        df['p_market'] = np.nan
    
    # Softmax per race
    def softmax_race(group):
        exp_vals = np.exp(group - group.max())
        return exp_vals / exp_vals.sum()
    
    df['prob_residual_softmax'] = df.groupby('race_id')['prob_residual_raw'].transform(softmax_race)
    
    # Calculate edge (model vs market)
    if 'p_market' in df.columns:
        df['edge'] = df['prob_residual_softmax'] - df['p_market']
    
    return df


def generate_tickets(
    race_df: pd.DataFrame,
    ticket_type: str,
    topn: int,
    prob_col: str
) -> Tuple[List[Dict], str]:
    """Generate tickets for a single race"""
    
    # Get top N horses
    if prob_col not in race_df.columns or race_df[prob_col].isna().all():
        return [], 'missing_predictions'
    
    if len(race_df) < topn:
        return [], 'invalid_race_size'
    
    # Sort by probability
    top_horses = race_df.nlargest(topn, prob_col)
    horse_numbers = top_horses['horse_number'].astype(int).tolist()
    
    # Check for valid horse numbers
    if any(h <= 0 or h > 18 for h in horse_numbers):
        return [], 'invalid_race_size'
    
    # Generate combinations/permutations
    if ticket_type == 'umaren':
        tickets = list(combinations(horse_numbers, 2))
        ordered = False
    elif ticket_type == 'sanrenpuku':
        tickets = list(combinations(horse_numbers, 3))
        ordered = False
    elif ticket_type == 'sanrentan':
        tickets = list(permutations(horse_numbers, 3))
        ordered = True
    else:
        return [], 'other_error'
    
    # Convert to ticket records
    ticket_records = []
    for t in tickets:
        ticket_key = format_combination(list(t), ordered=ordered)
        ticket_records.append({
            'horses': list(t),
            'ticket_key': ticket_key,
            'display': '-'.join(map(str, t))
        })
    
    return ticket_records, None


def run_daily_pipeline(date_str: str, config: Dict) -> Dict[str, Any]:
    """Run daily paper trade pipeline"""
    
    from utils.payout_loader import PayoutLoader
    from utils.race_filter import filter_jra_only
    
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    
    logger.info(f"=== Paper Trade Run: {date_str} ===")
    logger.info(f"Strategy: {config['strategy']['ticket']} BOX{config['strategy']['topn']}")
    
    # Setup output directories
    data_dir = Path(config['report']['data_dir']) / date_str
    report_dir = Path(config['report']['out_dir']) / date_str
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base data
    logger.info("Loading base data...")
    base_path = 'data/processed/preprocessed_data_v11.parquet'
    df = pd.read_parquet(base_path)
    
    # Filter to target date
    # Use 'race_date' or 'date' column if available, otherwise extract from race_id
    if 'race_date' in df.columns:
        df['_date'] = pd.to_datetime(df['race_date'])
    elif 'date' in df.columns:
        df['_date'] = pd.to_datetime(df['date'])
    else:
        # race_id format: YYYYMMDDCCRRNN (JRA) or varies
        # Filter by string prefix match on race_id
        target_prefix = date_str.replace('-', '')  # YYYYMMDD
        df = df[df['race_id'].astype(str).str[:8] == target_prefix]
        if len(df) == 0:
            logger.warning(f"No races found for {date_str}")
            return {'status': 'no_races', 'date': date_str}
        df['_date'] = pd.to_datetime(target_prefix, format='%Y%m%d')
    
    # Filter by date
    if '_date' in df.columns:
        df = df[df['_date'].dt.strftime('%Y-%m-%d') == date_str]
    
    if len(df) == 0:
        logger.warning(f"No races found for {date_str}")
        return {'status': 'no_races', 'date': date_str}
    
    # Apply JRA filter
    if config['data']['jra_only']:
        df = filter_jra_only(df)
    
    logger.info(f"Found {df['race_id'].nunique()} races, {len(df)} runners")
    
    # Load models and run inference
    logger.info("Loading models...")
    models = load_models(config)
    
    logger.info("Running inference...")
    df = run_inference(df, models, config)
    
    prob_col = config['model']['prob_col']
    
    # Load payout data
    logger.info("Loading payout data...")
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([year])
    
    # Strategy parameters
    ticket_type = config['strategy']['ticket']
    topn = config['strategy']['topn']
    bet_unit = config['strategy']['bet_unit']
    
    # Risk parameters
    bankroll = config['risk']['initial_bankroll']
    max_bet_frac = config['risk']['max_bet_frac']
    max_bet_per_race = config['risk'].get('max_bet_per_race', float('inf'))
    
    # Process each race
    all_predictions = []
    all_tickets = []
    all_ledger = []
    skip_reasons = defaultdict(list)
    
    current_equity = bankroll
    
    for race_id in sorted(df['race_id'].unique()):
        race_df = df[df['race_id'] == race_id].copy()
        
        # Generate tickets
        tickets, skip_reason = generate_tickets(race_df, ticket_type, topn, prob_col)
        
        if skip_reason:
            skip_reasons[skip_reason].append(race_id)
            continue
        
        # Calculate stake
        n_tickets = len(tickets)
        planned_bet = n_tickets * bet_unit
        max_allowed = min(current_equity * max_bet_frac, max_bet_per_race)
        
        if planned_bet > max_allowed:
            if planned_bet > current_equity:
                skip_reasons['insufficient_bankroll'].append(race_id)
                continue
            # Scale down
            scale_factor = max_allowed / planned_bet
            actual_stake = int(bet_unit * scale_factor / 10) * 10  # Round to 10 yen
            actual_stake = max(100, actual_stake)  # Minimum 100 yen
        else:
            actual_stake = bet_unit
        
        total_stake = n_tickets * actual_stake
        
        # Save predictions
        for _, row in race_df.iterrows():
            pred_record = {
                'race_id': race_id,
                'horse_id': row['horse_id'],
                'horse_number': row['horse_number'],
                prob_col: row[prob_col],
                'rank_pred': race_df[prob_col].rank(ascending=False)[row.name]
            }
            # Add horse_name if available
            if 'horse_name' in row.index:
                pred_record['horse_name'] = row['horse_name']
            # Add actual rank if available (for settled data)
            if 'rank' in row.index:
                pred_record['rank'] = row['rank']
            # Add market probability and odds
            if 'p_market' in row.index:
                pred_record['p_market'] = row['p_market']
            if 'odds' in row.index:
                pred_record['odds'] = row['odds']
            if 'edge' in row.index:
                pred_record['edge'] = row['edge']
            # Add raw score (before softmax)
            if 'raw_score' in row.index:
                pred_record['raw_score'] = row['raw_score']
            all_predictions.append(pred_record)
        
        # Save tickets (pending settlement)
        for t in tickets:
            ticket_record = {
                'race_id': race_id,
                'ticket_type': ticket_type,
                'ticket_key': t['ticket_key'],
                'horses': str(t['horses']),
                'display': t['display'],
                'stake': actual_stake,
                'status': 'pending'
            }
            all_tickets.append(ticket_record)
            
            # Ledger entry (payout to be filled in settle)
            ledger_record = {
                'race_id': race_id,
                'ticket_type': ticket_type,
                'ticket_key': t['ticket_key'],
                'horses': str(t['horses']),
                'display': t['display'],
                'stake': actual_stake,
                'payout': None,  # To be filled in settle
                'is_hit': None,
                'skip_reason': None
            }
            all_ledger.append(ledger_record)
        
        # Update equity (tentative - before knowing results)
        current_equity -= total_stake
    
    # Save outputs
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_parquet(data_dir / 'predictions.parquet', index=False)
        logger.info(f"Saved predictions: {len(pred_df)} rows")
    
    if all_tickets:
        tickets_df = pd.DataFrame(all_tickets)
        tickets_df.to_parquet(data_dir / 'tickets.parquet', index=False)
        logger.info(f"Saved tickets: {len(tickets_df)} rows")
    
    if all_ledger:
        ledger_df = pd.DataFrame(all_ledger)
        ledger_df.to_parquet(data_dir / 'ledger.parquet', index=False)
        logger.info(f"Saved ledger: {len(ledger_df)} rows")
    
    # Summary stats
    summary = {
        'date': date_str,
        'status': 'pending_settlement',
        'config_snapshot': config,
        'stats': {
            'total_races': df['race_id'].nunique(),
            'processed_races': len(set(t['race_id'] for t in all_tickets)) if all_tickets else 0,
            'total_tickets': len(all_tickets),
            'total_stake': sum(t['stake'] for t in all_tickets) if all_tickets else 0,
            'remaining_equity': current_equity
        },
        'skip_reasons': {k: len(v) for k, v in skip_reasons.items()},
        'skip_samples': {k: v[:10] for k, v in skip_reasons.items()}
    }
    
    # Save summary
    import json
    with open(report_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Generate preliminary report
    report = generate_run_report(summary, config)
    with open(report_dir / 'run_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Run complete. Total stake: ¥{summary['stats']['total_stake']:,}")
    logger.info(f"Skipped races: {dict(summary['skip_reasons'])}")
    
    return summary


def generate_run_report(summary: Dict, config: Dict) -> str:
    """Generate preliminary run report"""
    
    stats = summary['stats']
    skip = summary['skip_reasons']
    
    report = f"""# Paper Trade Run Report

**Date**: {summary['date']}
**Status**: Pending Settlement

## Configuration

| Parameter | Value |
|-----------|-------|
| Ticket Type | {config['strategy']['ticket']} |
| BOX Size | {config['strategy']['topn']} |
| Odds Source | {config['data']['odds_source']} |
| Slippage Factor | {config['data']['slippage_factor']} |
| Initial Bankroll | ¥{config['risk']['initial_bankroll']:,} |
| Max Bet Fraction | {config['risk']['max_bet_frac']*100:.1f}% |

## Run Summary

| Metric | Value |
|--------|-------|
| Total Races | {stats['total_races']} |
| Processed Races | {stats['processed_races']} |
| Total Tickets | {stats['total_tickets']} |
| Total Stake | ¥{stats['total_stake']:,} |
| Remaining Equity | ¥{stats['remaining_equity']:,.0f} |

## Skipped Races by Reason

| Reason | Count |
|--------|-------|
"""
    
    for reason, count in skip.items():
        report += f"| {SKIP_REASONS.get(reason, reason)} | {count} |\n"
    
    if not skip:
        report += "| (none) | 0 |\n"
    
    report += f"""
---

> **Note**: Results pending. Run `paper_trade_settle.py` after race results are available.
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Paper Trade Run")
    parser.add_argument('--date', type=str, required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, 
                        default='config/runtime/paper_trade.yaml',
                        help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run pipeline
    result = run_daily_pipeline(args.date, config)
    
    logger.info("Done!")
    return result


if __name__ == "__main__":
    main()
