"""
Phase 8: Win (単勝) EV Strategy Backtest

This script properly tests the value of time-series odds by:
1. Using snapshot-recalculated predictions (prob_residual_softmax_snap)
2. Calculating EV = prob × odds for win bets
3. Buying horses with EV > threshold
4. Comparing Final Odds EV vs Snapshot Odds EV

This is the correct way to evaluate "Operational ROI" with pre-race odds.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import argparse
from pathlib import Path

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_win_payouts(engine, year=2025):
    """Load win (単勝) payout data from jvd_hr"""
    query = f"""
    SELECT 
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
        haraimodoshi_tansho_1a as win_uma_1,
        haraimodoshi_tansho_1b as win_payout_1,
        haraimodoshi_tansho_2a as win_uma_2,
        haraimodoshi_tansho_2b as win_payout_2,
        haraimodoshi_tansho_3a as win_uma_3,
        haraimodoshi_tansho_3b as win_payout_3
    FROM jvd_hr
    WHERE kaisai_nen = '{year}'
    """
    df = pd.read_sql(query, engine)
    
    # Parse payouts (stored as zero-padded string, e.g., "000000250" = ¥250)
    print(f"Sample win payouts: {df[['win_uma_1', 'win_payout_1']].head()}")
    
    # Convert to dictionary: race_id -> {horse_number: payout}
    payout_map = {}
    for _, row in df.iterrows():
        race_id = row['race_id']
        payout_map[race_id] = {}
        
        for i in [1, 2, 3]:  # Up to 3 winners (dead heat)
            uma = row[f'win_uma_{i}']
            payout = row[f'win_payout_{i}']
            if pd.notna(uma) and pd.notna(payout):
                try:
                    uma_int = int(uma)
                    payout_int = int(payout)  # Already in yen per 100 yen bet
                    if uma_int > 0 and payout_int > 0:
                        payout_map[race_id][uma_int] = payout_int
                except:
                    pass
    
    return payout_map

def run_win_ev_backtest(df, payout_map, prob_col, odds_col, ev_threshold=1.0, bet_unit=100):
    """
    Run Win (単勝) EV strategy backtest
    
    Args:
        df: DataFrame with predictions and odds
        payout_map: dict of race_id -> {horse_number: payout}
        prob_col: Column name for probability (e.g., 'prob_residual_softmax' or 'prob_residual_softmax_snap')
        odds_col: Column name for odds (e.g., 'odds' or 'odds_snapshot')
        ev_threshold: Minimum EV to bet
        bet_unit: Bet amount per horse
    
    Returns:
        Results dict
    """
    results = []
    total_bet = 0
    total_payout = 0
    total_hits = 0
    total_bets = 0
    
    # Calculate EV
    df = df.copy()
    df['ev'] = df[prob_col] * df[odds_col]
    
    # Filter to horses with EV > threshold
    df_bet = df[df['ev'] > ev_threshold]
    
    # Get horse_number from preprocessed data if needed
    if 'horse_number' not in df_bet.columns:
        # Load mapping
        preproc = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
        preproc_2025 = preproc[preproc['race_id'].astype(str).str.startswith('2025')][['race_id', 'horse_id', 'horse_number']]
        df_bet = df_bet.merge(preproc_2025, on=['race_id', 'horse_id'], how='left')
    
    # Process each bet
    for race_id in df_bet['race_id'].unique():
        race_bets = df_bet[df_bet['race_id'] == race_id]
        
        if race_id not in payout_map:
            continue
        
        race_payouts = payout_map[race_id]
        
        for _, row in race_bets.iterrows():
            horse_num = int(row['horse_number']) if pd.notna(row['horse_number']) else None
            if horse_num is None:
                continue
            
            total_bet += bet_unit
            total_bets += 1
            
            if horse_num in race_payouts:
                payout = race_payouts[horse_num] * (bet_unit / 100)  # Scale to bet unit
                total_payout += payout
                total_hits += 1
                results.append({
                    'race_id': race_id,
                    'horse_number': horse_num,
                    'bet': bet_unit,
                    'payout': payout,
                    'hit': 1,
                    'prob': row[prob_col],
                    'odds': row[odds_col],
                    'ev': row['ev']
                })
            else:
                results.append({
                    'race_id': race_id,
                    'horse_number': horse_num,
                    'bet': bet_unit,
                    'payout': 0,
                    'hit': 0,
                    'prob': row[prob_col],
                    'odds': row[odds_col],
                    'ev': row['ev']
                })
    
    roi = (total_payout / total_bet * 100) if total_bet > 0 else 0
    hit_rate = (total_hits / total_bets * 100) if total_bets > 0 else 0
    
    return {
        'total_bets': total_bets,
        'total_bet': total_bet,
        'total_payout': total_payout,
        'profit': total_payout - total_bet,
        'roi': roi,
        'hits': total_hits,
        'hit_rate': hit_rate,
        'results': pd.DataFrame(results)
    }

def main():
    parser = argparse.ArgumentParser(description="Win EV Strategy Backtest")
    parser.add_argument('--ev_threshold', type=float, default=1.0)
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 8: WIN (単勝) EV STRATEGY BACKTEST")
    print("=" * 70)
    
    # Load data
    engine = get_db_engine()
    
    print("\nLoading payout data...")
    payout_map = load_win_payouts(engine, 2025)
    print(f"Loaded payouts for {len(payout_map)} races")
    
    print("\nLoading predictions (snapshot-recalculated)...")
    pred = pd.read_parquet('data/predictions/v13_market_residual_2025_snapshot_recalc.parquet')
    print(f"Predictions: {len(pred)} rows, {pred['race_id'].nunique()} races")
    
    # We have both prob_residual_softmax (final) and prob_residual_softmax_snap (snapshot)
    # And both odds (final) and odds_snapshot
    
    print(f"\nEV Threshold: {args.ev_threshold}")
    
    # Scenario 1: Final Odds + Final Prob (Baseline)
    print("\n--- Scenario 1: Final Odds + Final Prob ---")
    result_final = run_win_ev_backtest(
        pred, payout_map,
        prob_col='prob_residual_softmax',
        odds_col='odds',
        ev_threshold=args.ev_threshold
    )
    print(f"Bets: {result_final['total_bets']:,}")
    print(f"Total Bet: ¥{result_final['total_bet']:,.0f}")
    print(f"Total Payout: ¥{result_final['total_payout']:,.0f}")
    print(f"ROI: {result_final['roi']:.2f}%")
    print(f"Hit Rate: {result_final['hit_rate']:.2f}%")
    
    # Scenario 2: Snapshot Odds + Snapshot Prob (Operational)
    print("\n--- Scenario 2: Snapshot Odds + Snapshot Prob ---")
    result_snap = run_win_ev_backtest(
        pred, payout_map,
        prob_col='prob_residual_softmax_snap',
        odds_col='odds_snapshot',
        ev_threshold=args.ev_threshold
    )
    print(f"Bets: {result_snap['total_bets']:,}")
    print(f"Total Bet: ¥{result_snap['total_bet']:,.0f}")
    print(f"Total Payout: ¥{result_snap['total_payout']:,.0f}")
    print(f"ROI: {result_snap['roi']:.2f}%")
    print(f"Hit Rate: {result_snap['hit_rate']:.2f}%")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Final':<15} {'Snapshot':<15} {'Diff':<15}")
    print("-" * 70)
    print(f"{'Bets':<20} {result_final['total_bets']:<15,} {result_snap['total_bets']:<15,} {result_snap['total_bets'] - result_final['total_bets']:<15,}")
    print(f"{'ROI (%)':<20} {result_final['roi']:<15.2f} {result_snap['roi']:<15.2f} {(result_snap['roi'] - result_final['roi']):<15.2f}")
    print(f"{'Hit Rate (%)':<20} {result_final['hit_rate']:<15.2f} {result_snap['hit_rate']:<15.2f} {(result_snap['hit_rate'] - result_final['hit_rate']):<15.2f}")
    
    # Save results
    out_dir = Path('reports/phase8_snapshot/win_ev')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_final['results'].to_parquet(out_dir / 'ledger_final.parquet')
    result_snap['results'].to_parquet(out_dir / 'ledger_snapshot.parquet')
    
    print(f"\nLedgers saved to {out_dir}")

if __name__ == '__main__':
    main()
