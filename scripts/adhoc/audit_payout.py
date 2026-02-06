
import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_final_odds(year):
    path = f"data/odds_snapshots/{year}/odds_final.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # create lookup: (race_id, ticket_type, combination) -> odds
    # But checking availability is hard without iterating.
    return df

def audit_payouts(ledger_path, year=2025):
    """
    Re-calculate payouts from ledger and final odds to verify reported revenue.
    """
    logger.info(f"Auditing payouts in {ledger_path}...")
    
    if not os.path.exists(ledger_path):
        logger.error(f"Ledger not found: {ledger_path}")
        return

    df_bets = pd.read_csv(ledger_path)
    logger.info(f"Loaded {len(df_bets)} bets.")
    
    # Load Final Odds
    logger.info(f"Loading Odds Final for {year}...")
    try:
        odds_df = load_final_odds(year)
    except:
        logger.error("Could not load final odds.")
        return
        
    if odds_df.empty:
        logger.error("Odds Final data missing.")
        return
        
    # Pre-process odds for fast lookup
    # key: (str(race_id), ticket_type, str(combination)) -> odds
    odds_df['race_id'] = odds_df['race_id'].astype(str)
    odds_df['combination'] = odds_df['combination'].astype(str)
    
    # Drop duplicates if any
    odds_df = odds_df.drop_duplicates(subset=['race_id', 'ticket_type', 'combination'], keep='last')
    
    # Create dictionary
    # Combining keys into a tuple is faster than multi-index for random access
    extract_cols = zip(odds_df['race_id'], odds_df['ticket_type'], odds_df['combination'])
    payout_map = dict(zip(extract_cols, odds_df['odds']))
    
    # Calculate Payouts
    total_rev = 0
    wins = []
    
    # We also need hit information (Rank).
    # But wait, odds_final usually only contains winning combinations?
    # NO. odd_final usually contains ALL candidates final odds (for analysis).
    # Wait, does `odds_final.parquet` contain ONLY winning checks?
    # Usually `odds_final` is defined as "Payout Results" or "Final Payouts"?
    # If it's "Final Odds Snapshot", it contains odds for ALL horses.
    # We need RACE RESULTS (ranks) to know if we won!
    # Just multiplying by final odds assumes we WON? No.
    # We need to know IF we won.
    
    # The ledger doesn't have Result.
    # So we need to Load OOF (Target) to check results.
    logger.info("Loading OOF (Truth Data)...")
    oof_path = 'data/predictions/v13_oof_2024_2025_with_odds_features.parquet'
    df_oof = pd.read_parquet(oof_path)
    df_oof['race_id'] = df_oof['race_id'].astype(str)
    
    # Prepare Rank Map
    # race_id -> {horse_number: rank}
    # race_id -> {horse_number: frame_number}
    rank_map = {}
    frame_map = {}
    
    # optimize: filter relevant races
    rids = df_bets['race_id'].astype(str).unique()
    df_sub = df_oof[df_oof['race_id'].isin(rids)]
    
    for rid, grp in df_sub.groupby('race_id'):
        rank_map[rid] = dict(zip(grp['horse_number'], grp['rank']))
        frame_map[rid] = dict(zip(grp['horse_number'], grp['frame_number']))
        
    logger.info("Truth data loaded. Calculating Payouts...")
    
    for idx, row in df_bets.iterrows():
        rid = str(row['race_id'])
        ttype = row['ticket_type']
        combo = str(row['combination'])
        amt = row['amount']
        
        # Check Hit
        h2r = rank_map.get(rid, {})
        h2f = frame_map.get(rid, {})
        
        hit = False
        if ttype == 'win':
            if h2r.get(int(combo), 99) == 1: hit = True
        elif ttype == 'place':
            # Place logic: Top 3 (or 2)
            # Need n_horses to determine limit? 
            # Simplified: Top 3.
            if h2r.get(int(combo), 99) <= 3: hit = True
        elif ttype == 'umaren':
            parts = [int(x) for x in combo.split('-')]
            if h2r.get(parts[0], 99) <= 2 and h2r.get(parts[1], 99) <= 2: hit = True
            
        payout = 0
        if hit:
            final_odds = payout_map.get((rid, ttype, combo), 0.0)
            payout = amt * final_odds
            
        if payout > 0:
            total_rev += payout
            wins.append({
                'race_id': rid,
                'ticket_type': ttype,
                'combination': combo,
                'amount': amt,
                'odds': final_odds,
                'payout': payout
            })
            
    logger.info(f"Total Calculated Revenue: {total_rev:,.0f}")
    
    # Top 10 Wins
    df_wins = pd.DataFrame(wins)
    if not df_wins.empty:
        logger.info("Top 10 Wins:")
        print(df_wins.sort_values('payout', ascending=False).head(10))
    else:
        logger.info("No wins found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ledger', type=str)
    args = parser.parse_args()
    
    audit_payouts(args.ledger)
