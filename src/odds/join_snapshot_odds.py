import pandas as pd
import numpy as np
import os
from pathlib import Path

def join_snapshot_odds(candidates_df, year, snapshot_mode='T-10', snapshot_dir='data/odds_snapshots'):
    """
    Join candidates with odds snapshots.
    
    Args:
        candidates_df (pd.DataFrame): [race_id, ticket_type, combination, p_ticket]
        year (int): Year of races (to load correct partitioned parquet)
        snapshot_mode (str): 'T-60', 'T-30', 'T-10', 'T-5'
        snapshot_dir (str): Base dir
        
    Returns:
        pd.DataFrame: candidates_df with 'odds' and 'ev' columns.
                      'odds' is NaN if not found or ticket type not supported in chunks.
    """
    if candidates_df.empty:
        return candidates_df
        
    # Load Odds Snapshot
    path = Path(snapshot_dir) / str(year) / f"odds_{snapshot_mode}.parquet"
    if not path.exists():
        # Fallback: Maybe Win only file if old script? No, we rebuilt.
        # If missing, return with NaN odds
        print(f"Warning: Odds snapshot not found at {path}")
        candidates_df['odds'] = np.nan
        candidates_df['ev'] = np.nan
        return candidates_df
        
    try:
        odds_df = pd.read_parquet(path)
        # Columns: race_id, ticket_type, combination, odds, ninki, ...
        # Ensure types match for join
        odds_df['race_id'] = odds_df['race_id'].astype(str)
        odds_df['combination'] = odds_df['combination'].astype(str)
        
        # Prepare candidates types
        candidates_df['race_id'] = candidates_df['race_id'].astype(str)
        candidates_df['combination'] = candidates_df['combination'].astype(str)
        
        # Merge
        # Left join on (race_id, ticket_type, combination)
        merged = pd.merge(
            candidates_df, 
            odds_df[['race_id', 'ticket_type', 'combination', 'odds', 'ninki']],
            on=['race_id', 'ticket_type', 'combination'],
            how='left'
        )
        
        # Calculate EV
        # EV = p * odds
        merged['ev'] = merged['p_ticket'] * merged['odds']
        
        return merged
        
    except Exception as e:
        print(f"Error joining odds: {e}")
        candidates_df['odds'] = np.nan
        return candidates_df
