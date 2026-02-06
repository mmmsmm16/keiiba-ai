import logging
import sys
import os
sys.path.append(os.getcwd())
from src.odds.pckeiba_loader import PCKeibaLoader

logging.basicConfig(level=logging.INFO)

def main():
    loader = PCKeibaLoader()
    # Use a knwon date that likely has data (e.g. 2025-01-05 based on previous context)
    # Or fetch today? 
    # Let's try 2025-01-05 first as we know data exists in snapshots, maybe in DB too?
    # If DB is realtime only, it might be empty.
    # But user said "apd_sokuho" tables. usually they keep some history or latest.
    # Let's try 2025-01-05.
    
    date_str = "2025-12-14"
    print(f"Fetching odds for {date_str}...")
    
    # Create dummy race_id_map for verification
    # Need to know which venues were held on 2025-12-14. 
    # Usually Nakayama(06), Chukyo(07), etc.
    # Let's add wildcards or just many entries.
    race_id_map = {}
    for r in range(1, 13):
        # Nakayama (06)
        race_id_map[(2025, 12, 14, '06', r)] = f"2025061214{str(r).zfill(2)}"
        # Chukyo (07)
        race_id_map[(2025, 12, 14, '07', r)] = f"2025071214{str(r).zfill(2)}"
        # Hanshin? (09) - Maybe
        race_id_map[(2025, 12, 14, '09', r)] = f"2025091214{str(r).zfill(2)}"

    df = loader.get_latest_odds(date_str, race_id_map=race_id_map)
    
    if df.empty:
        print("No odds found (Check if race_id_map matches DB venue codes).")
    else:
        print(f"Found {len(df)} odds records.")
        print(df.head())
        print(df['ticket_type'].value_counts())
        
        # Check specific sample
        print("\nSample Win Odds:")
        print(df[df['ticket_type'] == 'win'].head())
        print("\nSample Umaren Odds:")
        print(df[df['ticket_type'] == 'umaren'].head())

if __name__ == "__main__":
    main()
