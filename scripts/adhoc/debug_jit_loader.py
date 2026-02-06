
import sys
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.loader import JraVanDataLoader

def check_today_load():
    loader = JraVanDataLoader()
    target_date = "2026-01-05" # As per user session
    
    print(f"Checking load for {target_date}...")
    df = loader.load(history_start_date=target_date, end_date=target_date)
    
    print(f"Loaded {len(df)} rows.")
    if df.empty:
        print("DF is empty!")
        return

    races = df['race_id'].unique()
    print(f"Unique Races ({len(races)}):")
    for rid in races:
        print(f" - {rid}")

if __name__ == "__main__":
    check_today_load()
