
import os
import sys
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.loader import InferenceDataLoader
import pandas as pd

def main():
    print("Initializing Loader...")
    loader = InferenceDataLoader()
    
    target_date = "20251207"
    print(f"Loading data for {target_date}...")
    
    try:
        df = loader.load(target_date=target_date)
        
        if df.empty:
            print("No data found.")
            return

        print(f"Loaded {len(df)} records.")
        print("Columns:", df.columns.tolist())
        
        # Check odds
        if 'odds' in df.columns:
            non_null_odds = df['odds'].count()
            print(f"Non-null odds count: {non_null_odds} / {len(df)}")
            
            if non_null_odds > 0:
                print("Sample Odds:")
                print(df[['race_id', 'horse_number', 'odds', 'popularity']].head(10))
            else:
                print("WARNING: Odds are all NULL.")
        else:
            print("ERROR: 'odds' column missing.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
