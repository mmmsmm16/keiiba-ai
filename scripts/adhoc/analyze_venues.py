
import pandas as pd

def analyze_venues():
    print("Loading base_features_all.parquet...")
    df = pd.read_parquet("data/processed/base_features_all.parquet")
    df['date'] = pd.to_datetime(df['date'])
    df_2024 = df[df['date'].dt.year == 2024].copy()
    
    print(f"Total 2024 Records: {len(df_2024)}")
    
    # Extract Venue Code (chars 4-6 of race_id)
    # race_id format: YYYY(4) + Venue(2) + ...
    df_2024['venue_code'] = df_2024['race_id'].astype(str).str[4:6]
    
    # Split by Odds Status
    zeros = df_2024[df_2024['odds'] == 0]
    valid = df_2024[df_2024['odds'] > 0]
    
    print("\n=== Venue Distribution: Zero Odds ===")
    print(zeros['venue_code'].value_counts().sort_index())
    
    print("\n=== Venue Distribution: Valid Odds ===")
    print(valid['venue_code'].value_counts().sort_index())
    
    print("\n(Note: JRA codes are usually 01-10)")

if __name__ == "__main__":
    analyze_venues()
