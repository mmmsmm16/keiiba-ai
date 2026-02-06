
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_PATH = "data/processed/base_features_all.parquet"

def inspect_odds():
    print("Loading base_features_all.parquet...")
    df = pd.read_parquet(DATA_PATH)
    
    print(f"\nTotal Records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'odds' not in df.columns:
        print("CRITICAL: 'odds' column not found!")
        return

    # Check basic stats
    print("\n=== Odds Statistics ===")
    print(df['odds'].describe())
    
    # Check NaN
    n_nan = df['odds'].isna().sum()
    print(f"\nNaN count: {n_nan} ({n_nan/len(df):.2%})")
    
    # Check negatives or zeros
    n_zero = (df['odds'] == 0).sum()
    n_neg = (df['odds'] < 0).sum()
    print(f"Zero count: {n_zero}")
    print(f"Negative count: {n_neg}")
    
    # Check for integers (payout?) vs floats (odds)
    # If standard odds, mostly floats like 3.5, 12.8, etc.
    # If payout, mostly integers > 100.
    sample = df['odds'].dropna().head(10)
    print("\n=== Sample Values ===")
    print(sample)
    
    # Check Winning Horses Odds
    if 'rank' in df.columns:
        print("\n=== Winning Horses (Rank=1) Odds Stats ===")
        wins = df[df['rank'] == 1]
        print(wins['odds'].describe())
        print("\nDistribution of Winning Odds:")
        print(wins['odds'].quantile([0.01, 0.1, 0.5, 0.9, 0.99]))
        
        # Check low odds winners
        low_odds_wins = wins[wins['odds'] < 1.1]
        print(f"\nWinners with odds < 1.1: {len(low_odds_wins)}")
        if len(low_odds_wins) > 0:
            print(low_odds_wins[['race_id', 'horse_name', 'odds', 'date']].head())
            
        # Check if rank is 0-indexed?
        print(f"\nRank values: {sorted(df['rank'].unique())[:10]} ...")
    else:
        print("Rank column missing, cannot verify distinct winners.")

if __name__ == "__main__":
    inspect_odds()
