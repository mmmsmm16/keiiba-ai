
import pandas as pd
pd.set_option('display.max_columns', None)

def check_2024_odds():
    print("Loading base_features_all.parquet...")
    df = pd.read_parquet("data/processed/base_features_all.parquet")
    
    # Filter 2024
    df['date'] = pd.to_datetime(df['date'])
    df_2024 = df[df['date'].dt.year == 2024]
    
    print(f"\nTotal Records 2024: {len(df_2024)}")
    
    if len(df_2024) == 0:
        print("No records for 2024 found in base_features_all!")
        return

    print("\n=== Odds Statistics (2024) ===")
    print(df_2024['odds'].describe())
    
    n_zero = (df_2024['odds'] == 0).sum()
    print(f"\nZero count 2024: {n_zero} ({n_zero/len(df_2024):.2%})")
    
    wins_2024 = df_2024[df_2024['rank'] == 1]
    n_zero_wins = (wins_2024['odds'] == 0).sum()
    print(f"\nWinning Horses (Rank=1) with Zero Odds: {n_zero_wins} / {len(wins_2024)}")
    
    if n_zero_wins > 0:
        print(wins_2024[wins_2024['odds'] == 0][['race_id', 'horse_name', 'odds', 'date']].head())

if __name__ == "__main__":
    check_2024_odds()
