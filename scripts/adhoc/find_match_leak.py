"""Find columns with high binary matching rate with target"""
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def find_match_leak():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    y = (df['rank'] == 1).astype(int)
    
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 
                 'is_win', 'is_top2', 'is_top3', 'year', 'rank_str', 'passing_rank']
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in id_cols]
    
    print(f"Checking {len(feature_cols)} features for match rate...")
    
    leak_found = False
    for col in feature_cols:
        try:
            # Check 0/1 columns
            if df[col].nunique() == 2:
                # Align 0/1 (sometimes it might be 1/-1 or something)
                vals = sorted(df[col].unique())
                normalized = (df[col] == vals[1]).astype(int)
                
                match_rate = (normalized == y).mean()
                if match_rate > 0.95:
                    print(f"  FOUND: {col} has match rate {match_rate:.4%} with target!")
                    leak_found = True
        except:
            continue
    
    if not leak_found:
        print("No high match rate binary features found.")

if __name__ == "__main__":
    find_match_leak()
