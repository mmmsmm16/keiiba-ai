
import pandas as pd
import sys

def main():
    files = [
        'data/predictions/v13_oof_2024_clean.parquet',
        'data/predictions/v13_oof_2025_clean.parquet'
    ]
    out = 'data/predictions/v13_oof_2024_2025_with_odds_features.parquet'
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
            print(f"Loaded {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
            return
            
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(out)
    print(f"Saved merged OOF to {out}: {len(merged)} rows")

if __name__ == "__main__":
    main()
