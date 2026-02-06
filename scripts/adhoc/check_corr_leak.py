"""Check feature-target correlation to find leakage"""
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def check_correlation():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Use 2023 for check
    df_check = df[df['year'] == 2023].copy()
    y = (df_check['rank'] == 1).astype(int)
    
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 
                 'is_win', 'is_top2', 'is_top3', 'year', 'rank_str', 'passing_rank']
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in id_cols]
    
    print(f"Checking {len(feature_cols)} features...")
    
    correlations = []
    for col in feature_cols:
        try:
            # Handle object/category by converting to numeric for correlation test
            if df_check[col].dtype.name == 'category' or df_check[col].dtype == 'object':
                series = df_check[col].astype('category').cat.codes
            else:
                series = df_check[col]
            
            corr = np.abs(series.corr(y))
            if corr > 0.5:  # Very high correlation
                correlations.append((col, corr))
        except:
            continue
            
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    if correlations:
        print("\nHigh Correlation Features (>0.5):")
        for col, corr in correlations:
            print(f"  {col:<30}: {corr:.4f}")
    else:
        print("\nNo features with correlation > 0.5 found.")

if __name__ == "__main__":
    check_correlation()
