import pandas as pd
import os

def main():
    path = "data/processed/preprocessed_data_v11.parquet"
    if not os.path.exists(path):
        print("Parquet not found.")
        return
        
    df = pd.read_parquet(path, columns=['race_id', 'horse_number', 'date', 'last_nige_rate', 'sire_heavy_win_rate'])
    print(f"Loaded {len(df)} rows.")
    
    # Check last_nige_rate
    print("\n[last_nige_rate]")
    nz = df[df['last_nige_rate'] > 0]
    print(f"Non-zero count: {len(nz)}")
    if not nz.empty:
        print(nz.head())
        print("Max:", df['last_nige_rate'].max())
    else:
        print("All zeros.")
        
    # Check sire_heavy_win_rate
    print("\n[sire_heavy_win_rate]")
    nz_sire = df[df['sire_heavy_win_rate'] > 0]
    print(f"Non-zero count: {len(nz_sire)}")
    if not nz_sire.empty:
        print(nz_sire.head())
        print("Max:", df['sire_heavy_win_rate'].max())
    else:
        print("All zeros.")

if __name__ == "__main__":
    main()
