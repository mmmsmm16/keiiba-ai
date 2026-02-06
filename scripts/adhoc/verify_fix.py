import pandas as pd
import os

def main():
    path = "data/processed/preprocessed_data_v11.parquet"
    if not os.path.exists(path):
        print("Data file not found (rebuild likely not done)")
        return

    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")
    
    targets = ['last_nige_rate', 'avg_first_corner_norm', 'is_sole_leader', 'sire_heavy_win_rate']
    
    print("\nVerifying Targets:")
    for col in targets:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            nulls = df[col].isnull().sum()
            
            print(f"[{col}]")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std : {std_val:.4f}")
            print(f"  Min/Max: {min_val} / {max_val}")
            print(f"  Nulls: {nulls}")
            
            if std_val == 0:
                print("  STATUS: FAIL (Constant)")
            elif nulls == len(df):
                print("  STATUS: FAIL (All Null)")
            else:
                print("  STATUS: PASS")
        else:
            print(f"[{col}] Not Found in DataFrame")
            
    # Also check race_pace_level_3f
    if 'race_pace_level_3f' in df.columns:
        null_rate = df['race_pace_level_3f'].isnull().mean()
        print(f"\n[race_pace_level_3f] Null Rate: {null_rate:.2%}")
        if null_rate < 0.1: # Allow some nulls for missing data
            print("  STATUS: PASS")
        else:
            print("  STATUS: FAIL (High nulls)")

if __name__ == "__main__":
    main()
