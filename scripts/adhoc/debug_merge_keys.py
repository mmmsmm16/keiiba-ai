
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

def debug_merge():
    print("Loading base features (2024)...")
    base = pd.read_parquet("data/processed/base_features_all.parquet")
    base['date'] = pd.to_datetime(base['date'])
    df_base = base[base['date'].dt.year == 2024].copy()
    
    print("Loading Q1 2024...")
    df_q1 = pd.read_parquet("data/temp_q1/year_2024.parquet")
    
    print(f"\nBase 2024 Records: {len(df_base)}")
    print(f"Q1 2024 Records: {len(df_q1)}")
    
    # Check Race ID formats
    print("\n--- Race ID Sample (Base) ---")
    print(df_base['race_id'].head().tolist())
    print("Type:", df_base['race_id'].dtype)
    
    print("\n--- Race ID Sample (Q1) ---")
    print(df_q1['race_id'].head().tolist())
    print("Type:", df_q1['race_id'].dtype)
    
    # Check Odds in Q1
    print("\n--- Q1 Odds Stats ---")
    print(df_q1['odds'].describe())
    q1_zeros = (df_q1['odds'] == 0).sum()
    print(f"Q1 Zeros: {q1_zeros} / {len(df_q1)} ({q1_zeros/len(df_q1):.2%})")
    
    # Check overlap
    base_ids = set(df_base['race_id'].astype(str))
    q1_ids = set(df_q1['race_id'].astype(str))
    
    overlap = base_ids.intersection(q1_ids)
    print(f"\nCommon Race IDs: {len(overlap)}")
    print(f"Unique Base IDs: {len(base_ids)}")
    print(f"Unique Q1 IDs: {len(q1_ids)}")
    
    if len(overlap) < len(base_ids):
        print("WARNING: Q1 missing some races present in Base.")
        missing = list(base_ids - q1_ids)[:5]
        print(f"Sample missing in Q1: {missing}")

if __name__ == "__main__":
    debug_merge()
