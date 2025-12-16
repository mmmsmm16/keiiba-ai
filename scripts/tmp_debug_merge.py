"""Compare race_id formats"""
import pandas as pd

print("=== Snapshot Data (T-10m) ===")
snap_df = pd.read_parquet('data/odds_snapshots/2025_win_T-10m_jra_only.parquet')
print(f"Count: {len(snap_df)}")
print(snap_df[['race_id', 'horse_number']].head().to_string())
print(f"race_id dtype: {snap_df['race_id'].dtype}")

print("\n=== Preprocessed Data (v11) ===")
# Read a subset
df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
print(f"race_id dtype: {df['race_id'].dtype}")
# Filter for 2025
if 'year' not in df.columns:
    df['year'] = df['race_id'].astype(str).str[:4].astype(int)
df_2025 = df[df['year'] == 2025]
print(f"2025 Rows: {len(df_2025)}")
h_col = 'umaban' if 'umaban' in df_2025.columns else 'horse_number'
print(df_2025[['race_id', h_col]].head().to_string())

# Check overlap
snap_ids = set(snap_df['race_id'].unique())
proc_ids = set(df_2025['race_id'].unique())
print(f"\nSnapshot unique races: {len(snap_ids)}")
print(f"Processed unique races: {len(proc_ids)}")
print(f"Intersection: {len(snap_ids.intersection(proc_ids))}")

# Show difference sample
diff = list(proc_ids - snap_ids)
if diff:
    print(f"\nExample missing race_id in snapshot: {diff[0]}")
    # Try to find similarity
    # Maybe processed data uses different format?
