"""Check actual features used in Optuna"""
import pandas as pd

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

print("Loading data...")
df = pd.read_parquet(DATA_PATH)

# Exclude meta columns (same as Optuna script)
meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 
             'is_win', 'is_top2', 'is_top3', 'year']
feature_cols = [c for c in df.columns if c not in meta_cols]

# Also exclude ID columns
id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
feature_cols = [c for c in feature_cols if c not in id_cols]

# Check for DIRECT leakage columns (not rate/stats)
direct_leak = []
for col in feature_cols:
    if col == 'rank_str':  # Direct rank
        direct_leak.append(col)
    elif col.startswith('is_') and ('win' in col.lower() or 'top' in col.lower()):
        direct_leak.append(col)

print(f"\nTotal features: {len(feature_cols)}")
print(f"Direct leakage columns found: {len(direct_leak)}")
for col in direct_leak:
    print(f"  - {col}")

# Check rank_str
if 'rank_str' in feature_cols:
    print(f"\n*** CRITICAL: 'rank_str' is included as a feature! ***")
    print(f"This is the rank as string - DIRECT LEAKAGE")
    
# Check correlation with target
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df_val = df[df['year'] == 2023]
y_val = (df_val['rank'] == 1).astype(int)

# Check if rank_str perfectly predicts target
if 'rank_str' in df_val.columns:
    from sklearn.metrics import roc_auc_score
    # rank_str が '1' なら 1, それ以外なら 0
    rank_str_encoded = (df_val['rank_str'].astype(str) == '1').astype(int)
    try:
        auc = roc_auc_score(y_val, rank_str_encoded)
        print(f"\nAUC using 'rank_str' only: {auc:.4f}")
    except:
        pass
