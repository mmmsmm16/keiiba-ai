"""Check if 'rank' or 'time' (current race result) is in model features"""
import pandas as pd
from pathlib import Path

dataset_path = Path('data/processed/lgbm_datasets_v11.pkl')
datasets = pd.read_pickle(dataset_path)
model_features = datasets['train']['X'].columns.tolist()

# Check for exact 'rank' or 'time' (not lag1_rank etc.)
print("Checking for current race result columns in model features...")

exact_forbidden = ['rank', 'time', 'rank_str', 'raw_time', 'rank_norm']
found = [c for c in model_features if c in exact_forbidden]

if found:
    print(f"❌ LEAKAGE DETECTED: {found}")
else:
    print("✅ No current-race result columns in model features")

# Also check for any column that starts with these (not lag1_)
possible_leak = []
for c in model_features:
    if c in exact_forbidden:
        possible_leak.append(c)
    elif c.startswith('rank') and not c.startswith('rank_') and 'lag' not in c and 'mean' not in c:
        possible_leak.append(c)
    elif c.startswith('time') and not c.startswith('time_') and 'lag' not in c and 'mean' not in c:
        possible_leak.append(c)

if possible_leak:
    print(f"Possible leaks: {possible_leak}")
else:
    print("No suspicious columns found")

print(f"\nTotal model features: {len(model_features)}")
