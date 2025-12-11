import pickle
import pandas as pd
import os

path = 'experiments/pipeline_fast_test/data/lgbm_datasets.pkl'
if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, 'rb') as f:
    data = pickle.load(f)

cols = list(data['train']['X'].columns)
cols.sort()

print(f"Total Features: {len(cols)}\n")

# 分類して表示を試みる
categories = {
    'Basic': ['race_', 'distance', 'frame', 'horse', 'age', 'sex', 'year', 'month', 'odds', 'weather'],
    'History (Lag)': ['lag'],
    'History (Stat)': ['avg_', 'max_', 'min_', 'total_', 'win_rate', 'roi_'],
    'Category (Jockey/Trainer)': ['jockey_', 'trainer_'],
    'Bloodline': ['sire_', 'mare_'],
    'Relative': ['relative_', 'diff_'],
    'Realtime': ['trend_'],
    'Disadvantage': ['disadv_'],
    'Embedding': ['_emb_']
}

classified = set()

for cat, keywords in categories.items():
    print(f"### {cat}")
    found = []
    for c in cols:
        if c in classified: continue
        for k in keywords:
            if k in c:
                found.append(c)
                classified.add(c)
                break
    for f in found:
        print(f"- {f}")
    print("")

print("### Others")
for c in cols:
    if c not in classified:
        print(f"- {c}")
