import pandas as pd
import numpy as np

# Load predictions
v5_df = pd.read_csv('reports/predictions_v5_2025.csv')
v4_df = pd.read_csv('reports/predictions_v4_2025.csv')

# Ensure race_id is string for merging
v5_df['race_id'] = v5_df['race_id'].astype(str)
v4_df['race_id'] = v4_df['race_id'].astype(str)

# Merge rank from v5 to v4 (v4 doesn't have rank)
if 'rank' not in v4_df.columns and 'rank' in v5_df.columns:
    actuals = v5_df[['race_id', 'horse_number', 'rank', 'surface']].drop_duplicates()
    v4_df = pd.merge(v4_df, actuals, on=['race_id', 'horse_number'], how='left', suffixes=('', '_v5'))
    if 'surface_v5' in v4_df.columns:
        v4_df['surface'] = v4_df['surface'].fillna(v4_df['surface_v5'])
        v4_df = v4_df.drop(columns=['surface_v5'])

# Filter for Turf only
v5_turf = v5_df[v5_df['surface'] == '芝'].copy()
v4_turf = v4_df[v4_df['surface'] == '芝'].copy()

print(f"Turf Races: v5={len(v5_turf['race_id'].unique())}, v4={len(v4_turf['race_id'].unique())}")

# Calculate pred_rank if not exists
if 'pred_rank' not in v5_turf.columns:
    v5_turf['pred_rank'] = v5_turf.groupby('race_id')['score'].rank(ascending=False, method='min')
if 'pred_rank' not in v4_turf.columns:
    v4_turf['pred_rank'] = v4_turf.groupby('race_id')['score'].rank(ascending=False, method='min')

def calc_stats(df, model_name):
    stats = []
    valid_df = df[df['rank'].notna() & (df['rank'] > 0)].copy()
    
    for r in range(1, 6):
        target = valid_df[valid_df['pred_rank'] == r]
        count = len(target)
        if count == 0:
            stats.append({'Model': model_name, 'Rank': r, 'Win': '0%', 'Ren': '0%', 'Fuku': '0%', 'N': 0})
            continue
        wins = len(target[target['rank'] == 1])
        rens = len(target[target['rank'] <= 2])
        fukus = len(target[target['rank'] <= 3])
        stats.append({
            'Model': model_name, 'Rank': r, 
            'Win': f'{wins/count:.1%}', 
            'Ren': f'{rens/count:.1%}', 
            'Fuku': f'{fukus/count:.1%}',
            'N': count
        })
    return pd.DataFrame(stats)

stats_v5 = calc_stats(v5_turf, 'v5 (JRA-Only)')
stats_v4 = calc_stats(v4_turf, 'v4 (Full)')

print()
print('=== 🌿 芝レース限定 パフォーマンス比較 ===')
print()

for rank in range(1, 6):
    r5 = stats_v5[stats_v5['Rank'] == rank].iloc[0]
    r4 = stats_v4[stats_v4['Rank'] == rank].iloc[0]
    print(f'[Rank {rank}]')
    print(f"  v4: Win={r4['Win']} | Ren={r4['Ren']} | Fuku={r4['Fuku']} (N={r4['N']})")
    print(f"  v5: Win={r5['Win']} | Ren={r5['Ren']} | Fuku={r5['Fuku']} (N={r5['N']})")
    print()
