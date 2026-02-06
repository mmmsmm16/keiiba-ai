import pandas as pd

v5_df = pd.read_csv('reports/predictions_v5_2025.csv')
v4_df = pd.read_csv('reports/predictions_v4_2025.csv')

v5_df['race_id'] = v5_df['race_id'].astype(str)
v4_df['race_id'] = v4_df['race_id'].astype(str)

if 'rank' not in v4_df.columns and 'rank' in v5_df.columns:
    actuals = v5_df[['race_id', 'horse_number', 'rank', 'surface']].drop_duplicates()
    v4_df = pd.merge(v4_df, actuals, on=['race_id', 'horse_number'], how='left', suffixes=('', '_v5'))

v5_turf = v5_df[v5_df['surface'] == '芝'].copy()
v4_turf = v4_df[v4_df['surface'] == '芝'].copy()

if 'pred_rank' not in v5_turf.columns:
    v5_turf['pred_rank'] = v5_turf.groupby('race_id')['score'].rank(ascending=False, method='min')
if 'pred_rank' not in v4_turf.columns:
    v4_turf['pred_rank'] = v4_turf.groupby('race_id')['score'].rank(ascending=False, method='min')

with open('reports/turf_comparison.txt', 'w', encoding='utf-8') as f:
    f.write(f"Turf Races: {len(v5_turf['race_id'].unique())}\n\n")
    
    for r in range(1, 6):
        v5t = v5_turf[v5_turf['pred_rank'] == r]
        v4t = v4_turf[v4_turf['pred_rank'] == r]
        v5t = v5t[v5t['rank'].notna() & (v5t['rank'] > 0)]
        v4t = v4t[v4t['rank'].notna() & (v4t['rank'] > 0)]
        
        v5_win = len(v5t[v5t['rank']==1])/len(v5t) if len(v5t)>0 else 0
        v4_win = len(v4t[v4t['rank']==1])/len(v4t) if len(v4t)>0 else 0
        v5_ren = len(v5t[v5t['rank']<=2])/len(v5t) if len(v5t)>0 else 0
        v4_ren = len(v4t[v4t['rank']<=2])/len(v4t) if len(v4t)>0 else 0
        v5_fuku = len(v5t[v5t['rank']<=3])/len(v5t) if len(v5t)>0 else 0
        v4_fuku = len(v4t[v4t['rank']<=3])/len(v4t) if len(v4t)>0 else 0
        
        f.write(f"[Rank {r}]\n")
        f.write(f"  v4: Win={v4_win:.1%} Ren={v4_ren:.1%} Fuku={v4_fuku:.1%} (N={len(v4t)})\n")
        f.write(f"  v5: Win={v5_win:.1%} Ren={v5_ren:.1%} Fuku={v5_fuku:.1%} (N={len(v5t)})\n\n")

print("Done. Output saved to reports/turf_comparison.txt")
