import pandas as pd
import numpy as np

# Load OOF data with calibrated predictions
oof = pd.read_parquet('data/predictions/v13_wf_2025_full_retrained_oof_calibrated.parquet')

# Load preprocessed data for popularity and odds
prep = pd.read_parquet('data/processed/preprocessed_data.parquet', columns=['race_id', 'horse_number', 'popularity', 'odds', 'date'])
prep['race_id'] = prep['race_id'].astype(str)
prep = prep.dropna(subset=['horse_number'])
prep['horse_number'] = prep['horse_number'].astype(float).astype(int)

# Merge
oof['race_id'] = oof['race_id'].astype(str)
oof = oof.dropna(subset=['horse_number'])
oof['horse_number'] = oof['horse_number'].astype(float).astype(int)
df = pd.merge(oof, prep, on=['race_id', 'horse_number'], how='left')

# Filter valid data
df = df.dropna(subset=['odds', 'rank', 'popularity'])
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

# Add month from date
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# Stats per month
results = []
np.random.seed(42)  # For reproducibility of placebo

for month in sorted(df['month'].unique()):
    month_df = df[df['month'] == month]
    
    month_stats = {'month': month, 'model_bet': 0, 'model_ret': 0, 
                   'pop_bet': 0, 'pop_ret': 0, 'placebo_bet': 0, 'placebo_ret': 0, 'races': 0}
    
    for rid, grp in month_df.groupby('race_id'):
        month_stats['races'] += 1
        
        # Model Top 1
        model_top1 = grp.loc[grp['pred_logit'].idxmax()]
        month_stats['model_bet'] += 100
        if model_top1['rank'] == 1:
            month_stats['model_ret'] += int(model_top1['odds'] * 100)
        
        # Popularity Top 1
        pop_top1 = grp.loc[grp['popularity'].idxmin()]
        month_stats['pop_bet'] += 100
        if pop_top1['rank'] == 1:
            month_stats['pop_ret'] += int(pop_top1['odds'] * 100)
        
        # Placebo (Random)
        random_pick = grp.sample(1).iloc[0]
        month_stats['placebo_bet'] += 100
        if random_pick['rank'] == 1:
            month_stats['placebo_ret'] += int(random_pick['odds'] * 100)
    
    # Calc ROI
    month_stats['model_roi'] = month_stats['model_ret'] / month_stats['model_bet'] * 100 if month_stats['model_bet'] > 0 else 0
    month_stats['pop_roi'] = month_stats['pop_ret'] / month_stats['pop_bet'] * 100 if month_stats['pop_bet'] > 0 else 0
    month_stats['placebo_roi'] = month_stats['placebo_ret'] / month_stats['placebo_bet'] * 100 if month_stats['placebo_bet'] > 0 else 0
    
    results.append(month_stats)

# Print table
print('| 月 | レース数 | AIモデルTop1 | 人気1位 | プラシーボ |')
print('| :--- | :--- | :--- | :--- | :--- |')
total_model_bet, total_model_ret = 0, 0
total_pop_bet, total_pop_ret = 0, 0
total_placebo_bet, total_placebo_ret = 0, 0
total_races = 0
for r in results:
    print(f"| {r['month']}月 | {r['races']} | {r['model_roi']:.1f}% | {r['pop_roi']:.1f}% | {r['placebo_roi']:.1f}% |")
    total_model_bet += r['model_bet']; total_model_ret += r['model_ret']
    total_pop_bet += r['pop_bet']; total_pop_ret += r['pop_ret']
    total_placebo_bet += r['placebo_bet']; total_placebo_ret += r['placebo_ret']
    total_races += r['races']

total_model_roi = total_model_ret / total_model_bet * 100
total_pop_roi = total_pop_ret / total_pop_bet * 100
total_placebo_roi = total_placebo_ret / total_placebo_bet * 100
print(f'| **合計** | **{total_races}** | **{total_model_roi:.1f}%** | **{total_pop_roi:.1f}%** | **{total_placebo_roi:.1f}%** |')
