"""
Analyze why horse_id has such high feature importance.
"""
import pandas as pd
import joblib
import numpy as np

# Load model and check feature importance
m = joblib.load('models/experiments/exp_t2_no_odds/model.pkl')
imp = pd.DataFrame({'feature': m.feature_name(), 'gain': m.feature_importance()})
imp = imp.sort_values('gain', ascending=False)

print('=== Top 10 Feature Importance (Gain) ===')
print(imp.head(10).to_string(index=False))

# Calculate percentage
total = imp['gain'].sum()
print()
print('=== Top 5 as % of Total Gain ===')
for _, row in imp.head(5).iterrows():
    feat = row['feature']
    pct = 100*row['gain']/total
    print(f'  {feat}: {pct:.1f}%')

# Check horse_id cardinality
df = pd.read_parquet('data/temp_t2/T2_features.parquet')
print()
print('=== Horse ID Statistics ===')
unique_horses = df['horse_id'].nunique()
total_records = len(df)
print(f'Unique horse_ids: {unique_horses:,}')
print(f'Total records: {total_records:,}')
print(f'Avg records per horse: {total_records/unique_horses:.1f}')

# Most frequent horses
horse_counts = df['horse_id'].value_counts()
print()
print('=== Horse Appearance Distribution ===')
print(f'Max appearances: {horse_counts.max()}')
print(f'Median appearances: {horse_counts.median():.0f}')
one_app = (horse_counts == 1).sum()
many_app = (horse_counts >= 10).sum()
print(f'Horses with 1 appearance: {one_app:,} ({100*one_app/len(horse_counts):.1f}%)')
print(f'Horses with 10+ appearances: {many_app:,}')

# Check if horse_id is acting as a "memory" of past performance
# Group by horse and calculate win rate
tgt = pd.read_parquet('data/temp_t2/T2_targets.parquet')
df['race_id'] = df['race_id'].astype(str)
tgt['race_id'] = tgt['race_id'].astype(str)
merged = pd.merge(df, tgt, on=['race_id', 'horse_number'])
merged['is_win'] = (merged['rank'] == 1).astype(int)

horse_stats = merged.groupby('horse_id').agg({
    'is_win': ['sum', 'count', 'mean']
}).reset_index()
horse_stats.columns = ['horse_id', 'wins', 'races', 'win_rate']

print()
print('=== Horse Win Rate Distribution ===')
print(f'Horses with 0 wins: {(horse_stats["wins"]==0).sum():,}')
print(f'Horses with 1+ wins: {(horse_stats["wins"]>=1).sum():,}')
print(f'Horses with 3+ wins: {(horse_stats["wins"]>=3).sum():,}')

# Top winning horses
top_winners = horse_stats.nlargest(10, 'wins')
print()
print('=== Top 10 Winning Horses ===')
print(top_winners.to_string(index=False))

# Show why horse_id is important: 
# LightGBM can use horse_id to encode "this horse tends to win"
print()
print('=== Why horse_id is Important ===')
print('horse_id acts as a "memory" feature that encodes:')
print('  1. Historical win tendency of each horse')
print('  2. Overall quality/class of the horse')
print('  3. Implicit information not captured by other features')
print()
print('This is both a strength (captures real patterns) and a risk:')
print('  - Risk of overfitting to specific horses')
print('  - Poor generalization to new/unseen horses')
