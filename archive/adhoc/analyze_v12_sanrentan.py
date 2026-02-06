#!/usr/bin/env python
"""
v12三連単戦略の詳細分析スクリプト
的中率、最大連敗数、ROI等を計算
(Option C 戦略: 1頭軸4頭フォーメーション 6点買い)
"""
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from itertools import permutations

# Load predictions
preds = pd.read_parquet('experiments/v12_tabnet_revival/reports/predictions.parquet')

# Load payout data
def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

engine = get_db_engine()
query = text("SELECT * FROM jvd_hr WHERE kaisai_nen = '2025'")
pay_df = pd.read_sql(query, engine)
pay_df['race_id'] = (
    pay_df['kaisai_nen'].astype(str) +
    pay_df['keibajo_code'].astype(str) +
    pay_df['kaisai_kai'].astype(str) +
    pay_df['kaisai_nichime'].astype(str) +
    pay_df['race_bango'].astype(str)
)

# Build payout map
payout_map = {}
for _, row in pay_df.iterrows():
    rid = row['race_id']
    payout_map[rid] = {'sanrentan': {}}
    for i in range(1, 7):
        col_a = f'haraimodoshi_sanrentan_{i}a'
        col_b = f'haraimodoshi_sanrentan_{i}b'
        if col_a in row and row[col_a] and str(row[col_a]).strip():
            try:
                comb = str(row[col_a]).strip()
                pay = int(float(str(row[col_b]).strip()))
                payout_map[rid]['sanrentan'][comb] = pay
            except:
                pass

# Analyze sanrentan with Option C logic
results = []

for race_id, group in preds.groupby('race_id'):
    if race_id not in payout_map:
        continue
    
    sorted_horses = group.sort_values('score', ascending=False)
    if len(sorted_horses) < 6:
        continue
    
    # Top1馬情報
    h = sorted_horses['horse_number'].astype(int).tolist()
    top1 = sorted_horses.iloc[0]
    pop = int(top1['popularity']) if pd.notna(top1.get('popularity', np.nan)) else 99
    
    # スコア差
    scores = sorted_horses['score'].head(6).values
    gap = scores[0] - scores[5] if len(scores) >= 6 else 0.5
    
    # Option C 条件: 7番人気以上 OR gap < 0.3
    if not (pop >= 7 or gap < 0.3):
        continue
    
    # 三連単1頭軸4頭 (6点)
    axis = h[0]
    opps = h[1:4]  # Top2-4位
    perms = list(permutations(opps, 2))
    
    cost = len(perms) * 100  # 6点 × 100円 = 600円
    ret = 0
    hit = False
    
    for p in perms:
        comb_str = f"{axis:02}{p[0]:02}{p[1]:02}"
        if comb_str in payout_map[race_id]['sanrentan']:
            ret += payout_map[race_id]['sanrentan'][comb_str]
            hit = True
    
    results.append({
        'race_id': race_id,
        'date': top1['date'],
        'venue': top1.get('venue', ''),
        'popularity': pop,
        'gap': gap,
        'condition': '7pop' if pop >= 7 else 'gap',
        'axis': axis,
        'opps': opps,
        'cost': cost,
        'return': ret,
        'hit': hit,
        'profit': ret - cost
    })

df = pd.DataFrame(results)

print("=" * 60)
print("v12 三連単戦略 (Option C: 1頭軸4頭) 詳細分析")
print("=" * 60)
print(f"対象レース数: {len(df)}")
print(f"  - 7番人気以上: {len(df[df['condition'] == '7pop'])}")
print(f"  - 接戦(gap<0.3): {len(df[df['condition'] == 'gap'])}")
print(f"\n的中数: {df['hit'].sum()}")
print(f"的中率: {df['hit'].mean() * 100:.2f}%")

total_cost = df['cost'].sum()
total_ret = df['return'].sum()
print(f"\n総投資額: {total_cost:,}円 ({len(df)}レース × 600円)")
print(f"総回収額: {total_ret:,}円")
print(f"ROI: {total_ret / total_cost * 100:.1f}%")
print(f"純利益: {total_ret - total_cost:+,}円")

# Max losing streak
if len(df) > 0:
    df = df.sort_values('date')
    streak = 0
    max_streak = 0
    for hit in df['hit']:
        if not hit:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"\n最大連敗数: {max_streak}")

# Monthly stats
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
monthly = df.groupby('month').agg({
    'hit': ['sum', 'count'],
    'return': 'sum',
    'cost': 'sum'
})
monthly.columns = ['hits', 'bets', 'return', 'cost']
monthly['hit_rate'] = monthly['hits'] / monthly['bets'] * 100
monthly['roi'] = monthly['return'] / monthly['cost'] * 100
monthly['profit'] = monthly['return'] - monthly['cost']
print("\n月別成績:")
print(monthly.to_string())

# Show hits
print("\n的中レース詳細:")
hits = df[df['hit']].sort_values('return', ascending=False)
for _, row in hits.head(10).iterrows():
    print(f"  {row['date']} | 軸{row['axis']} → 相手{row['opps']} | 払戻 {row['return']:,}円 (ROI {row['return']/600*100:.0f}%)")

# 条件別分析
print("\n" + "=" * 60)
print("条件別分析")
print("=" * 60)
for cond in ['7pop', 'gap']:
    cond_df = df[df['condition'] == cond]
    if len(cond_df) > 0:
        label = "7番人気以上" if cond == '7pop' else "接戦(gap<0.3)"
        print(f"\n【{label}】")
        print(f"  レース数: {len(cond_df)}")
        print(f"  的中数: {cond_df['hit'].sum()}")
        print(f"  的中率: {cond_df['hit'].mean() * 100:.2f}%")
        print(f"  投資: {cond_df['cost'].sum():,}円")
        print(f"  回収: {cond_df['return'].sum():,}円")
        print(f"  ROI: {cond_df['return'].sum() / cond_df['cost'].sum() * 100:.1f}%")
