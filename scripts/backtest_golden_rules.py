"""
Out-of-Sample Backtest: 過去最適化戦略を2025年に適用

過去データ（2024年以前）でグリッドサーチした戦略パラメータを
2025年データに適用してROIを計算する真のバックテスト。
"""
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from sqlalchemy import create_engine, text
import os

# --- Load v7 2025 predictions (JRA only) ---
v7 = pd.read_parquet('experiments/v7_ensemble_full/reports/predictions.parquet')
print(f"v7 predictions: {len(v7)} rows, {v7['race_id'].nunique()} races")

# --- Load payout data ---
db_path = os.environ.get('DATABASE_URL', 'mssql+pyodbc://PCKEIBA')
try:
    engine = create_engine(db_path)
    payout_df = pd.read_sql("""
        SELECT 
            race_id,
            haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
            haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
            haraimodoshi_umaren_2a, haraimodoshi_umaren_2b,
            haraimodoshi_umaren_3a, haraimodoshi_umaren_3b,
            haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
            haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
            haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
            haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
            haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
            haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
            haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b,
            haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b,
            haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
        FROM jvd_hr
        WHERE SUBSTRING(race_id, 1, 4) = '2025'
    """, engine)
    print(f"Payout data: {len(payout_df)} races")
except Exception as e:
    print(f"DB connection failed: {e}")
    # Try parquet fallback
    payout_df = pd.read_parquet('experiments/payouts_2024_2025.parquet')
    payout_df = payout_df[payout_df['race_id'].str[:4] == '2025']
    print(f"Payout data (parquet): {len(payout_df)} races")

# Build payout map
payout_map = {}
for _, row in payout_df.iterrows():
    rid = row['race_id']
    payout_map[rid] = {'tansho': {}, 'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}}
    
    # Tansho
    if pd.notna(row.get('haraimodoshi_tansho_1a')):
        try:
            payout_map[rid]['tansho'][str(row['haraimodoshi_tansho_1a']).zfill(2)] = int(row['haraimodoshi_tansho_1b'])
        except: pass
    
    # Umaren
    for i in range(1, 4):
        if pd.notna(row.get(f'haraimodoshi_umaren_{i}a')):
            try:
                comb = str(row[f'haraimodoshi_umaren_{i}a'])
                pay = int(row[f'haraimodoshi_umaren_{i}b'])
                payout_map[rid]['umaren'][comb] = pay
            except: pass
    
    # Sanrenpuku
    for i in range(1, 4):
        if pd.notna(row.get(f'haraimodoshi_sanrenpuku_{i}a')):
            try:
                comb = str(row[f'haraimodoshi_sanrenpuku_{i}a'])
                pay = int(row[f'haraimodoshi_sanrenpuku_{i}b'])
                payout_map[rid]['sanrenpuku'][comb] = pay
            except: pass
    
    # Sanrentan
    for i in range(1, 7):
        if pd.notna(row.get(f'haraimodoshi_sanrentan_{i}a')):
            try:
                comb = str(row[f'haraimodoshi_sanrentan_{i}a'])
                pay = int(row[f'haraimodoshi_sanrentan_{i}b'])
                payout_map[rid]['sanrentan'][comb] = pay
            except: pass

print(f"Payout map: {len(payout_map)} races")

# --- Define Strategy Functions ---
def strategy_sanrentan_nagashi(df, payout_map, min_prob=0, min_ev=0, min_odds=0, max_odds=999, n_opps=6):
    """3連単1着軸流し"""
    stats = {'bet': 0, 'return': 0, 'count': 0, 'hit': 0}
    
    for race_id, group in df.groupby('race_id'):
        if race_id not in payout_map: continue
        
        sorted_horses = group.sort_values('score', ascending=False)
        if len(sorted_horses) < n_opps + 1: continue
        
        axis = sorted_horses.iloc[0]
        axis_odds = float(axis['odds']) if pd.notna(axis['odds']) else 0
        axis_prob = float(axis['prob']) if pd.notna(axis.get('prob', np.nan)) else 0
        axis_ev = float(axis['expected_value']) if pd.notna(axis.get('expected_value', np.nan)) else 0
        
        # Apply conditions
        if axis_prob < min_prob: continue
        if axis_ev < min_ev: continue
        if axis_odds < min_odds or axis_odds >= max_odds: continue
        
        axis_num = int(axis['horse_number'])
        opps = sorted_horses.iloc[1:n_opps+1]['horse_number'].astype(int).tolist()
        
        perms = list(permutations(opps, 2))
        stats['bet'] += len(perms) * 100
        stats['count'] += 1
        
        race_payouts = payout_map[race_id].get('sanrentan', {})
        for p in perms:
            comb_str = f"{axis_num:02}{p[0]:02}{p[1]:02}"
            if comb_str in race_payouts:
                stats['return'] += race_payouts[comb_str]
                stats['hit'] += 1
    
    roi = stats['return'] / stats['bet'] * 100 if stats['bet'] > 0 else 0
    return {
        'roi': roi,
        'bet': stats['bet'],
        'return': stats['return'],
        'races': stats['count'],
        'hits': stats['hit']
    }

def strategy_tansho(df, payout_map, min_prob=0, min_ev=0, min_odds=0, max_odds=999):
    """単勝"""
    stats = {'bet': 0, 'return': 0, 'count': 0, 'hit': 0}
    
    for race_id, group in df.groupby('race_id'):
        if race_id not in payout_map: continue
        
        sorted_horses = group.sort_values('score', ascending=False)
        best = sorted_horses.iloc[0]
        odds = float(best['odds']) if pd.notna(best['odds']) else 0
        prob = float(best['prob']) if pd.notna(best.get('prob', np.nan)) else 0
        ev = float(best['expected_value']) if pd.notna(best.get('expected_value', np.nan)) else 0
        
        # Apply conditions
        if prob < min_prob: continue
        if ev < min_ev: continue
        if odds < min_odds or odds >= max_odds: continue
        
        horse_num = int(best['horse_number'])
        stats['bet'] += 100
        stats['count'] += 1
        
        tansho_payouts = payout_map[race_id].get('tansho', {})
        horse_key = str(horse_num).zfill(2)
        if horse_key in tansho_payouts:
            stats['return'] += tansho_payouts[horse_key]
            stats['hit'] += 1
    
    roi = stats['return'] / stats['bet'] * 100 if stats['bet'] > 0 else 0
    return {
        'roi': roi,
        'bet': stats['bet'],
        'return': stats['return'],
        'races': stats['count'],
        'hits': stats['hit']
    }

# --- Run Out-of-Sample Backtest ---
print()
print('=' * 70)
print('=== OUT-OF-SAMPLE BACKTEST: 過去最適化戦略 を 2025年v7 に適用 ===')
print('=' * 70)

# Golden Rule 1: 穴馬3連単 (Odds >= 10, EV >= 1.3)
print()
print('[Golden Rule 1] 穴馬3連単 - 軸:Odds>=10, EV>=1.3, 相手6頭')
result = strategy_sanrentan_nagashi(v7, payout_map, min_odds=10.0, min_ev=1.3, n_opps=6)
print(f"  ROI: {result['roi']:.2f}%")
print(f"  投資: {result['bet']:,}円, 回収: {result['return']:,}円")
print(f"  レース数: {result['races']}, 的中: {result['hits']}")

# Golden Rule 2: 本命3連単 (Odds < 3, EV >= 1.0)
print()
print('[Golden Rule 2] 本命3連単 - 軸:Odds<3, EV>=1.0, 相手6頭')
result = strategy_sanrentan_nagashi(v7, payout_map, max_odds=3.0, min_ev=1.0, n_opps=6)
print(f"  ROI: {result['roi']:.2f}%")
print(f"  投資: {result['bet']:,}円, 回収: {result['return']:,}円")
print(f"  レース数: {result['races']}, 的中: {result['hits']}")

# Golden Rule 3: 大穴単勝 (Odds >= 10)
print()
print('[Golden Rule 3] 大穴単勝 - Odds>=10')
result = strategy_tansho(v7, payout_map, min_odds=10.0)
print(f"  ROI: {result['roi']:.2f}%")
print(f"  投資: {result['bet']:,}円, 回収: {result['return']:,}円")
print(f"  レース数: {result['races']}, 的中: {result['hits']}")

# Additional strategies from LEADERBOARD
print()
print('[Strategy 4] 条件付単勝 - Prob>=0.20, EV>=1.2, Odds>=3')
result = strategy_tansho(v7, payout_map, min_prob=0.20, min_ev=1.2, min_odds=3.0)
print(f"  ROI: {result['roi']:.2f}%")
print(f"  投資: {result['bet']:,}円, 回収: {result['return']:,}円")
print(f"  レース数: {result['races']}, 的中: {result['hits']}")

print()
print('[Strategy 5] 3連単 無条件 - 相手6頭')
result = strategy_sanrentan_nagashi(v7, payout_map, n_opps=6)
print(f"  ROI: {result['roi']:.2f}%")
print(f"  投資: {result['bet']:,}円, 回収: {result['return']:,}円")
print(f"  レース数: {result['races']}, 的中: {result['hits']}")

print()
print('[Strategy 6] 3連単 相手7頭 - 無条件')
result = strategy_sanrentan_nagashi(v7, payout_map, n_opps=7)
print(f"  ROI: {result['roi']:.2f}%")
print(f"  投資: {result['bet']:,}円, 回収: {result['return']:,}円")
print(f"  レース数: {result['races']}, 的中: {result['hits']}")

print()
print('=' * 70)
