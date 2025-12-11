"""v4 vs v7 同条件戦略比較スクリプト"""
import pandas as pd
import numpy as np

# --- Load predictions ---
v4_raw = pd.read_parquet('experiments/predictions_ensemble_v4_2025.parquet')
v7 = pd.read_parquet('experiments/v7_ensemble_full/reports/predictions.parquet')

# --- Filter v4 to JRA only ---
jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
v4_raw['venue_code'] = v4_raw['venue'].astype(str).str[:2]
v4 = v4_raw[v4_raw['venue_code'].isin(jra_codes)].copy()

print('=== After JRA Filter ===')
print(f"v4: {len(v4)} rows, {v4['race_id'].nunique()} races")
print(f"v7: {len(v7)} rows, {v7['race_id'].nunique()} races")

# --- Define evaluation functions ---
def single_bet_roi(df, score_col='score'):
    results = []
    for race_id, group in df.groupby('race_id'):
        if group[score_col].isnull().all(): continue
        best = group.loc[group[score_col].idxmax()]
        odds = float(best['odds']) if pd.notna(best['odds']) else 0
        rank = int(best['rank']) if pd.notna(best['rank']) else 99
        ret = odds * 100 if rank == 1 else 0
        results.append({'bet': 100, 'return': ret, 'hit': 1 if rank == 1 else 0})
    if not results: return 0, 0, 0
    total_bet = sum(r['bet'] for r in results)
    total_ret = sum(r['return'] for r in results)
    roi = total_ret / total_bet * 100
    acc = sum(r['hit'] for r in results) / len(results)
    return roi, acc, len(results)

def conditional_single_bet(df, min_odds=0, max_odds=999, min_ev=0):
    """条件付き単勝ベット"""
    results = []
    for race_id, group in df.groupby('race_id'):
        if group['score'].isnull().all(): continue
        best = group.loc[group['score'].idxmax()]
        odds = float(best['odds']) if pd.notna(best['odds']) else 0
        ev = float(best['expected_value']) if pd.notna(best.get('expected_value', np.nan)) else 0
        rank = int(best['rank']) if pd.notna(best['rank']) else 99
        
        # Apply conditions
        if min_odds <= odds < max_odds and ev >= min_ev:
            ret = odds * 100 if rank == 1 else 0
            results.append({'bet': 100, 'return': ret, 'hit': 1 if rank == 1 else 0})
    
    if not results: return 0, 0, 0
    total_bet = sum(r['bet'] for r in results)
    total_ret = sum(r['return'] for r in results)
    roi = total_ret / total_bet * 100
    acc = sum(r['hit'] for r in results) / len(results)
    return roi, acc, len(results)

# --- Run comparisons ---
print()
print('=' * 60)
print('=== STRATEGY COMPARISON: v4 vs v7 (JRA 2025) ===')
print('=' * 60)

# 1. Simple single bet (max score)
v4_roi, v4_acc, v4_n = single_bet_roi(v4)
v7_roi, v7_acc, v7_n = single_bet_roi(v7)
print()
print('[1] 単勝 (Max Score) - 無条件')
print(f'    v4: ROI={v4_roi:.2f}%, Acc={v4_acc:.2%}, Races={v4_n}')
print(f'    v7: ROI={v7_roi:.2f}%, Acc={v7_acc:.2%}, Races={v7_n}')
print(f'    差: ROI={v7_roi-v4_roi:+.2f}%')

# 2. 穴馬狙い: Odds >= 10, EV >= 1.3
v4_roi, v4_acc, v4_n = conditional_single_bet(v4, min_odds=10.0, min_ev=1.3)
v7_roi, v7_acc, v7_n = conditional_single_bet(v7, min_odds=10.0, min_ev=1.3)
print()
print('[2] 単勝 (穴馬: Odds>=10, EV>=1.3)')
print(f'    v4: ROI={v4_roi:.2f}%, Acc={v4_acc:.2%}, Races={v4_n}')
print(f'    v7: ROI={v7_roi:.2f}%, Acc={v7_acc:.2%}, Races={v7_n}')
diff = v7_roi - v4_roi if v4_n > 0 and v7_n > 0 else 0
print(f'    差: ROI={diff:+.2f}%')

# 3. 本命狙い: Odds < 3, EV >= 1.0
v4_roi, v4_acc, v4_n = conditional_single_bet(v4, max_odds=3.0, min_ev=1.0)
v7_roi, v7_acc, v7_n = conditional_single_bet(v7, max_odds=3.0, min_ev=1.0)
print()
print('[3] 単勝 (本命: Odds<3, EV>=1.0)')
print(f'    v4: ROI={v4_roi:.2f}%, Acc={v4_acc:.2%}, Races={v4_n}')
print(f'    v7: ROI={v7_roi:.2f}%, Acc={v7_acc:.2%}, Races={v7_n}')
diff = v7_roi - v4_roi if v4_n > 0 and v7_n > 0 else 0
print(f'    差: ROI={diff:+.2f}%')

# 4. 高EV狙い: EV >= 1.5
v4_roi, v4_acc, v4_n = conditional_single_bet(v4, min_ev=1.5)
v7_roi, v7_acc, v7_n = conditional_single_bet(v7, min_ev=1.5)
print()
print('[4] 単勝 (高EV: EV>=1.5)')
print(f'    v4: ROI={v4_roi:.2f}%, Acc={v4_acc:.2%}, Races={v4_n}')
print(f'    v7: ROI={v7_roi:.2f}%, Acc={v7_acc:.2%}, Races={v7_n}')
diff = v7_roi - v4_roi if v4_n > 0 and v7_n > 0 else 0
print(f'    差: ROI={diff:+.2f}%')

# 5. 中穴狙い: 5 <= Odds < 20, EV >= 1.2
v4_roi, v4_acc, v4_n = conditional_single_bet(v4, min_odds=5.0, max_odds=20.0, min_ev=1.2)
v7_roi, v7_acc, v7_n = conditional_single_bet(v7, min_odds=5.0, max_odds=20.0, min_ev=1.2)
print()
print('[5] 単勝 (中穴: 5<=Odds<20, EV>=1.2)')
print(f'    v4: ROI={v4_roi:.2f}%, Acc={v4_acc:.2%}, Races={v4_n}')
print(f'    v7: ROI={v7_roi:.2f}%, Acc={v7_acc:.2%}, Races={v7_n}')
diff = v7_roi - v4_roi if v4_n > 0 and v7_n > 0 else 0
print(f'    差: ROI={diff:+.2f}%')

print()
print('=' * 60)
