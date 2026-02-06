"""
Multi-bet strategy ROI analysis
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.loader import JraVanDataLoader
import pandas as pd
import glob
from sqlalchemy import text

# Load predictions
files = sorted(glob.glob('reports/jra/wf_incremental/monthly/results_no_odds_2025-*.parquet'))
pred_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
pred_df['pred_rank'] = pred_df.groupby('race_id')['calib_prob'].rank(ascending=False, method='first').astype(int)
n_races = pred_df['race_id'].nunique()
print(f"Predictions: {len(pred_df)} records, {n_races} races")

# Load payout data
loader = JraVanDataLoader()

# Build race_id and get payout columns
payout_sql = text("""
SELECT 
    kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
    haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
    haraimodoshi_wide_1a, haraimodoshi_wide_1b,
    haraimodoshi_wide_2a, haraimodoshi_wide_2b,
    haraimodoshi_wide_3a, haraimodoshi_wide_3b,
    haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
    haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b
FROM jvd_hr
WHERE kaisai_nen = '2025'
""")
payout_df = pd.read_sql(payout_sql, loader.engine)
print(f"Payout data: {len(payout_df)} races")

# Parse umaren combination (e.g., "0311" -> (3, 11))
def parse_umaban(s):
    if pd.isna(s) or len(str(s)) < 4:
        return None, None
    s = str(s).zfill(4)
    return int(s[:2]), int(s[2:])

# Parse sanrenpuku/sanrentan (e.g., "030711" -> (3, 7, 11))
def parse_umaban3(s):
    if pd.isna(s) or len(str(s)) < 6:
        return None, None, None
    s = str(s).zfill(6)
    return int(s[:2]), int(s[2:4]), int(s[4:])

# Parse payout amount
def parse_payout(s):
    try:
        return int(str(s).strip())
    except:
        return 0

# Get Top5 horse numbers per race
top5 = pred_df[pred_df['pred_rank'] <= 5][['race_id', 'horse_number', 'pred_rank']].copy()
top5_pivot = top5.pivot(index='race_id', columns='pred_rank', values='horse_number').reset_index()
top5_pivot.columns = ['race_id', 'top1', 'top2', 'top3', 'top4', 'top5']

# Merge
merged = top5_pivot.merge(payout_df, on='race_id', how='inner')
print(f"Merged: {len(merged)} races")

# ========== 1. 単勝 Top1 ==========
top1_df = pred_df[pred_df['pred_rank'] == 1].copy()
top1_wins = top1_df[top1_df['rank'] == 1]
win_invest = len(top1_df) * 100
win_return = (top1_wins['odds'] * 100).sum()
win_roi = (win_return / win_invest - 1) * 100

print(f"\n=== 1. 単勝 (Top1) ===")
print(f"投資: {win_invest:,}円, 回収: {win_return:,.0f}円, ROI: {win_roi:.1f}%")

# ========== 2. 馬連 Box Top5 (10点) ==========
def check_umaren_hit(row):
    top5_set = {row['top1'], row['top2'], row['top3'], row['top4'], row['top5']}
    try:
        uma1, uma2 = parse_umaban(row['haraimodoshi_umaren_1a'])
        if uma1 and uma2 and {uma1, uma2} <= top5_set:
            return parse_payout(row['haraimodoshi_umaren_1b'])
    except:
        pass
    return 0

merged['umaren_payout'] = merged.apply(check_umaren_hit, axis=1)
umaren_hits = (merged['umaren_payout'] > 0).sum()
umaren_invest = len(merged) * 10 * 100  # 10 combinations
umaren_return = merged['umaren_payout'].sum()
umaren_roi = (umaren_return / umaren_invest - 1) * 100

print(f"\n=== 2. 馬連 Box (Top1-5, 10点) ===")
print(f"投資: {umaren_invest:,}円, 回収: {umaren_return:,.0f}円, ROI: {umaren_roi:.1f}%")
print(f"的中: {umaren_hits}/{len(merged)} ({umaren_hits/len(merged)*100:.1f}%)")

# ========== 3. ワイド Box Top5 (10点) ==========
def check_wide_hit(row):
    top5_set = {row['top1'], row['top2'], row['top3'], row['top4'], row['top5']}
    total_pay = 0
    for suffix in ['1', '2', '3']:
        try:
            uma1, uma2 = parse_umaban(row[f'haraimodoshi_wide_{suffix}a'])
            if uma1 and uma2 and {uma1, uma2} <= top5_set:
                total_pay += parse_payout(row[f'haraimodoshi_wide_{suffix}b'])
        except:
            pass
    return total_pay

merged['wide_payout'] = merged.apply(check_wide_hit, axis=1)
wide_hits = (merged['wide_payout'] > 0).sum()
wide_invest = len(merged) * 10 * 100  # 10 combinations
wide_return = merged['wide_payout'].sum()
wide_roi = (wide_return / wide_invest - 1) * 100

print(f"\n=== 3. ワイド Box (Top1-5, 10点) ===")
print(f"投資: {wide_invest:,}円, 回収: {wide_return:,.0f}円, ROI: {wide_roi:.1f}%")
print(f"的中: {wide_hits}/{len(merged)} ({wide_hits/len(merged)*100:.1f}%)")

# ========== 4. 三連複 Box Top5 (10点) ==========
def check_sanrenpuku_hit(row):
    top5_set = {row['top1'], row['top2'], row['top3'], row['top4'], row['top5']}
    try:
        uma1, uma2, uma3 = parse_umaban3(row['haraimodoshi_sanrenpuku_1a'])
        if uma1 and uma2 and uma3 and {uma1, uma2, uma3} <= top5_set:
            return parse_payout(row['haraimodoshi_sanrenpuku_1b'])
    except:
        pass
    return 0

merged['sanrenpuku_payout'] = merged.apply(check_sanrenpuku_hit, axis=1)
sanrenpuku_hits = (merged['sanrenpuku_payout'] > 0).sum()
sanrenpuku_invest = len(merged) * 10 * 100  # 10 combinations
sanrenpuku_return = merged['sanrenpuku_payout'].sum()
sanrenpuku_roi = (sanrenpuku_return / sanrenpuku_invest - 1) * 100

print(f"\n=== 4. 三連複 Box (Top1-5, 10点) ===")
print(f"投資: {sanrenpuku_invest:,}円, 回収: {sanrenpuku_return:,.0f}円, ROI: {sanrenpuku_roi:.1f}%")
print(f"的中: {sanrenpuku_hits}/{len(merged)} ({sanrenpuku_hits/len(merged)*100:.1f}%)")

# ========== 5. 三連単 Box Top5 (60点) ==========
def check_sanrentan_hit(row):
    top5_set = {row['top1'], row['top2'], row['top3'], row['top4'], row['top5']}
    try:
        uma1, uma2, uma3 = parse_umaban3(row['haraimodoshi_sanrentan_1a'])
        if uma1 and uma2 and uma3 and {uma1, uma2, uma3} <= top5_set:
            return parse_payout(row['haraimodoshi_sanrentan_1b'])
    except:
        pass
    return 0

merged['sanrentan_payout'] = merged.apply(check_sanrentan_hit, axis=1)
sanrentan_hits = (merged['sanrentan_payout'] > 0).sum()
sanrentan_invest = len(merged) * 60 * 100  # 60 combinations (5P3)
sanrentan_return = merged['sanrentan_payout'].sum()
sanrentan_roi = (sanrentan_return / sanrentan_invest - 1) * 100

print(f"\n=== 5. 三連単 Box (Top1-5, 60点) ===")
print(f"投資: {sanrentan_invest:,}円, 回収: {sanrentan_return:,.0f}円, ROI: {sanrentan_roi:.1f}%")
print(f"的中: {sanrentan_hits}/{len(merged)} ({sanrentan_hits/len(merged)*100:.1f}%)")

# Summary
print("\n" + "="*60)
print("=== Summary ===")
print(f"{'戦略':<18} {'点数':>6} {'投資額':>14} {'回収額':>14} {'ROI':>10}")
print("-"*65)
print(f"{'単勝 Top1':<18} {'1':>6} {win_invest:>14,} {win_return:>14,.0f} {win_roi:>9.1f}%")
print(f"{'馬連 Box5':<18} {'10':>6} {umaren_invest:>14,} {umaren_return:>14,.0f} {umaren_roi:>9.1f}%")
print(f"{'ワイド Box5':<18} {'10':>6} {wide_invest:>14,} {wide_return:>14,.0f} {wide_roi:>9.1f}%")
print(f"{'三連複 Box5':<18} {'10':>6} {sanrenpuku_invest:>14,} {sanrenpuku_return:>14,.0f} {sanrenpuku_roi:>9.1f}%")
print(f"{'三連単 Box5':<18} {'60':>6} {sanrentan_invest:>14,} {sanrentan_return:>14,.0f} {sanrentan_roi:>9.1f}%")
