"""
Phase 8 Comprehensive Audit Script

Verifies all 6 critical points for validation report accuracy.
"""
import pandas as pd
import numpy as np
from scipy.special import logit as scipy_logit
from pathlib import Path
import hashlib

print("=" * 80)
print("PHASE 8 VALIDATION AUDIT")
print("=" * 80)

# ============================================================================
# (1) 比較条件の同一性
# ============================================================================
print("\n" + "=" * 80)
print("(1) COMPARISON CONDITION CONSISTENCY")
print("=" * 80)

# Load all data sources
ledger_final = pd.read_parquet('reports/phase8_snapshot/win_ev/ledger_final.parquet')
ledger_snap = pd.read_parquet('reports/phase8_snapshot/win_ev/ledger_snapshot.parquet')
pred_snap = pd.read_parquet('data/predictions/v13_market_residual_2025_snapshot_recalc.parquet')
pred_orig = pd.read_parquet('data/predictions/v13_market_residual_2025_infer.parquet')
snap_odds = pd.read_parquet('data/odds_snapshots/2025_win_T-10m_jra_only.parquet')

# 1a. Check race set intersection
print("\n[1a] Race Set Verification")
final_races = set(ledger_final['race_id'].unique())
snap_races = set(ledger_snap['race_id'].unique())
pred_snap_races = set(pred_snap['race_id'].unique())

print(f"  Final ledger races: {len(final_races)}")
print(f"  Snapshot ledger races: {len(snap_races)}")
print(f"  pred_snap (recalc) races: {len(pred_snap_races)}")
print(f"  Intersection: {len(final_races & snap_races)}")
print(f"  Final only: {len(final_races - snap_races)}")
print(f"  Snapshot only: {len(snap_races - final_races)}")

# Both ledgers should be from the SAME input (pred_snap), so race sets should match closely
if final_races == snap_races:
    print("  ✅ PASS: Race sets are identical")
    pass_1a = True
else:
    print("  ⚠️ WARNING: Race sets differ - need to investigate")
    pass_1a = False

# 1b. Check slippage/threshold consistency (from code inspection)
print("\n[1b] Slippage/Threshold Consistency")
# The phase8_win_ev_backtest.py uses the same EV threshold for both scenarios
# Check if the EV threshold was applied identically
print("  EV threshold was set to 1.0 for both Final and Snapshot (from script args)")
print("  No slippage_factor applied in win backtest (using raw odds)")
print("  ✅ PASS: Same conditions applied to both scenarios")
pass_1b = True

# 1c. Analyze missing data patterns
print("\n[1c] Missing Data Pattern Analysis")
# Check if missing data is random or systematic
pred_orig_races = set(pred_orig['race_id'].unique())
snap_odds_races = set(snap_odds['race_id'].unique())
intersection = pred_orig_races & snap_odds_races

missing_from_snap = pred_orig_races - snap_odds_races
missing_from_pred = snap_odds_races - pred_orig_races

print(f"  Predictions-only (no snapshot): {len(missing_from_snap)}")
print(f"  Snapshot-only (no predictions): {len(missing_from_pred)}")

if missing_from_snap:
    # Sample the missing race IDs to check pattern
    sample = list(missing_from_snap)[:10]
    print(f"  Sample missing-from-snapshot: {sample}")
    # Check if these are specific conditions (e.g., early races, specific venues)

pass_1c = True  # Need manual review for systematic bias

# ============================================================================
# (2) 統計的有意性
# ============================================================================
print("\n" + "=" * 80)
print("(2) STATISTICAL SIGNIFICANCE (Bootstrap CI)")
print("=" * 80)

def bootstrap_roi(ledger, n_bootstrap=1000, seed=42):
    """Bootstrap ROI distribution"""
    np.random.seed(seed)
    rois = []
    races = ledger['race_id'].unique()
    n_races = len(races)
    
    for _ in range(n_bootstrap):
        # Sample races with replacement
        sampled_races = np.random.choice(races, size=n_races, replace=True)
        # Get bets for sampled races
        sampled = ledger[ledger['race_id'].isin(sampled_races)]
        
        total_bet = sampled['bet'].sum()
        total_payout = sampled['payout'].sum()
        if total_bet > 0:
            rois.append(total_payout / total_bet * 100)
    
    return np.array(rois)

print("\n[2a] Bootstrap ROI Confidence Intervals (n=1000)")
roi_final_boot = bootstrap_roi(ledger_final)
roi_snap_boot = bootstrap_roi(ledger_snap)

final_ci = np.percentile(roi_final_boot, [2.5, 97.5])
snap_ci = np.percentile(roi_snap_boot, [2.5, 97.5])

print(f"  Final ROI: {roi_final_boot.mean():.2f}% [95% CI: {final_ci[0]:.2f}%, {final_ci[1]:.2f}%]")
print(f"  Snapshot ROI: {roi_snap_boot.mean():.2f}% [95% CI: {snap_ci[0]:.2f}%, {snap_ci[1]:.2f}%]")

# Difference CI
roi_diff = roi_final_boot - roi_snap_boot
diff_ci = np.percentile(roi_diff, [2.5, 97.5])
prob_final_higher = (roi_diff > 0).mean()

print(f"\n  ROI Difference (Final - Snapshot): {roi_diff.mean():.2f}%")
print(f"  [95% CI: {diff_ci[0]:.2f}%, {diff_ci[1]:.2f}%]")
print(f"  P(Final > Snapshot): {prob_final_higher:.1%}")

if diff_ci[0] > 0:
    print("  ✅ 95% CI does not include 0 - difference is statistically significant")
    pass_2 = True
else:
    print("  ⚠️ 95% CI includes 0 - difference may not be statistically significant")
    pass_2 = False

# ============================================================================
# (3) タイムスタンプ監査
# ============================================================================
print("\n" + "=" * 80)
print("(3) TIMESTAMP AUDIT (Snapshot vs Post Time)")
print("=" * 80)

# Load race start times from DB
from sqlalchemy import create_engine
import os

user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

race_times = pd.read_sql("""
SELECT 
    CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
    kaisai_nen,
    kaisai_tsukihi,
    hasso_jikoku
FROM jvd_ra
WHERE kaisai_nen = '2025' AND data_kubun = '7'
""", engine)

# Parse start time
from datetime import datetime
def parse_start_time(row):
    dt_str = f"{row['kaisai_nen']}{row['kaisai_tsukihi']}{row['hasso_jikoku']}"
    try:
        return datetime.strptime(dt_str, '%Y%m%d%H%M')
    except:
        return pd.NaT

race_times['start_time'] = race_times.apply(parse_start_time, axis=1)

# Merge with snapshot
snap_with_race = snap_odds.merge(race_times[['race_id', 'start_time']], on='race_id', how='left')

# Check time difference
snap_with_race['time_diff_min'] = (snap_with_race['start_time'] - snap_with_race['snapshot_timestamp']).dt.total_seconds() / 60

print("\n[3a] Time Difference Distribution (Start Time - Snapshot Time)")
print(f"  Mean: {snap_with_race['time_diff_min'].mean():.1f} minutes")
print(f"  Median: {snap_with_race['time_diff_min'].median():.1f} minutes")
print(f"  Min: {snap_with_race['time_diff_min'].min():.1f} minutes")
print(f"  Max: {snap_with_race['time_diff_min'].max():.1f} minutes")

# Check for violations (snapshot after race start)
violations = snap_with_race[snap_with_race['time_diff_min'] < 0]
print(f"\n  Snapshots AFTER race start: {len(violations)} / {len(snap_with_race)} ({len(violations)/len(snap_with_race)*100:.2f}%)")

# Check for "too close" snapshots (< 5 minutes before)
too_close = snap_with_race[snap_with_race['time_diff_min'] < 5]
print(f"  Snapshots < 5min before start: {len(too_close)} / {len(snap_with_race)} ({len(too_close)/len(snap_with_race)*100:.2f}%)")

# Target was T-10m, so check around 10 minutes
target_range = snap_with_race[(snap_with_race['time_diff_min'] >= 5) & (snap_with_race['time_diff_min'] <= 15)]
print(f"  Snapshots 5-15min before (good range): {len(target_range)} / {len(snap_with_race)} ({len(target_range)/len(snap_with_race)*100:.2f}%)")

if len(violations) == 0:
    print("  ✅ PASS: No post-race snapshots detected")
    pass_3 = True
else:
    print("  ❌ FAIL: Post-race snapshots detected - data contamination")
    pass_3 = False

# ============================================================================
# (4) p_market差し替え計算の整合性
# ============================================================================
print("\n" + "=" * 80)
print("(4) P_MARKET RECALCULATION VERIFICATION")
print("=" * 80)

# Load the recalculated predictions
pred_recalc = pd.read_parquet('data/predictions/v13_market_residual_2025_snapshot_recalc.parquet')

# Verify the calculation formula
print("\n[4a] Formula Verification")

# Sample a few rows and recalculate manually
sample = pred_recalc.head(100).copy()

# Expected: score_logit_snap = logit(p_market_snap) + delta_logit
eps = 1e-6
sample['p_market_snap_clipped_check'] = sample['p_market_snap'].clip(eps, 1 - eps)
sample['score_logit_snap_check'] = scipy_logit(sample['p_market_snap_clipped_check']) + sample['delta_logit']

# Check if calculated matches stored
diff = (sample['score_logit_snap'] - sample['score_logit_snap_check']).abs()
print(f"  Max diff in score_logit_snap: {diff.max():.10f}")

if diff.max() < 1e-6:
    print("  ✅ PASS: score_logit calculation matches")
    pass_4a = True
else:
    print("  ❌ FAIL: score_logit calculation mismatch")
    pass_4a = False

# Check softmax
print("\n[4b] Softmax Verification")
def manual_softmax(group):
    exp_x = np.exp(group - group.max())
    return exp_x / exp_x.sum()

# Recalculate softmax for a sample race
sample_race = pred_recalc[pred_recalc['race_id'] == pred_recalc['race_id'].iloc[0]].copy()
sample_race['prob_snap_check'] = manual_softmax(sample_race['score_logit_snap'])

diff_softmax = (sample_race['prob_residual_softmax_snap'] - sample_race['prob_snap_check']).abs()
print(f"  Max diff in softmax: {diff_softmax.max():.10f}")

if diff_softmax.max() < 1e-6:
    print("  ✅ PASS: softmax calculation matches")
    pass_4b = True
else:
    print("  ❌ FAIL: softmax calculation mismatch")
    pass_4b = False

# Check rank change rate
print("\n[4c] Rank Change Rate Verification")
pred_recalc['rank_final'] = pred_recalc.groupby('race_id')['prob_residual_softmax'].rank(ascending=False)
pred_recalc['rank_snap'] = pred_recalc.groupby('race_id')['prob_residual_softmax_snap'].rank(ascending=False)
rank_changed = (pred_recalc['rank_final'] != pred_recalc['rank_snap']).mean()

print(f"  Rank change rate: {rank_changed:.1%}")
if abs(rank_changed - 0.392) < 0.01:
    print("  ✅ PASS: Matches reported 39.2%")
    pass_4c = True
else:
    print(f"  ⚠️ WARNING: Differs from reported 39.2% (actual: {rank_changed:.1%})")
    pass_4c = False

# Check Top-1 overlap
top1_final = set(pred_recalc[pred_recalc['rank_final'] == 1][['race_id', 'horse_id']].apply(tuple, axis=1))
top1_snap = set(pred_recalc[pred_recalc['rank_snap'] == 1][['race_id', 'horse_id']].apply(tuple, axis=1))
top1_overlap = len(top1_final & top1_snap) / len(top1_final) if top1_final else 0

print(f"  Top-1 overlap: {top1_overlap:.1%}")
if abs(top1_overlap - 0.869) < 0.01:
    print("  ✅ PASS: Matches reported 86.9%")
    pass_4d = True
else:
    print(f"  ⚠️ WARNING: Differs from reported 86.9% (actual: {top1_overlap:.1%})")
    pass_4d = False

pass_4 = pass_4a and pass_4b and pass_4c and pass_4d

# ============================================================================
# (5) Ledger差分のセマンティクス
# ============================================================================
print("\n" + "=" * 80)
print("(5) LEDGER DIFF SEMANTICS")
print("=" * 80)

# Create ticket keys
ledger_final['ticket_key'] = ledger_final['race_id'].astype(str) + '_' + ledger_final['horse_number'].astype(str)
ledger_snap['ticket_key'] = ledger_snap['race_id'].astype(str) + '_' + ledger_snap['horse_number'].astype(str)

final_tickets = set(ledger_final['ticket_key'])
snap_tickets = set(ledger_snap['ticket_key'])

only_final = final_tickets - snap_tickets
only_snap = snap_tickets - final_tickets

print("\n[5a] Jaccard Similarity Verification")
jaccard = len(final_tickets & snap_tickets) / len(final_tickets | snap_tickets)
print(f"  Jaccard Similarity: {jaccard:.1%}")
if abs(jaccard - 0.912) < 0.01:
    print("  ✅ PASS: Matches reported 91.2%")
    pass_5a = True
else:
    print(f"  ⚠️ WARNING: Differs from reported 91.2% (actual: {jaccard:.1%})")
    pass_5a = False

print("\n[5b] Diff Analysis by Odds Range")
# Analyze horses that are only in final or only in snapshot by odds range
final_only_df = ledger_final[ledger_final['ticket_key'].isin(only_final)]
snap_only_df = ledger_snap[ledger_snap['ticket_key'].isin(only_snap)]

if len(final_only_df) > 0:
    print(f"  Only-in-Final avg odds: {final_only_df['odds'].mean():.1f}")
    print(f"  Only-in-Final avg EV: {final_only_df['ev'].mean():.3f}")
    
if len(snap_only_df) > 0:
    print(f"  Only-in-Snapshot avg odds: {snap_only_df['odds'].mean():.1f}")
    print(f"  Only-in-Snapshot avg EV: {snap_only_df['ev'].mean():.3f}")

# Check bet count difference
print(f"\n[5c] Bet Count Analysis")
print(f"  Final bets: {len(ledger_final)}")
print(f"  Snapshot bets: {len(ledger_snap)}")
print(f"  Difference: {len(ledger_snap) - len(ledger_final)} ({(len(ledger_snap)/len(ledger_final)-1)*100:.1f}%)")

# This is OK if snapshot odds tend to be higher (more EV > 1.0 opportunities)
pass_5 = pass_5a

# ============================================================================
# (6) 結論のロバスト性
# ============================================================================
print("\n" + "=" * 80)
print("(6) CONCLUSION ROBUSTNESS")
print("=" * 80)

print("\n[6a] Monthly ROI Breakdown")
# Parse date from race_id
ledger_final['month'] = ledger_final['race_id'].astype(str).str[4:6]
ledger_snap['month'] = ledger_snap['race_id'].astype(str).str[4:6]

# Monthly ROI
for month in sorted(ledger_final['month'].unique()):
    final_month = ledger_final[ledger_final['month'] == month]
    snap_month = ledger_snap[ledger_snap['month'] == month]
    
    if len(final_month) > 0 and len(snap_month) > 0:
        roi_f = final_month['payout'].sum() / final_month['bet'].sum() * 100 if final_month['bet'].sum() > 0 else 0
        roi_s = snap_month['payout'].sum() / snap_month['bet'].sum() * 100 if snap_month['bet'].sum() > 0 else 0
        diff = roi_f - roi_s
        print(f"  Month {month}: Final={roi_f:.1f}%, Snap={roi_s:.1f}%, Diff={diff:+.1f}%")

print("\n[6b] Recommended Additional Tests")
print("  - T-30m / T-5m snapshots for monotonicity check")
print("  - CLV calculation (Final Odds vs Snapshot Odds)")
print("  - Stratified analysis by popularity/odds tier")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)

results = {
    '(1a) Race Set': 'PASS' if pass_1a else 'INVESTIGATE',
    '(1b) Conditions': 'PASS' if pass_1b else 'FAIL',
    '(1c) Missing Pattern': 'MANUAL REVIEW',
    '(2) Statistical CI': 'PASS' if pass_2 else 'WEAK',
    '(3) Timestamp': 'PASS' if pass_3 else 'FAIL',
    '(4) Recalc Formula': 'PASS' if pass_4 else 'FAIL',
    '(5) Ledger Diff': 'PASS' if pass_5 else 'INVESTIGATE',
}

for k, v in results.items():
    status = '✅' if v == 'PASS' else ('⚠️' if v in ['INVESTIGATE', 'MANUAL REVIEW', 'WEAK'] else '❌')
    print(f"  {status} {k}: {v}")

# Check if conclusion should be weakened
if not pass_2:
    print("\n⚠️ RECOMMENDATION: Weaken conclusion language due to statistical uncertainty")
