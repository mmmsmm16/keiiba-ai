"""
Phase 8: Population Alignment Analysis

Fixes the intersection inconsistency and recalculates with proper denominator.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PHASE 8: POPULATION ALIGNMENT ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. Load all data and establish TRUE intersection
# ============================================================================
print("\n[1] Establishing True Intersection")

# Load prediction data (this is the source of truth for what we CAN evaluate)
pred_snap = pd.read_parquet('data/predictions/v13_market_residual_2025_snapshot_recalc.parquet')
pred_races = set(pred_snap['race_id'].unique())
print(f"  pred_snap races: {len(pred_races)}")

# Load existing ledgers
ledger_final = pd.read_parquet('reports/phase8_snapshot/win_ev/ledger_final.parquet')
ledger_snap = pd.read_parquet('reports/phase8_snapshot/win_ev/ledger_snapshot.parquet')

final_bet_races = set(ledger_final['race_id'].unique())
snap_bet_races = set(ledger_snap['race_id'].unique())

print(f"  Final ledger bet_races: {len(final_bet_races)}")
print(f"  Snapshot ledger bet_races: {len(snap_bet_races)}")

# The TRUE intersection should be pred_snap_races (2,951)
# The 2908/2919 are "bet_races" (races where at least 1 horse had EV > 1.0)
total_races = pred_races
print(f"\n  TRUE INTERSECTION (total_races): {len(total_races)}")
print(f"  Final bet_races: {len(final_bet_races)} ({len(final_bet_races)/len(total_races)*100:.1f}%)")
print(f"  Snapshot bet_races: {len(snap_bet_races)} ({len(snap_bet_races)/len(total_races)*100:.1f}%)")

# ============================================================================
# 2. Identify race differences
# ============================================================================
print("\n[2] Race Difference Analysis")

# Races with bets in Final but NOT in Snapshot
only_final_bets = final_bet_races - snap_bet_races
# Races with bets in Snapshot but NOT in Final  
only_snap_bets = snap_bet_races - final_bet_races
# Races with bets in both
both_bets = final_bet_races & snap_bet_races
# Races with NO bets in either
no_bets = total_races - final_bet_races - snap_bet_races

print(f"  Bets in BOTH: {len(both_bets)}")
print(f"  Bets ONLY in Final: {len(only_final_bets)}")
print(f"  Bets ONLY in Snapshot: {len(only_snap_bets)}")
print(f"  NO bets in either: {len(no_bets)}")

# List out the differences (up to 50)
print(f"\n  Races ONLY in Final (showing first 20):")
for i, r in enumerate(sorted(only_final_bets)[:20]):
    print(f"    {r}")

print(f"\n  Races ONLY in Snapshot (showing first 20):")
for i, r in enumerate(sorted(only_snap_bets)[:20]):
    print(f"    {r}")

# ============================================================================
# 3. Create aligned ledgers (all 2,951 races)
# ============================================================================
print("\n[3] Creating Aligned Ledgers (Full Population)")

def create_aligned_ledger(ledger, all_races, label):
    """Create ledger with all races, filling missing with 0 bet/payout"""
    
    # Aggregate existing bets by race
    race_agg = ledger.groupby('race_id').agg({
        'bet': 'sum',
        'payout': 'sum',
        'hit': 'sum'
    }).reset_index()
    race_agg['bet_count'] = ledger.groupby('race_id').size().values
    race_agg.columns = ['race_id', 'stake', 'payout', 'hits', 'bet_count']
    
    # Create full race list
    all_races_df = pd.DataFrame({'race_id': list(all_races)})
    
    # Left join to keep all races
    aligned = all_races_df.merge(race_agg, on='race_id', how='left')
    aligned = aligned.fillna({'stake': 0, 'payout': 0, 'hits': 0, 'bet_count': 0})
    
    # Calculate profit per race
    aligned['profit'] = aligned['payout'] - aligned['stake']
    aligned['has_bet'] = aligned['stake'] > 0
    
    print(f"\n  {label} Aligned Ledger:")
    print(f"    Total races: {len(aligned)}")
    print(f"    Bet races: {aligned['has_bet'].sum()}")
    print(f"    No-bet races: {(~aligned['has_bet']).sum()}")
    print(f"    Total stake: ¥{aligned['stake'].sum():,.0f}")
    print(f"    Total payout: ¥{aligned['payout'].sum():,.0f}")
    print(f"    Total profit: ¥{aligned['profit'].sum():,.0f}")
    print(f"    ROI: {aligned['payout'].sum() / aligned['stake'].sum() * 100:.2f}%")
    
    return aligned

aligned_final = create_aligned_ledger(ledger_final, total_races, "Final")
aligned_snap = create_aligned_ledger(ledger_snap, total_races, "Snapshot")

# ============================================================================
# 4. Bootstrap CI on FULL population (race-level)
# ============================================================================
print("\n[4] Bootstrap CI on Full Population (2,951 races)")

def bootstrap_roi_race_level(aligned_ledger, n_bootstrap=1000, seed=42):
    """Bootstrap ROI at race level (includes 0-bet races)"""
    np.random.seed(seed)
    
    profits = aligned_ledger['profit'].values
    stakes = aligned_ledger['stake'].values
    n_races = len(aligned_ledger)
    
    rois = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_races, size=n_races, replace=True)
        total_stake = stakes[idx].sum()
        total_profit = profits[idx].sum()
        if total_stake > 0:
            roi = (total_stake + total_profit) / total_stake * 100
            rois.append(roi)
    
    return np.array(rois)

roi_final_boot = bootstrap_roi_race_level(aligned_final)
roi_snap_boot = bootstrap_roi_race_level(aligned_snap)

final_ci = np.percentile(roi_final_boot, [2.5, 97.5])
snap_ci = np.percentile(roi_snap_boot, [2.5, 97.5])

print(f"\n  Final ROI: {roi_final_boot.mean():.2f}% [95% CI: {final_ci[0]:.2f}%, {final_ci[1]:.2f}%]")
print(f"  Snapshot ROI: {roi_snap_boot.mean():.2f}% [95% CI: {snap_ci[0]:.2f}%, {snap_ci[1]:.2f}%]")

# Difference CI
roi_diff = roi_final_boot - roi_snap_boot
diff_ci = np.percentile(roi_diff, [2.5, 97.5])
prob_final_higher = (roi_diff > 0).mean()

print(f"\n  ROI Difference (Final - Snapshot): {roi_diff.mean():.2f}%")
print(f"  [95% CI: {diff_ci[0]:.2f}%, {diff_ci[1]:.2f}%]")
print(f"  P(Final > Snapshot): {prob_final_higher:.1%}")

if diff_ci[0] > 0:
    ci_conclusion = "✅ 95% CI does NOT include 0 - statistically significant"
else:
    ci_conclusion = "⚠️ 95% CI includes 0 - NOT statistically significant"
print(f"  {ci_conclusion}")

# ============================================================================
# 5. Summary Statistics for Report
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY FOR REPORT UPDATE")
print("=" * 80)

summary = {
    'total_races': len(total_races),
    'final': {
        'bet_races': int(aligned_final['has_bet'].sum()),
        'no_bet_races': int((~aligned_final['has_bet']).sum()),
        'total_stake': int(aligned_final['stake'].sum()),
        'total_payout': int(aligned_final['payout'].sum()),
        'total_profit': int(aligned_final['profit'].sum()),
        'roi': aligned_final['payout'].sum() / aligned_final['stake'].sum() * 100,
        'ci_low': final_ci[0],
        'ci_high': final_ci[1],
    },
    'snapshot': {
        'bet_races': int(aligned_snap['has_bet'].sum()),
        'no_bet_races': int((~aligned_snap['has_bet']).sum()),
        'total_stake': int(aligned_snap['stake'].sum()),
        'total_payout': int(aligned_snap['payout'].sum()),
        'total_profit': int(aligned_snap['profit'].sum()),
        'roi': aligned_snap['payout'].sum() / aligned_snap['stake'].sum() * 100,
        'ci_low': snap_ci[0],
        'ci_high': snap_ci[1],
    },
    'diff': {
        'mean': roi_diff.mean(),
        'ci_low': diff_ci[0],
        'ci_high': diff_ci[1],
        'p_final_higher': prob_final_higher,
    }
}

print(f"""
## Population/Denominator Definition

| Metric | Final | Snapshot |
|--------|-------|----------|
| **Total Races (denominator)** | {summary['total_races']} | {summary['total_races']} |
| Bet Races (≥1 EV>1 horse) | {summary['final']['bet_races']} | {summary['snapshot']['bet_races']} |
| No-Bet Races | {summary['final']['no_bet_races']} | {summary['snapshot']['no_bet_races']} |
| Total Stake | ¥{summary['final']['total_stake']:,} | ¥{summary['snapshot']['total_stake']:,} |
| Total Payout | ¥{summary['final']['total_payout']:,} | ¥{summary['snapshot']['total_payout']:,} |
| **ROI** | **{summary['final']['roi']:.2f}%** | **{summary['snapshot']['roi']:.2f}%** |
| 95% CI | [{summary['final']['ci_low']:.1f}%, {summary['final']['ci_high']:.1f}%] | [{summary['snapshot']['ci_low']:.1f}%, {summary['snapshot']['ci_high']:.1f}%] |

### ROI Difference
- Δ = {summary['diff']['mean']:.2f}%
- 95% CI = [{summary['diff']['ci_low']:.2f}%, {summary['diff']['ci_high']:.2f}%]
- P(Final > Snapshot) = {summary['diff']['p_final_higher']:.1%}
- {ci_conclusion}
""")

# Save aligned ledgers
out_dir = Path('reports/phase8_snapshot/win_ev')
aligned_final.to_parquet(out_dir / 'aligned_ledger_final.parquet')
aligned_snap.to_parquet(out_dir / 'aligned_ledger_snapshot.parquet')
print(f"\nSaved aligned ledgers to {out_dir}")
