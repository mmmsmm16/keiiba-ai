
import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_history_boundary():
    logger.info("--- Test: History Stats Boundary (Horse ID) ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    
    # 2 Horses, 2 Races each.
    # H1: R1(Rank1), R2(Rank1) -> MeanRank5 should be 1.0 at R2.
    # H2: R3(Rank10), R4(Rank10) -> MeanRank5 should be 10.0 at R4.
    # If H2 mixes with H1, it might be different.
    
    data = {
        'race_id': ['R1', 'R2', 'R3', 'R4'],
        'horse_id': ['H1', 'H1', 'H2', 'H2'], # Sorted order usually
        'horse_number': [1, 1, 1, 1],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'rank': [1, 1, 10, 10], 
        'time_diff': [0.0, 0.0, 1.0, 1.0],
        'honshokin': [0, 0, 0, 0],
        'fukashokin': [0, 0, 0, 0],
        'grade_code': ['G1', 'G1', 'G1', 'G1']
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Needs 'is_win' internally usually
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    # Compute
    res = pipeline._compute_history_stats(df)
    
    # Check H2 R4 (Index 3).
    # Previous: R3 (Rank 10). Mean = 10.
    # If it included H1 (Rank 1, 1), mean would be small.
    
    h2_mean_rank = res.loc[res['race_id']=='R4', 'mean_rank_5'].values[0]
    logger.info(f"H2 (R4) MeanRank5: {h2_mean_rank}")
    
    if np.isclose(h2_mean_rank, 10.0):
        logger.info("PASS: H2 stats independent of H1.")
    else:
        logger.error(f"FAIL: H2 stats contaminated? {h2_mean_rank}")
        sys.exit(1)

def test_history_future_leak():
    logger.info("--- Test: History Stats Future Leak ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    
    # H1: R1 (Rank 10), R2 (Rank 1).
    # At R1: No history (NaN).
    # At R2: History is R1 (Rank 10). MeanRank5 should be 10.0.
    # It should NOT see R2's Rank 1.
    
    data = {
        'race_id': ['R1', 'R2'],
        'horse_id': ['H1', 'H1'],
        'horse_number': [1, 1],
        'date': ['2024-01-01', '2024-01-02'],
        'rank': [10, 1],
        'time_diff': [1.0, 0.0],
        'honshokin': [0, 0],
        'fukashokin': [0, 0],
        'grade_code': ['G1', 'G1']
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    res = pipeline._compute_history_stats(df)
    
    r2_mean = res.loc[res['race_id']=='R2', 'mean_rank_5'].values[0]
    logger.info(f"H1 (R2) MeanRank5: {r2_mean}")
    
    if np.isclose(r2_mean, 10.0):
        logger.info("PASS: Future (Current) val not included.")
    elif np.isclose(r2_mean, 5.5): # (10+1)/2 => Self included
        logger.error("FAIL: Self/Future leak detected (Current race included in history).")
        sys.exit(1)
    else:
        logger.error(f"FAIL: Unexpected value {r2_mean}")
        sys.exit(1)

def test_pace_boundary():
    logger.info("--- Test: Pace Stats Boundary (Horse ID) ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    # nige_rate uses is_nige (passing_rank==1)
    # H1: R1(Nige), R2(Nige) -> Rate 1.0 at R2 ?? No, R2 prev is R1.
    # H2: R3(Not), R4(Not) -> Rate 0.0 at R4.
    
    data = {
        'race_id': ['R1', 'R2', 'R3', 'R4'],
        'horse_id': ['H1', 'H1', 'H2', 'H2'], 
        'horse_number': [1, 1, 1, 1],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'passage_rate': ["1-1-1", "1-1-1", "2-2-2", "2-2-2"], # String format expected
        'time': [100, 100, 100, 100],
        'last_3f': [35, 35, 35, 35]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare internal columns if needed (DataCleanser usually does this)
    # FeaturePipeline expects 'passage_rate' or 'corner_1'.
    
    res = pipeline._compute_pace_stats(df)
    
    # H2 R4: Prev is R3 (Rank 2 -> Not Nige). Rate = 0/1 = 0.0.
    h2_rate = res.loc[res['race_id']=='R4', 'last_nige_rate'].values[0]
    logger.info(f"H2 (R4) LastNigeRate: {h2_rate}")
    
    if h2_rate == 0.0:
        logger.info("PASS: H2 stats independent/Correct.")
    else:
        logger.error(f"FAIL: {h2_rate} != 0.0")
        sys.exit(1)

def test_bloodline_boundary():
    logger.info("--- Test: Bloodline Stats Boundary (Sire ID) ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    # S1: R1(Win), R2(Win).
    # S2: R3(Lose), R4(Lose).
    # R3 (S2) should NOT inherit stats from R1, R2.
    # 'sire_win_sum'
    
    data = {
        'race_id': ['R1', 'R2', 'R3', 'R4'],
        'horse_id': ['H1', 'H2', 'H3', 'H4'], # Different horses
        'sire_id': ['S1', 'S1', 'S2', 'S2'],
        'horse_number': [1, 1, 1, 1],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'rank': [1, 1, 4, 4], # Added rank
        'surface': ['芝', '芝', '芝', '芝'] # '芝' instead of 'Turf' if pipeline expects raw? 
        # Actually pipeline checks: df_sorted['is_turf'] = (df_sorted['surface'] == '芝')
        # So I should use Japanese '芝'.
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    res = pipeline._compute_bloodline_stats(df)
    
    # R3 (S2 First Race): win_sum should be 0. n_races=0. Rate=0.
    # R4 (S2 Second Race): win_sum should be 0 (R3 was Lose). n_races=1. Rate=0.
    
    r3_val = res.loc[res['race_id']=='R3', 'sire_win_rate'].values[0]
    logger.info(f"S2 (R3 First) SireWinRate: {r3_val}")
    
    if r3_val == 0.0:
        logger.info("PASS: S2 independent of S1.")
    else:
        logger.error(f"FAIL: S2 started with {r3_val}. Leakage from S1?")
        sys.exit(1)

def test_pace_pressure_leakage():
    logger.info("--- Test: Pace Pressure Stats (Self-Exclusion & Boundary) ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    
    # R1: H1 (Nige), H2 (Nige), H3 (Not)
    # H1: LastNigeRate should be 0.
    # H2: LastNigeRate should be 0.
    
    # R2: H1, H2, H3
    # H1: LastNigeRate (from R1) = 1.0 -> IsCandidate=1
    # H2: LastNigeRate (from R1) = 1.0 -> IsCandidate=1
    # H3: LastNigeRate = 0.
    
    # Expected at R2:
    # RaceNigeCountBin = H1(1) + H2(1) + H3(0) = 2.
    # H1 Interaction: (Count(2) - IsCand(1)) * Rate(1.0) = 1 * 1.0 = 1.0
    # H2 Interaction: (Count(2) - IsCand(1)) * Rate(1.0) = 1 * 1.0 = 1.0
    # H3 Interaction: (Count(2) - IsCand(0)) * Rate(0.0) = 2 * 0.0 = 0.0
    
    data = {
        'race_id': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2'],
        'horse_id': ['H1', 'H2', 'H3', 'H1', 'H2', 'H3'],
        'horse_number': [1, 2, 3, 1, 2, 3],
        'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-02'],
        'pass_1': [1, 1, 2, 1, 1, 2] # Use pass_1 directly as raw load would
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    res = pipeline._compute_pace_pressure_stats(df)
    
    # Check H1 at R2
    row_h1_r2 = res[(res['race_id']=='R2') & (res['horse_id']=='H1')].iloc[0]
    
    logger.info(f"H1 (R2) LastNigeRate: {row_h1_r2['last_nige_rate']}")
    logger.info(f"H1 (R2) RaceNigeCountBin: {row_h1_r2['race_nige_count_bin']}")
    logger.info(f"H1 (R2) IsNigeInteraction: {row_h1_r2['is_nige_interaction']}")
    
    if not np.isclose(row_h1_r2['last_nige_rate'], 1.0):
        logger.error("FAIL: H1 LastNigeRate incorrect.")
        sys.exit(1)
        
    if row_h1_r2['race_nige_count_bin'] != 2:
         logger.error(f"FAIL: RaceNigeCountBin incorrect. Expected 2, got {row_h1_r2['race_nige_count_bin']}")
         sys.exit(1)
         
    # Self exclusion check: (2 - 1) * 1.0 = 1.0
    if not np.isclose(row_h1_r2['is_nige_interaction'], 1.0):
        logger.error(f"FAIL: Interaction incorrect (SelfExclusion fail?). Expected 1.0, got {row_h1_r2['is_nige_interaction']}")
        sys.exit(1)

    if not np.isclose(row_h1_r2['is_nige_interaction'], 1.0):
        logger.error(f"FAIL: Interaction incorrect (SelfExclusion fail?). Expected 1.0, got {row_h1_r2['is_nige_interaction']}")
        sys.exit(1)

    logger.info("PASS: Pace Pressure Self-Exclusion & Calculation correct.")

def test_relative_stats():
    logger.info("--- Test: Relative Stats (Group Isolation & Z-Score) ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    
    # R1: H1(Impost 55), H2(Impost 57) -> Mean 56, Std 1.414
    # H1: (55-56)/1.414 = -0.707
    # H2: (57-56)/1.414 = +0.707
    
    # R2: H3(Impost 55), H4(Impost 55) -> Mean 55, Std 0
    # H3: Z=0 (Handle Std=0)
    
    data = {
        'race_id': ['R1', 'R1', 'R2', 'R2'],
        'horse_id': ['H1', 'H2', 'H3', 'H4'],
        'horse_number': [1, 2, 1, 2],
        'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'impost': [55.0, 57.0, 55.0, 55.0],
        'time': [0,0,0,0], # Dummy
        'distance': [1600, 1600, 1600, 1600],
        'venue': ['01', '01', '01', '01'],
        'surface': ['T', 'T', 'T', 'T']
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    res = pipeline._compute_relative_stats(df)
    
    # Check R1
    r1 = res[res['race_id']=='R1']
    h1_z = r1.loc[r1['horse_id']=='H1', 'relative_impost_z'].values[0]
    if not np.isclose(abs(h1_z), 0.707106, atol=1e-4):
        logger.error(f"FAIL: R1 Z-score incorrect. Got {h1_z}")
        sys.exit(1)
        
    # Check R2 (Std=0)
    r2 = res[res['race_id']=='R2']
    h3_z = r2.loc[r2['horse_id']=='H3', 'relative_impost_z'].values[0]
    if h3_z != 0.0:
        logger.error(f"FAIL: R2 Z-score (Std=0) incorrect. Expected 0.0, got {h3_z}")
        sys.exit(1)
        
    logger.info("PASS: Relative Stats calculation correct.")

def test_jockey_trainer_stats():
    logger.info("--- Test: Jockey x Trainer Stats (Smoothing & Self-Exclusion) ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    
    # Pair (J1, T1)
    # Race 1: Win (Rank 1)
    # Race 2: Win (Rank 1)
    # Expected at Race 2:
    # RunCount = 1
    # Top3Sum = 1 (from Race 1)
    # GlobalTop3Rate = (2/2) = 1.0 (for this toy data)
    # jt_top3_rate_smoothed = (1 + 20*1.0) / (1 + 20) = 21/21 = 1.0
    
    data = {
        'race_id': ['R1', 'R2', 'R3'],
        'horse_id': ['H1', 'H1', 'H2'],
        'horse_number': [1, 1, 1],
        'jockey_id': ['J1', 'J1', 'J2'],
        'trainer_id': ['T1', 'T1', 'T2'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'rank': [1, 1, 10]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    res = pipeline._compute_jockey_trainer_stats(df)
    
    # Check J1xT1 at R2
    r2 = res[res['race_id']=='R2'].iloc[0]
    logger.info(f"J1xT1 (R2) RunCount: {r2['jt_run_count']}")
    logger.info(f"J1xT1 (R2) Top3RateSmooth: {r2['jt_top3_rate_smoothed']}")
    
    if r2['jt_run_count'] != 1:
        logger.error(f"FAIL: RunCount should be 1, got {r2['jt_run_count']}")
        sys.exit(1)
        
    # Check global rate used in smoothed:
    # rank: [1, 1, 10] -> is_top3: [1, 1, 0] -> mean = 2/3 = 0.666
    # smoothed = (1 + 20 * (2/3)) / (1 + 20) = (1 + 13.333) / 21 = 14.333 / 21 = 0.6825
    expected_smooth = (1 + 20 * (2/3)) / (1 + 20)
    if not np.isclose(r2['jt_top3_rate_smoothed'], expected_smooth):
        logger.error(f"FAIL: Smoothed rate incorrect. Expected {expected_smooth}, got {r2['jt_top3_rate_smoothed']}")
        sys.exit(1)
        
    # Check Isolation: J2xT2 at R3
    # Prev count for J2xT2 should be 0.
    r3 = res[res['race_id']=='R3'].iloc[0]
    if r3['jt_run_count'] != 0:
        logger.error(f"FAIL: J2xT2 RunCount should be 0, got {r3['jt_run_count']}")
        sys.exit(1)
        
    logger.info("PASS: Jockey x Trainer Stats calculation and smoothing correct.")


if __name__ == "__main__":
    test_history_boundary()
    test_history_future_leak()
    test_pace_boundary()
    test_bloodline_boundary()
    test_pace_pressure_leakage()
    test_relative_stats()
    test_jockey_trainer_stats()
    logger.info("All general leakage tests passed.")
