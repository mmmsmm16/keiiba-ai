
import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data():
    data = {
        'race_id': ['R01', 'R02', 'R03', 'R04', 'R05', 'R06'],
        'horse_id': ['H01', 'H02', 'H01', 'H02', 'H01', 'H03'],
        'horse_number': [1, 2, 1, 2, 1, 1],
        'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03'],
        'venue': ['Tokyo', 'Tokyo', 'Tokyo', 'Tokyo', 'Tokyo', 'Nakayama'], 
        'distance': [1600, 1600, 1600, 1600, 1600, 1600],
        'surface': ['Turf', 'Turf', 'Turf', 'Turf', 'Turf', 'Turf'],
        'time': [96.0, 97.0, 95.0, 96.5, 94.0, 100.0],
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

def test_future_independence():
    logger.info("--- Test 1: Future Independence ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    
    df = create_dummy_data()
    si_orig = pipeline._compute_speed_index_stats(df)
    
    # Modify R05 (Future). Check R04 (Past). 
    # R04 depends on R01, R02, R03? No, R04 is H02.
    # H02 History: R02.
    # Wait, H02 races: R02(Date1), R04(Date2).
    # R04 avg_speed_index depends on SI(R02).
    
    # H01 races: R01, R03, R05.
    # R03 avg_speed_index depends on SI(R01).
    # R05 avg_speed_index depends on SI(R01), SI(R03).
    
    # If I change Time(R05), will SI(R03) change?
    # SI(R03) uses Expanding Mean/Std at R03 (History: R01, R02).
    # R05 is in future. Should NOT affect Expanding Mean/Std at R03.
    # So SI(R03) is stable.
    # R05 (H01) modification should not affect R03 (H01) avg_speed_index.
    
    df_mod = df.copy()
    df_mod.loc[df_mod['race_id'] == 'R05', 'time'] = 60.0 
    
    si_mod = pipeline._compute_speed_index_stats(df_mod)
    
    # Check R03 (H01, index 2)
    val_orig = si_orig.loc[si_orig['race_id'] == 'R03', 'avg_speed_index'].values[0]
    val_mod = si_mod.loc[si_mod['race_id'] == 'R03', 'avg_speed_index'].values[0]
    
    logger.info(f"Target (R03) avg_speed_index: Orig={val_orig}, Mod={val_mod}")
    
    if np.isclose(val_orig, val_mod):
        logger.info("PASS: Future change did not affect past row.")
    else:
        logger.error(f"FAIL: Future change affected past row! ({val_orig} vs {val_mod})")
        sys.exit(1)

def test_logic_check():
    logger.info("--- Test 2: Calculation Logic & Self Exclusion ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    df = create_dummy_data()
    # R01(H01), R02(H02), R03(H01), R04(H02), R05(H01).
    # Time: 96, 97, 95, 96.5, 94.
    
    # SI(R01): Hist=[]. SI=0.
    # SI(R02): Hist=[96]. Std=NaN. SI=0.
    # SI(R03): Hist=[96, 97]. Mean=96.5, Std=0.707. Time=95. SI=(96.5-95)/0.707 = 2.12.
    # SI(R04): Hist=[96, 97, 95]. Mean=96. Std=1.0. Time=96.5. SI=(96-96.5)/1 = -0.5.
    
    # H01 History: R01, R03, R05.
    # R01 avg_SI: NaN -> 0.
    # R03 avg_SI: Mean(SI(R01)) = 0.
    # R05 avg_SI: Mean(SI(R01), SI(R03)) = Mean(0, 2.12) = 1.06.
    
    si = pipeline._compute_speed_index_stats(df)
    
    r05_avg = si.loc[si['race_id'] == 'R05', 'avg_speed_index'].values[0]
    logger.info(f"R05 avg_speed_index: {r05_avg} (Expected ~1.06)")
    
    if np.isclose(r05_avg, 1.06, atol=0.05): # approx
        logger.info("PASS: Value matches 'Past Only' calculation.")
    else:
        logger.error(f"FAIL: Value mismatch. Got {r05_avg}.")
        sys.exit(1)

def test_group_isolation():
    logger.info("--- Test 3: Group Isolation ---")
    pipeline = FeaturePipeline(cache_dir="tmp/cache_test")
    df = create_dummy_data()
    
    # R06 (Nakayama) H03.
    # History for H03: None. avg_SI should be 0.
    # Also SI(R06) calculation:
    # Group Nakayama/1600/Turf. History: None.
    # So Expanding Mean/Std undefined -> SI=0.
    
    si = pipeline._compute_speed_index_stats(df)
    r06_avg = si.loc[si['race_id'] == 'R06', 'avg_speed_index'].values[0]
    
    logger.info(f"R06 avg_speed_index: {r06_avg}")
    
    if r06_avg == 0.0:
        logger.info("PASS: New group started with 0.0.")
    else:
        logger.error(f"FAIL: New group has non-zero value {r06_avg}.")
        sys.exit(1)

if __name__ == "__main__":
    test_future_independence()
    test_logic_check()
    test_group_isolation()
    logger.info("All tests passed successfully.")
