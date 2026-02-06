import pandas as pd
import numpy as np
import sys
import os

# srcをパスに追加
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.bloodline_features import BloodlineFeatureEngineer

def verify_bloodline_features():
    print("=== Bloodline Feature Verification (Phase 18) ===")
    
    engineer = BloodlineFeatureEngineer()
    
    # Inject Mock Data into bloodline_map (Pre-loaded)
    engineer.bloodline_map = pd.DataFrame({
        'horse_id': ['H1', 'H2', 'H3', 'H4', 'H5'],
        'sire_id': ['S1', 'S1', 'S1', 'S2', 'S2'],
        'mare_id': ['M1', 'M2', 'M3', 'M4', 'M5'],
        'bms_id': ['B1', 'B2', 'B3', 'B4', 'B5']
    })
    
    # Test Scenarios
    # Day 1: 2024-01-01
    # - R1: H1 (Sire S1) -> Win (1st)
    # - R1: H2 (Sire S1) -> Lose (4th)  <-- Same Race Leak Check
    # - R2: H4 (Sire S2) -> Win (1st)
    
    # Day 2: 2024-01-02
    # - R3: H3 (Sire S1) -> Expects Stats from Day 1 (1 Win, 2 Starts)
    # - R3: H5 (Sire S2) -> Expects Stats from Day 1 (1 Win, 1 Start)
    
    data = {
        'race_id': ['R1', 'R1', 'R2', 'R3', 'R3'],
        'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'horse_id': ['H1', 'H2', 'H4', 'H3', 'H5'],
        'distance': [1600, 1600, 2000, 1600, 1200], # M, M, I, M, S
        'surface': ['芝', '芝', 'ダート', '芝', 'ダート'],
        'course_id': ['C1', 'C1', 'C2', 'C1', 'C2'],
        'rank': [1, 4, 1, 2, 2]
    }
    df = pd.DataFrame(data)
    
    # Note: verify script must convert date because add_features does it locally or expects datetime?
    # add_features does: df['date'] = pd.to_datetime(df['date'])
    # So string input is fine.
    
    print("Input Data:")
    print(df[['race_id', 'horse_id', 'date', 'rank']].to_string())
    
    # Run
    df_out = engineer.add_features(df.copy())
    
    print("\nFeature Generation Complete.")
    
    # --- Check 1: Same Day Leak Prevention (H2 in R1) ---
    # H2 is in R1 with H1. H1 Won.
    # H2 should NOT see H1's Win.
    # H2 should see Prior only.
    
    row_h2 = df_out[(df_out['horse_id'] == 'H2')].iloc[0]
    rate_h2 = row_h2['sire_overall_win_rate']
    count_h2 = row_h2['sire_overall_n_samples']
    
    # Prior: (0 + 30*0.08) / (0 + 30) = 2.4/30 = 0.08
    print(f"\n[Check 1] H2 (Day 1) Sire Rate: {rate_h2:.4f} (Count: {count_h2})")
    
    if count_h2 == 0 and abs(rate_h2 - 0.08) < 0.001:
        print("✅ PASS: H2 does not see H1's win (Count 0, Prior Rate).")
    else:
        print(f"❌ FAIL: H2 saw something! Count={count_h2}, Rate={rate_h2}")

    # --- Check 2: Next Day Stats (H3 in R3) ---
    # H3 is Day 2. Should see Day 1 Results for S1.
    # S1 Day 1: H1 (Win), H2 (Lose). -> 1 Win, 2 Starts.
    # Smoothed: (1 + 30*0.08) / (2 + 30) = (1 + 2.4) / 32 = 3.4 / 32 = 0.10625
    
    row_h3 = df_out[(df_out['horse_id'] == 'H3')].iloc[0]
    rate_h3 = row_h3['sire_overall_win_rate']
    count_h3 = row_h3['sire_overall_n_samples']
    
    print(f"\n[Check 2] H3 (Day 2) Sire Rate: {rate_h3:.4f} (Count: {count_h3})")
    
    expected_rate = (1 + 30*0.08) / (2 + 30)
    if count_h3 == 2 and abs(rate_h3 - expected_rate) < 0.001:
        print(f"✅ PASS: H3 sees Day 1 stats correctly ({count_h3} starts).")
    else:
        print(f"❌ FAIL: Expected Count 2, Rate {expected_rate:.4f}. Got Count={count_h3}, Rate={rate_h3}")

    # --- Check 3: Distance Type Logic ---
    # H3 (1600m) -> Mile. S1 Day 1 at Mile (H1, H2) -> 1 Win / 2 Starts.
    row_h3_dist = row_h3['sire_distance_win_rate']
    count_h3_dist = row_h3['sire_distance_n_samples'] # if column exists? I added _n_samples
    
    print(f"\n[Check 3] H3 Distance (Mile) Rate: {row_h3_dist:.4f}")
    if abs(row_h3_dist - expected_rate) < 0.001:
        print("✅ PASS: Distance Stats Correct (Mile).")
    else:
        print(f"❌ FAIL: Distance Stats Mismatch.")
        
    # --- Check 4: Course Logic ---
    # H3 (C1). S1 Day 1 at C1 (H1, H2) -> 1 Win / 2 Starts.
    # Same expectation.
    row_h3_course = row_h3['sire_course_win_rate']
    if abs(row_h3_course - expected_rate) < 0.001:
        print("✅ PASS: Course Stats Correct.")
    else:
        print("❌ FAIL: Course Stats Mismatch.")
        
    # --- Check 5: Surface Logic ---
    # H3 (Turf). S1 Day 1 at Turf (H1, H2) -> 1 Win / 2 Starts.
    row_h3_surf = row_h3['sire_surface_win_rate']
    if abs(row_h3_surf - expected_rate) < 0.001:
        print("✅ PASS: Surface Stats Correct.")
    else:
        print("❌ FAIL: Surface Stats Mismatch.")

if __name__ == "__main__":
    verify_bloodline_features()
