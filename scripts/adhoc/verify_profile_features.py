import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.profile_features import JockeyTrainerProfileEngineer

def verify_profile_features():
    print("=== Profile Feature Verification (Phase 18-B) ===")
    
    engineer = JockeyTrainerProfileEngineer()
    
    # Mock Data
    # J1: Jockey 1, T1: Trainer 1
    # Day 1:
    # R1: H1 (J1, T1) -> Win (1st)
    # R1: H2 (J1, T2) -> Lose (4th) <= Leak Check (Same Race, Same Jockey)
    # R2: H3 (J2, T1) -> Win (1st)
    
    # Day 2:
    # R3: H4 (J1, T1) -> Expects J1xT1 stats (1/1)
    
    data = {
        'race_id': ['R1', 'R1', 'R2', 'R3'],
        'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02'],
        'horse_id': ['H1', 'H2', 'H3', 'H4'],
        'jockey_id': ['J1', 'J1', 'J2', 'J1'],
        'trainer_id': ['T1', 'T2', 'T1', 'T1'],
        'course_id': ['C1', 'C1', 'C2', 'C1'],
        'rank': [1, 4, 1, 2]
    }
    df = pd.DataFrame(data)
    
    print("Input Data:")
    print(df[['race_id', 'horse_id', 'date', 'jockey_id', 'trainer_id', 'rank']].to_string())
    
    # Run
    df_out = engineer.add_features(df.copy())
    
    print("\nFeature Generation Complete.")
    
    # Check 1: Leak Prevention (H2 in R1)
    # H2 is J1. J1 won on H1 in R1. H2 should not see it.
    row_h2 = df_out[df_out['horse_id'] == 'H2'].iloc[0]
    rate_h2 = row_h2['jockey_overall_win_rate']
    count_h2 = row_h2['jockey_overall_n_samples']
    
    print(f"\n[Check 1] H2 (Day 1) Jockey Rate: {rate_h2:.4f} (Count: {count_h2})")
    # Expected: Prior (0.08)
    if count_h2 == 0 and abs(rate_h2 - 0.08) < 0.001:
        print("✅ PASS: H2 Leaks Prevented.")
    else:
        print(f"❌ FAIL: H2 saw H1! Count={count_h2}, Rate={rate_h2}")

    # Check 2: Next Day Stats (H4 in R3)
    # J1xT1 stats.
    # Day 1: J1xT1 -> R1(H1, Win=1).
    # Count: 1. Win: 1.
    # Smoothed: (1 + 30*0.08) / (1 + 30) = 3.4 / 31 = 0.109677...
    
    row_h4 = df_out[df_out['horse_id'] == 'H4'].iloc[0]
    rate_h4 = row_h4['jockey_trainer_win_rate']
    count_h4 = row_h4['jockey_trainer_n_samples']
    
    print(f"\n[Check 2] H4 (Day 2) JxT Rate: {rate_h4:.4f} (Count: {count_h4})")
    
    expected = (1 + 30*0.08) / (1 + 30)
    if count_h4 == 1 and abs(rate_h4 - expected) < 0.001:
        print("✅ PASS: JxT Stats Correct.")
    else:
        print(f"❌ FAIL: Expected {expected:.4f}, Got {rate_h4} (Count {count_h4})")
        
    print("DONE")

if __name__ == "__main__":
    verify_profile_features()
