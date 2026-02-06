
import pandas as pd
import numpy as np
import sys
import os

# srcをパスに追加
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.pace_features import PaceFeatureEngineer

def verify_pace_features():
    print("=== Pace Feature Verification ===")
    
    # ダミーデータ作成
    # Horse A: Race 1 (High Pace), Race 2 (Slow Pace)
    # Horse B: Race 1 (High Pace), Race 2 (Slow Pace)
    
    # PCI = AvgSpeedFirst / AvgSpeedLast
    # First = (dist-600)/(time-last)
    # Last = 600/last
    
    # Race 1: 1200m. 
    # Horse A: Time 70.0, Last 35.0 (First 35.0) -> FirstDist=600, LastDist=600.
    # SpeedFirst = 600/35 = 17.14
    # SpeedLast = 600/35 = 17.14
    # PCI = 1.0
    
    # Horse B: Time 70.0, Last 34.0 (First 36.0)
    # SpeedFirst = 600/36 = 16.66
    # SpeedLast = 600/34 = 17.64
    # PCI = 16.66/17.64 = 0.944 (Slow)

    data = {
        'race_id': ['R1', 'R1', 'R2', 'R2'],
        'horse_id': ['H1', 'H2', 'H1', 'H2'],
        'horse_number': [1, 2, 1, 2],
        'date': ['2024-01-01', '2024-01-01', '2024-02-01', '2024-02-01'],
        'distance': [1200, 1200, 1200, 1200],
        'raw_time': [70.0, 70.0, 70.0, 70.0],
        'last_3f': [35.0, 34.0, 34.0, 35.0], # R1: H1=Even, H2=Slow(PCI<1). R2: H1=Slow, H2=Even
        'rank': [1, 2, 1, 2]
    }
    
    df = pd.DataFrame(data)
    print("Input Data:")
    print(df[['race_id', 'horse_id', 'date', 'last_3f', 'rank']])
    
    # 特徴量生成
    engineer = PaceFeatureEngineer()
    df_out = engineer.add_features(df.copy())
    
    print("\nOutput Columns:", df_out.columns.tolist())
    
    # 1. Leak Check
    if 'raw_pci' in df_out.columns:
        print("❌ FAIL: raw_pci column exists (Leak!)")
    else:
        print("✅ PASS: raw_pci column removed")
        
    if 'raw_rpci' in df_out.columns:
        print("❌ FAIL: raw_rpci column exists (Leak!)")
    else:
        print("✅ PASS: raw_rpci column removed")

    # 2. Lag Check (Race 2 should have Race 1's PCI)
    # H1 Race 1 PCI = 1.0
    # H1 Race 2 Lag1 PCI should be 1.0
    
    h1_r2 = df_out[(df_out['horse_id'] == 'H1') & (df_out['race_id'] == 'R2')].iloc[0]
    lag1_pci = h1_r2['lag1_pci']
    
    print(f"\nH1 Race 2 Lag1 PCI: {lag1_pci}")
    if abs(lag1_pci - 1.0) < 0.01:
        print("✅ PASS: lag1_pci is correct")
    else:
        print(f"❌ FAIL: lag1_pci mismatch (Expected 1.0, Got {lag1_pci})")

    # 3. ERPCI Check
    # Race 2 ERPCI
    # Members: H1, H2
    # H1 mean_pci_5 (at R2) = 1.0 (from R1)
    # H2 mean_pci_5 (at R2) = 0.944 (from R1)
    # ERPCI Mean = (1.0 + 0.944) / 2 = 0.972
    
    erpci_r2 = h1_r2['erpci_mean']
    expected_erpci = (1.0 + (16.66/17.64)) / 2
    
    print(f"\nR2 ERPCI Mean: {erpci_r2} (Expected: {expected_erpci:.4f})")
    if abs(erpci_r2 - expected_erpci) < 0.01:
        print("✅ PASS: erpci_mean is correct")
    else:
        print(f"❌ FAIL: erpci_mean mismatch")

    # 4. Mismatch Check
    # H1 Mismatch = H1 Mean - ERPCI = 1.0 - 0.972 = 0.028
    mismatch = h1_r2['pace_mismatch']
    expected_mismatch = 1.0 - expected_erpci
    print(f"\nH1 Pace Mismatch: {mismatch} (Expected: {expected_mismatch:.4f})")
    if abs(mismatch - expected_mismatch) < 0.01:
        print("✅ PASS: pace_mismatch is correct")
    else:
        print("❌ FAIL: pace_mismatch mismatch")

if __name__ == "__main__":
    verify_pace_features()
