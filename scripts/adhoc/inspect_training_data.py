import pandas as pd
import pickle
import os

print("=" * 80)
print("学習データ形式の確認")
print("=" * 80)

# 1. Parquet (前処理済みデータ) - 基本情報のみ
parquet_path = 'data/processed/preprocessed_data_v10_leakfix.parquet'
if os.path.exists(parquet_path):
    print(f"\n[1] Parquet File: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  - Shape: {df.shape} (rows, columns)")
    print(f"  - Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"\n  - Column categories:")
    
    # カラムをカテゴリ分け
    id_cols = [c for c in df.columns if 'id' in c.lower() or c in ['race_id', 'horse_name']]
    meta_cols = [c for c in df.columns if c in ['date', 'venue', 'race_number', 'distance', 'surface', 'weather', 'state', 'class_level', 'frame_number', 'horse_number']]
    target_cols = [c for c in df.columns if c in ['rank', 'target', 'time', 'odds', 'popularity']]
    lag_cols = [c for c in df.columns if c.startswith('lag')]
    stat_cols = [c for c in df.columns if any(x in c for x in ['_win_rate', '_top3_rate', '_avg_', '_count', '_n_races'])]
    emb_cols = [c for c in df.columns if '_emb_' in c]
    other_cols = [c for c in df.columns if c not in id_cols + meta_cols + target_cols + lag_cols + stat_cols + emb_cols]
    
    print(f"    - ID columns ({len(id_cols)}): {id_cols[:5]}...")
    print(f"    - Meta columns ({len(meta_cols)}): {meta_cols[:5]}...")
    print(f"    - Target columns ({len(target_cols)}): {target_cols}")
    print(f"    - Lag features ({len(lag_cols)}): {lag_cols[:3]}...")
    print(f"    - Stats features ({len(stat_cols)}): {stat_cols[:3]}...")
    print(f"    - Embedding features ({len(emb_cols)}): {emb_cols[:3]}...")
    print(f"    - Other features ({len(other_cols)}): {other_cols[:5]}...")

# 2. Pickle (LGBM用データセット)
pickle_path = 'data/processed/lgbm_datasets_v10_leakfix.pkl'
if os.path.exists(pickle_path):
    print(f"\n[2] LGBM Dataset: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    
    for key in ['train', 'valid', 'test']:
        if key in datasets:
            data = datasets[key]
            print(f"\n  [{key.upper()}]")
            print(f"    - X shape: {data['X'].shape}")
            print(f"    - y shape: {data['y'].shape}")
            print(f"    - Feature count: {len(data['X'].columns)}")
    
    # 特徴量リスト（最初の50個）
    if 'train' in datasets:
        print(f"\n  [Feature List (first 50)]:")
        for i, col in enumerate(datasets['train']['X'].columns[:50]):
            print(f"    {i+1:3}. {col}")
        if len(datasets['train']['X'].columns) > 50:
            print(f"    ... (Total: {len(datasets['train']['X'].columns)} features)")

print("\n" + "=" * 80)
