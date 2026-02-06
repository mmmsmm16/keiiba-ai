import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# プロジェクトルートへのパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

def diagnose():
    print("=== Learning Diagnosis ===")
    
    # 1. データのロード (直近1年分で十分)
    print("Loading Raw Data (2023-2024)...")
    loader = JraVanDataLoader()
    raw_df = loader.load(history_start_date='2023-01-01', end_date='2024-12-31')
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. 特徴量パイプライン実行
    print("Generating Features...")
    pipeline = FeaturePipeline(cache_dir="data/features")
    # v02で使用しているブロックを指定
    features_df = pipeline.load_features(clean_df, ['base_attributes', 'history_stats'])
    
    print(f"Features Generated: {len(features_df)} rows")
    
    # 3. ターゲットの結合 (マージ検証)
    # clean_dfにある正解データ 'rank' を結合
    # キー: race_id, horse_number (これらがズレていると相関が出ない)
    target_df = clean_df[['race_id', 'horse_number', 'rank']].copy()
    
    # 3着以内フラグ (Binary Target for AUC)
    target_df['target_binary'] = (target_df['rank'] <= 3).astype(int)
    
    merged_df = pd.merge(features_df, target_df, on=['race_id', 'horse_number'], how='inner')
    print(f"Merged for Analysis: {len(merged_df)} rows")
    
    # 4. データの健全性チェック
    if 'lag1_rank' not in merged_df.columns:
        print("❌ Error: 'lag1_rank' column not found in features.")
        return

    # 欠損除去
    valid_data = merged_df.dropna(subset=['lag1_rank', 'rank'])
    print(f"Valid Rows (non-null lag1_rank): {len(valid_data)}")

    # 相関係数
    corr = valid_data[['lag1_rank', 'rank']].corr().iloc[0, 1]
    print(f"\nCorrelation (lag1_rank vs rank): {corr:.4f}")
    
    if abs(corr) < 0.05:
        print("⚠️ CRITICAL WARNING: 相関がほぼゼロです。マージキー(race_id, horse_number)の不整合か、シフト処理のミスの可能性が高いです。")
    else:
        print("✅ Correlation OK. データには相関があります。")
        
    # 5. ヒューリスティックAUC (モデルなしでの予測力)
    # 「前走着順が良い(小さい)ほど、今回も勝つ」 -> 負の値をスコアにする
    y_true = valid_data['target_binary']
    y_score = -valid_data['lag1_rank'] 
    
    heuristic_auc = roc_auc_score(y_true, y_score)
    print(f"\n[Heuristic AUC] (lag1_rank only): {heuristic_auc:.4f}")
    
    print("-" * 30)
    if heuristic_auc > 0.6:
        print("👉 結論: 【データは正常】です。")
        print("   原因は 'LightGBMの学習パラメータ(LambdaRank設定)' にあります。")
        print("   (groupの作り方、ラベルの設定、metric等を見直してください)")
    else:
        print("👉 結論: 【データが破損】しています。")
        print("   特徴量作成ロジック、特にマージ処理を見直してください。")

if __name__ == "__main__":
    diagnose()
