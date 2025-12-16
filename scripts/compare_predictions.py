"""
予測比較レポート生成スクリプト

parquet vs InferencePreprocessor の予測結果を比較し、
daily_report 形式のマークダウンを出力

Usage:
    docker compose exec app python scripts/compare_predictions.py --date 2025-09-13
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import lightgbm as lgb
from scipy.special import expit
from sqlalchemy import create_engine

# パス
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/v13_market_residual')
PARQUET_PATH = os.path.join(os.path.dirname(__file__), '../data/processed/preprocessed_data_v11.parquet')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../reports/prediction_comparison')


def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


def load_models():
    models = []
    for fold in ['2022', '2023', '2024']:
        path = os.path.join(MODEL_DIR, f'v13_fold_{fold}.txt')
        if os.path.exists(path):
            models.append(lgb.Booster(model_file=path))
    return models


def run_prediction(df, models, source_name):
    """予測実行"""
    feature_cols = models[0].feature_name()
    
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    X = df[feature_cols].fillna(0)
    
    preds = []
    for model in models:
        preds.append(model.predict(X))
    avg_pred = np.mean(preds, axis=0)
    
    df[f'raw_score_{source_name}'] = avg_pred
    df[f'prob_raw_{source_name}'] = expit(avg_pred)
    
    def softmax_race(group):
        exp_vals = np.exp(group - group.max())
        return exp_vals / exp_vals.sum()
    
    df[f'prob_{source_name}'] = df.groupby('race_id')[f'prob_raw_{source_name}'].transform(softmax_race)
    df[f'rank_{source_name}'] = df.groupby('race_id')[f'prob_{source_name}'].rank(ascending=False)
    
    return df


def generate_report(date_str: str):
    print(f"予測比較レポート生成: {date_str}")
    
    models = load_models()
    print(f"モデル: {len(models)} folds")
    
    # ========================================
    # 1. parquet からデータ取得 (Method A)
    # ========================================
    print("Method A: parquet からデータ取得...")
    df_full = pd.read_parquet(PARQUET_PATH)
    target_date = pd.to_datetime(date_str.replace('-', ''))
    df_parquet = df_full[df_full['date'] == target_date].copy()
    
    has_parquet = not df_parquet.empty
    if has_parquet:
        print(f"  parquet: {len(df_parquet)} rows, {df_parquet['race_id'].nunique()} races")
        df_parquet = run_prediction(df_parquet, models, 'parquet')
    else:
        print("  parquet にデータなし")
    
    # ========================================
    # 2. InferencePreprocessor で生成 (Method B)
    # ========================================
    print("Method B: InferencePreprocessor で生成...")
    from inference.loader import InferenceDataLoader
    from inference.preprocessor import InferencePreprocessor
    
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor(history_path=PARQUET_PATH)
    
    target_date_str = date_str.replace('-', '')
    raw_df = loader.load(target_date=target_date_str)
    
    has_inference = not raw_df.empty
    if has_inference:
        history_df = pd.read_parquet(PARQUET_PATH)
        X, ids, full_df = preprocessor.preprocess(raw_df, history_df=history_df, return_full_df=True)
        
        df_inference = full_df.copy()
        df_inference['race_id'] = ids['race_id'].values
        print(f"  InferencePreprocessor: {len(df_inference)} rows")
        df_inference = run_prediction(df_inference, models, 'inference')
    else:
        print("  DBにデータなし")
        df_inference = pd.DataFrame()
    
    # ========================================
    # 3. レポート生成
    # ========================================
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"comparison_{date_str}.md")
    
    lines = []
    lines.append(f"# 予測比較レポート: {date_str}")
    lines.append("")
    lines.append(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # サマリー
    lines.append("## データソース")
    lines.append("")
    lines.append("| Method | Source | Status | Rows |")
    lines.append("|--------|--------|--------|------|")
    lines.append(f"| **A: parquet** | `preprocessed_data_v11.parquet` | {'✅' if has_parquet else '❌'} | {len(df_parquet) if has_parquet else 0} |")
    lines.append(f"| **B: Inference** | `InferencePreprocessor` | {'✅' if has_inference else '❌'} | {len(df_inference) if has_inference else 0} |")
    lines.append("")
    
    if not has_parquet and not has_inference:
        lines.append("> ⚠️ どちらの方式でもデータが取得できませんでした")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"レポート生成: {report_path}")
        return
    
    # スコア統計
    lines.append("## スコア統計")
    lines.append("")
    lines.append("| Metric | A: parquet | B: Inference | 差分 |")
    lines.append("|--------|------------|--------------|------|")
    
    if has_parquet and has_inference:
        mean_a = df_parquet['raw_score_parquet'].mean()
        mean_b = df_inference['raw_score_inference'].mean()
        std_a = df_parquet['raw_score_parquet'].std()
        std_b = df_inference['raw_score_inference'].std()
        lines.append(f"| Mean | {mean_a:.4f} | {mean_b:.4f} | {mean_b - mean_a:+.4f} |")
        lines.append(f"| Std | {std_a:.4f} | {std_b:.4f} | {std_b - std_a:+.4f} |")
    lines.append("")
    
    # レースごとの比較
    lines.append("---")
    lines.append("")
    lines.append("## レース別比較")
    lines.append("")
    
    # 共通のレースIDを取得
    if has_parquet:
        race_ids = sorted(df_parquet['race_id'].unique())
    else:
        race_ids = sorted(df_inference['race_id'].unique())
    
    for i, race_id in enumerate(race_ids[:10]):  # 最初の10レース
        lines.append(f"### Race {i+1}: `{race_id}`")
        lines.append("")
        
        # BOX4 比較
        lines.append("**🎯 BOX4 選出馬**")
        lines.append("")
        lines.append("| Method | Top4馬番 |")
        lines.append("|--------|---------|")
        
        if has_parquet:
            race_a = df_parquet[df_parquet['race_id'] == race_id]
            top4_a = race_a.nlargest(4, 'prob_parquet')['horse_number'].astype(int).tolist()
            lines.append(f"| A: parquet | {top4_a} |")
        
        if has_inference:
            race_b = df_inference[df_inference['race_id'] == race_id]
            if not race_b.empty:
                top4_b = race_b.nlargest(4, 'prob_inference')['horse_number'].astype(int).tolist()
                lines.append(f"| B: Inference | {top4_b} |")
        
        lines.append("")
        
        # 一致チェック
        if has_parquet and has_inference and not race_b.empty:
            match = set(top4_a) == set(top4_b)
            lines.append(f"**一致**: {'✅ 完全一致' if match else '❌ 不一致'}")
            lines.append("")
        
        # 詳細テーブル
        lines.append("**📊 予測スコア詳細**")
        lines.append("")
        
        if has_parquet and has_inference and not race_b.empty:
            # 両方のデータをマージ
            race_a = race_a[['horse_number', 'horse_name', 'raw_score_parquet', 'prob_parquet', 'rank_parquet']].copy()
            race_b = race_b[['horse_number', 'raw_score_inference', 'prob_inference', 'rank_inference']].copy()
            
            merged = race_a.merge(race_b, on='horse_number', how='outer')
            merged = merged.sort_values('rank_parquet')
            
            lines.append("| 馬番 | 馬名 | Score(A) | Score(B) | 差分 | Rank(A) | Rank(B) |")
            lines.append("|------|------|----------|----------|------|---------|---------|")
            
            for _, row in merged.head(8).iterrows():
                hn = int(row['horse_number'])
                name = str(row.get('horse_name', ''))[:10]
                score_a = row.get('raw_score_parquet', 0)
                score_b = row.get('raw_score_inference', 0)
                diff = score_b - score_a if pd.notna(score_a) and pd.notna(score_b) else 0
                rank_a = int(row.get('rank_parquet', 0)) if pd.notna(row.get('rank_parquet')) else '-'
                rank_b = int(row.get('rank_inference', 0)) if pd.notna(row.get('rank_inference')) else '-'
                
                marker = '⭐' if rank_a <= 4 else ''
                lines.append(f"| {marker}{hn} | {name} | {score_a:.3f} | {score_b:.3f} | {diff:+.3f} | {rank_a} | {rank_b} |")
        
        elif has_parquet:
            race_a = race_a.sort_values('rank_parquet')
            lines.append("| 馬番 | 馬名 | Score(A) | Prob(A) | Rank(A) |")
            lines.append("|------|------|----------|---------|---------|")
            for _, row in race_a.head(8).iterrows():
                hn = int(row['horse_number'])
                name = str(row.get('horse_name', ''))[:10]
                lines.append(f"| {hn} | {name} | {row['raw_score_parquet']:.3f} | {row['prob_parquet']:.1%} | {int(row['rank_parquet'])} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 結論
    lines.append("## 結論")
    lines.append("")
    if has_parquet and has_inference:
        lines.append("> **推奨**: parquet を使用してください。InferencePreprocessor はリアルタイム用のフォールバックです。")
    lines.append("")
    
    # 書き出し
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"レポート生成完了: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='比較日付 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    generate_report(args.date)
