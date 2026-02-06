"""
Analyze 2026 Performance by Race Class (Final Fix)
"""
import pandas as pd
import numpy as np
import os
import joblib
import sys
from datetime import datetime, timedelta

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

V13_MODEL_PATH = 'models/experiments/exp_lambdarank_hard_weighted/model.pkl'
V13_FEATS_PATH = 'models/experiments/exp_lambdarank_hard_weighted/features.csv'

def get_class_name(grade, joken):
    if grade in ['1', '2', '3']: return "重賞 (G1-G3)"
    if grade in ['4', '5']: return "オープン (L/OP)"
    j = str(joken)
    if '000' in j or j == '0': return "新馬・未勝利"
    if '701' in j: return "1勝クラス"
    if '702' in j: return "2勝クラス"
    if '703' in j: return "3勝クラス"
    if j == '999': return "オープン"
    return "その他"

def main():
    print("Loading 2026 Data (Jan 1 - Jan 24) from DB...")
    loader = JraVanDataLoader()
    df_db = loader.load(history_start_date='2026-01-01', end_date='2026-01-25', skip_odds=True)
    
    if df_db.empty:
        print("No 2026 data found."); return

    # Preserve RAW meta for grouping
    raw_meta = df_db[['race_id', 'horse_number', 'odds', 'rank', 'grade_code', 'kyoso_joken_code']].copy()
    raw_meta['race_id'] = raw_meta['race_id'].astype(str)
    raw_meta['horse_number'] = raw_meta['horse_number'].astype(int)

    pipeline = FeaturePipeline(cache_dir='data/features_v14/prod_cache')
    df_feat = pipeline.load_features(df_db, list(pipeline.registry.keys()))
    df_feat['race_id'] = df_feat['race_id'].astype(str)
    df_feat['horse_number'] = df_feat['horse_number'].astype(int)

    print(f"Applying V13 Model to {len(df_feat)} records...")
    model = joblib.load(V13_MODEL_PATH)
    feats = pd.read_csv(V13_FEATS_PATH, header=None).iloc[:, 0].tolist()
    if feats[0] == '0' or feats[0] == 'feature': feats = feats[1:]
    
    X = df_feat.reindex(columns=feats, fill_value=-999.0).fillna(-999.0)
    df_feat['score'] = model.predict_proba(X)[:, -1] if hasattr(model, 'predict_proba') else model.predict(X)
    
    # Merge for Accuracy calculation
    df = pd.merge(df_feat[['race_id', 'horse_number', 'score']], raw_meta, on=['race_id', 'horse_number'], how='inner')
    
    # Rank
    df['v13_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first')
    df['actual_rank'] = pd.to_numeric(df['rank'], errors='coerce').fillna(99).astype(int)
    df['is_win'] = (df['actual_rank'] == 1).astype(int)
    df['is_top3'] = (df['actual_rank'] <= 3).astype(int)
    df['race_class'] = [get_class_name(g, j) for g, j in zip(df['grade_code'], df['kyoso_joken_code'])]

    top1 = df[df['v13_rank'] == 1].copy()
    
    # Accuracy by Class
    class_stats = top1.groupby('race_class').agg({
        'race_id': 'count',
        'is_win': 'mean',
        'is_top3': 'mean',
        'odds': 'mean'
    }).rename(columns={'race_id': 'Races', 'is_win': 'WinRate', 'is_top3': 'Top3Rate', 'odds': 'AvgOdds'})
    
    # Win ROI
    top1_wins = top1[top1['is_win'] == 1]
    returns = top1_wins.groupby('race_class')['odds'].sum()
    class_stats['WinROI'] = (returns / class_stats['Races']) * 100
    
    print("\n=== V13 Performance by Class (2026 Jan) ===")
    print(class_stats.sort_values('WinROI', ascending=False).to_string(
        formatters={'WinRate':'{:.1%}'.format, 'Top3Rate':'{:.1%}'.format, 'WinROI':'{:.1f}%'.format}))

if __name__ == "__main__":
    main()
