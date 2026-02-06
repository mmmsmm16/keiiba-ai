"""
v14 ROIモデル vs v12 比較評価スクリプト
- v14: PyTorch ROI最適化モデル
- v12: LightGBM+CatBoost+TabNet Ensemble
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from collections import defaultdict
from scipy.special import softmax
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# パス設定
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, project_root)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payouts(year):
    engine = get_db_engine()
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen = '{year}'")
    df = pd.read_sql(query, engine)
    
    df['race_id'] = (
        df['kaisai_nen'].astype(str) +
        df['keibajo_code'].astype(str) +
        df['kaisai_kai'].astype(str) +
        df['kaisai_nichime'].astype(str) +
        df['race_bango'].astype(str)
    )
    return df

def build_payout_map(pay_df):
    payout_map = defaultdict(lambda: {'tansho': {}})
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        for i in range(1, 4):
            col_a = f'haraimodoshi_tansho_{i}a'
            col_b = f'haraimodoshi_tansho_{i}b'
            if col_a in row and row[col_a] and str(row[col_a]).strip():
                try:
                    key = str(row[col_a]).strip()
                    val = int(float(str(row[col_b]).strip()))
                    payout_map[rid]['tansho'][key] = val
                except:
                    pass
    return dict(payout_map)

def evaluate_model_predictions(df, model_name, payout_map=None):
    """モデル予測を評価"""
    results = {
        'model': model_name,
        'races': 0,
        'hits': 0,
        'cost': 0,
        'return': 0
    }
    
    for race_id, grp in df.groupby('race_id'):
        if 'score' not in grp.columns or grp['score'].isnull().all():
            continue
        
        # Top1予測
        sorted_g = grp.sort_values('score', ascending=False)
        top1 = sorted_g.iloc[0]
        
        results['races'] += 1
        results['cost'] += 100
        
        # 的中判定
        rank = top1.get('rank', 99)
        if pd.isna(rank):
            rank = 99
        
        if rank == 1:
            results['hits'] += 1
            odds = top1.get('odds', 0)
            if pd.isna(odds):
                odds = 0
            results['return'] += odds * 100
    
    # 指標計算
    if results['cost'] > 0:
        results['roi'] = results['return'] / results['cost'] * 100
        results['accuracy'] = results['hits'] / results['races'] * 100
    else:
        results['roi'] = 0
        results['accuracy'] = 0
    
    return results

def predict_with_v12(df, feature_cols):
    """v12モデルで予測"""
    from src.model.ensemble import EnsembleModel
    
    model = EnsembleModel()
    # CPUモードでロード（CUDAエラー回避）
    model.load_model('experiments/v12_tabnet_revival/models/ensemble.pkl', device_name='cpu')
    
    # 欠損カラム補完
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    X = df[feature_cols]
    scores = model.predict(X)
    df['score'] = scores
    
    return df

def predict_with_v14(df, feature_cols):
    """v14 ROIモデルで予測"""
    import torch
    from src.model.roi_model import ROIModel
    
    model = ROIModel()
    model.load('experiments/v14_roi/models/roi_model_best.pt')
    
    # 数値型カラムのみ使用
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    actual_feature_cols = numeric_df.columns.tolist()
    
    # モデルの入力次元に合わせる
    if model.input_dim != len(actual_feature_cols):
        logger.warning(f"Feature mismatch: model expects {model.input_dim}, got {len(actual_feature_cols)}")
        # 足りないカラムは0埋め
        diff = model.input_dim - len(actual_feature_cols)
        if diff > 0:
            for i in range(diff):
                actual_feature_cols.append(f'_dummy_{i}')
                df[f'_dummy_{i}'] = 0
    
    # レース単位で予測
    all_scores = []
    
    for race_id, grp in df.groupby('race_id'):
        grp = grp.sort_values('horse_number')
        
        # 3D形式に変換
        X = grp[actual_feature_cols[:model.input_dim]].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        
        n_horses = len(grp)
        max_horses = 18
        
        # パディング
        X_padded = np.zeros((1, max_horses, model.input_dim), dtype=np.float32)
        mask = np.zeros((1, max_horses), dtype=np.float32)
        
        n = min(n_horses, max_horses)
        X_padded[0, :n, :] = X[:n]
        mask[0, :n] = 1.0
        
        # 予測
        scores = model.predict(X_padded, mask)
        
        # スコアをDataFrameに戻す
        race_scores = scores[0, :n]
        for i, (idx, row) in enumerate(grp[:n].iterrows()):
            all_scores.append({'idx': idx, 'score': race_scores[i]})
    
    # スコアを元のDataFrameに追加
    score_df = pd.DataFrame(all_scores).set_index('idx')
    df['score'] = score_df['score']
    
    return df

def main():
    print("\n" + "="*80)
    print("📊 v14 ROI Model vs v12 Ensemble 比較評価")
    print("="*80)
    
    year = 2025
    
    # データ読み込み
    print("\n1. データ読み込み...")
    data_path = 'experiments/v14_roi/data/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'] == year].copy()
    
    # JRAのみ
    jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df = df[df['venue_code'].isin(jra_codes)].copy()
    
    logger.info(f"Loaded {len(df)} rows for {year} (JRA only)")
    
    # 数値変換
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').fillna(1).astype(int)
    
    # 払戻データ
    print("2. 払戻データ読み込み...")
    pay_df = load_payouts(year)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Loaded payouts for {len(payout_map)} races")
    
    # 特徴量カラム取得
    with open('experiments/v14_roi/data/lgbm_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    results = []
    
    # v12評価
    print("\n3. v12 Ensemble評価...")
    try:
        df_v12 = df.copy()
        df_v12 = predict_with_v12(df_v12, feature_cols)
        v12_result = evaluate_model_predictions(df_v12, 'v12 Ensemble')
        results.append(v12_result)
        logger.info(f"v12: ROI {v12_result['roi']:.1f}%, Acc {v12_result['accuracy']:.1f}%")
    except Exception as e:
        logger.error(f"v12 evaluation failed: {e}")
    
    # v14評価
    print("\n4. v14 ROI Model評価...")
    try:
        df_v14 = df.copy()
        df_v14 = predict_with_v14(df_v14, feature_cols)
        v14_result = evaluate_model_predictions(df_v14, 'v14 ROI')
        results.append(v14_result)
        logger.info(f"v14: ROI {v14_result['roi']:.1f}%, Acc {v14_result['accuracy']:.1f}%")
    except Exception as e:
        logger.error(f"v14 evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 結果表示
    print("\n" + "="*80)
    print("📊 評価結果比較")
    print("="*80)
    print(f"{'モデル':<20} | {'ROI':>8} | {'的中率':>8} | {'レース数':>8} | {'利益':>12}")
    print("-"*70)
    
    for r in results:
        profit = r['return'] - r['cost']
        print(f"{r['model']:<20} | {r['roi']:>7.1f}% | {r['accuracy']:>7.1f}% | {r['races']:>8} | {profit:>+12,.0f}円")
    
    # 勝者
    if len(results) >= 2:
        best = max(results, key=lambda x: x['roi'])
        print(f"\n🏆 勝者: {best['model']} (ROI {best['roi']:.1f}%)")
    
    # ファイル保存
    os.makedirs('reports', exist_ok=True)
    with open('reports/v14_vs_v12_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("=== v14 ROI Model vs v12 Ensemble 比較 ===\n\n")
        for r in results:
            profit = r['return'] - r['cost']
            f.write(f"{r['model']}: ROI {r['roi']:.1f}%, Acc {r['accuracy']:.1f}%, {r['races']}レース, 利益 {profit:+,.0f}円\n")
    
    print("\n結果を reports/v14_vs_v12_comparison.txt に保存しました")
    print("\n✅ 比較評価完了!")

if __name__ == "__main__":
    main()
