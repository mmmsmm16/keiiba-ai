import sys
import os
import pickle
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from scipy.special import softmax

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    # 1. データのロード (Parquetから元データを取得)
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    if not os.path.exists(data_path):
        logger.error(f"データファイルがありません: {data_path}")
        return

    # テストデータ(2024年)のみロード
    # 全データを読むと重いので、本来はフィルタリングして読むか分割保存すべきだが、
    # ここでは既存コードに合わせて全読み込み->フィルタ
    df = pd.read_parquet(data_path)
    test_df = df[df['year'] == 2024].copy()

    if test_df.empty:
        logger.error("テストデータ(2024年)がありません。")
        return

    # 2. モデルのロード
    model_path = os.path.join(os.path.dirname(__file__), '../../models/ensemble_model.pkl')
    if not os.path.exists(model_path):
        logger.error(f"モデルファイルがありません: {model_path}")
        return

    model = EnsembleModel()
    model.load_model(model_path)

    # 3. 特徴量の整合性確認と予測
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    if datasets['train']['X'] is None:
        logger.error("学習データの特徴量情報がありません。")
        return

    feature_cols = datasets['train']['X'].columns.tolist()
    
    # テストデータにカラムが存在するか確認
    missing_cols = set(feature_cols) - set(test_df.columns)
    if missing_cols:
        logger.warning(f"不足しているカラムがあります（0で埋めます）: {missing_cols}")
        for c in missing_cols:
            test_df[c] = 0

    X_test = test_df[feature_cols]

    logger.info("予測を実行中...")
    scores = model.predict(X_test)
    test_df['score'] = scores

    # 確率と期待値の計算
    logger.info("勝率と期待値を計算中...")
    # レースごとにSoftmax
    test_df['prob'] = test_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    # 期待値 = 確率 * オッズ (欠損は0)
    test_df['expected_value'] = test_df['prob'] * test_df['odds'].fillna(0)

    # 4. シミュレーションと保存
    output_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
    os.makedirs(output_dir, exist_ok=True)
    
    simulation_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategies': {},
        'roi_curve': []
    }

    # Strategy 1: レース内で最も期待値が高い馬を1点買い
    logger.info("--- Simulation: Max Expected Value (1点買い) ---")
    sim_ev = simulate_single_choice(test_df, 'expected_value')
    simulation_results['strategies']['max_ev'] = sim_ev
    logger.info(f"Max EV - ROI: {sim_ev['roi']:.2f}%, Hit: {sim_ev['accuracy']:.2%}")

    # Strategy 2: レース内で最もスコアが高い馬を1点買い
    logger.info("--- Simulation: Max Score (1点買い) ---")
    sim_score = simulate_single_choice(test_df, 'score')
    simulation_results['strategies']['max_score'] = sim_score
    logger.info(f"Max Score - ROI: {sim_score['roi']:.2f}%, Hit: {sim_score['accuracy']:.2%}")

    # Strategy 3: 期待値閾値ごとのROIカーブ
    # 期待値が X 以上の馬を全て買う（単勝）
    logger.info("--- Simulation: EV Thresholds (ROI Curve) ---")
    curve_data = simulate_threshold_curve(test_df)
    simulation_results['roi_curve'] = curve_data

    # Save to JSON
    json_path = os.path.join(output_dir, 'latest_simulation.json')
    with open(json_path, 'w') as f:
        json.dump(simulation_results, f, indent=4, cls=NpEncoder)
        
    logger.info(f"シミュレーション結果を保存しました: {json_path}")


def simulate_single_choice(df, target_col):
    results = []
    
    for race_id, group in df.groupby('race_id'):
        if group[target_col].isnull().all():
            continue

        best_idx = group[target_col].idxmax()
        best_horse = group.loc[best_idx]
        
        actual_rank = best_horse['rank']
        odds = best_horse['odds']
        
        bet = 100
        return_amount = odds * 100 if actual_rank == 1 else 0
        
        results.append({'bet': bet, 'return': return_amount, 'hit': 1 if actual_rank == 1 else 0})
        
    sim_df = pd.DataFrame(results)
    if sim_df.empty:
        return {'roi': 0, 'accuracy': 0, 'total_bet': 0, 'total_return': 0}
        
    total_bet = sim_df['bet'].sum()
    total_return = sim_df['return'].sum()
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    accuracy = sim_df['hit'].mean()
    
    return {'roi': roi, 'accuracy': accuracy, 'total_bet': total_bet, 'total_return': total_return}

def simulate_threshold_curve(df):
    target_col = 'expected_value'
    # 閾値を0.5刻みなどで設定
    # 期待値は prob * odds。prob約0.1 * odds10 = 1.0 (等倍)。
    # 0.5 (回収率50%期待) から 2.0 (回収率200%期待) くらいまでスキャン
    thresholds = np.arange(0.5, 3.0, 0.1)
    
    curve_data = []
    
    for th in thresholds:
        # 閾値を超える条件の行を抽出
        bets = df[df[target_col] >= th].copy()
        
        if bets.empty:
            curve_data.append({
                'threshold': th,
                'roi': 0,
                'bet_count': 0,
                'accuracy': 0
            })
            continue
            
        bets['bet_amount'] = 100
        bets['return_amount'] = bets.apply(lambda row: row['odds'] * 100 if row['rank'] == 1 else 0, axis=1)
        
        total_bet = bets['bet_amount'].sum()
        total_return = bets['return_amount'].sum()
        roi = total_return / total_bet * 100 if total_bet > 0 else 0
        bet_count = len(bets)
        accuracy = (bets['rank'] == 1).mean()
        
        curve_data.append({
            'threshold': round(th, 2),
            'roi': roi,
            'bet_count': bet_count,
            'accuracy': accuracy
        })
        
    return curve_data

if __name__ == "__main__":
    main()
