import sys
import os
import pickle
import pandas as pd
import numpy as np
import logging
from scipy.special import softmax

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. データのロード (Parquetから元データを取得)
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    if not os.path.exists(data_path):
        logger.error("データファイルがありません")
        return

    df = pd.read_parquet(data_path)
    test_df = df[df['year'] == 2024].copy()

    if test_df.empty:
        logger.error("テストデータ(2024年)がありません。")
        return

    # 2. モデルのロード
    model_path = os.path.join(os.path.dirname(__file__), '../../models/lgbm_model.pkl')
    if not os.path.exists(model_path):
        logger.error("モデルファイルがありません")
        return

    model = KeibaLGBM()
    model.load_model(model_path)

    # 3. 予測
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    if datasets['train']['X'] is None:
        logger.error("学習データの特徴量情報がありません。")
        return

    feature_cols = datasets['train']['X'].columns.tolist()
    X_test = test_df[feature_cols]

    logger.info("予測を実行中...")
    scores = model.predict(X_test)
    test_df['score'] = scores

    # 確率と期待値の計算
    logger.info("勝率と期待値を計算中...")
    # レースごとにSoftmaxをかけて確率(勝率の近似)に変換
    test_df['prob'] = test_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    # 期待値 = 確率 * オッズ (払い戻し期待値)
    # oddsが欠損している場合は期待値0とする
    test_df['expected_value'] = test_df['prob'] * test_df['odds'].fillna(0)

    # 4. シミュレーション
    strategies = {
        '最大スコア (勝率重視)': 'score',
        '最大期待値 (回収率重視)': 'expected_value'
    }

    for name, target_col in strategies.items():
        logger.info(f"--- シミュレーション: {name} ---")
        run_simulation(test_df, target_col)

def run_simulation(df, target_col):
    results = []

    for race_id, group in df.groupby('race_id'):
        # ターゲット値が最大の馬を選択
        if group[target_col].isnull().all():
            continue # データがない場合はスキップ

        best_horse_idx = group[target_col].idxmax()
        best_horse = group.loc[best_horse_idx]

        actual_rank = best_horse['rank']
        odds = best_horse['odds']

        bet = 100
        return_amount = 0

        # 1着なら払い戻し
        if actual_rank == 1:
            return_amount = odds * 100

        results.append({
            'race_id': race_id,
            'bet': bet,
            'return': return_amount,
            'hit': 1 if actual_rank == 1 else 0
        })

    sim_df = pd.DataFrame(results)

    if sim_df.empty:
        logger.warning("シミュレーション対象のレースがありませんでした。")
        return

    total_bet = sim_df['bet'].sum()
    total_return = sim_df['return'].sum()
    accuracy = sim_df['hit'].mean()
    roi = total_return / total_bet * 100 if total_bet > 0 else 0

    logger.info(f"対象レース数: {len(sim_df)}")
    logger.info(f"的中率: {accuracy:.2%}")
    logger.info(f"総投資額: {total_bet}円")
    logger.info(f"総回収額: {total_return:.0f}円")
    logger.info(f"回収率: {roi:.2f}%")

if __name__ == "__main__":
    main()
