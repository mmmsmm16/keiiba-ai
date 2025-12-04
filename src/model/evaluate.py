import sys
import os
import pickle
import pandas as pd
import numpy as np
import logging

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
    # 学習時と同じ特徴量を選択する
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    if datasets['train']['X'] is None:
        logger.error("学習データの特徴量情報がありません。")
        return

    feature_cols = datasets['train']['X'].columns.tolist()

    # テストデータから特徴量を抽出
    # 欠損カラムがある場合はエラーになるので注意 (通常は同じパイプラインを通るので一致する)
    X_test = test_df[feature_cols]

    logger.info("予測を実行中...")
    scores = model.predict(X_test)
    test_df['score'] = scores

    # 4. シミュレーション (単勝)
    logger.info("回収率シミュレーション (予測スコア1位の単勝を購入)...")

    results = []

    # レースごとにグループ化
    for race_id, group in test_df.groupby('race_id'):
        # スコア最大の馬を選択
        best_horse_idx = group['score'].idxmax()
        best_horse = group.loc[best_horse_idx]

        # 実際の結果
        actual_rank = best_horse['rank']
        odds = best_horse['odds']

        # 賭け金
        bet = 100
        return_amount = 0

        # 1着なら払い戻し (オッズ × 100)
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
