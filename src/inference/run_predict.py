import sys
import os
import argparse
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.inference_loader import SokuhoDataLoader
from preprocessing.cleansing import DataCleanser
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.aggregators import HistoryAggregator
from preprocessing.category_aggregators import CategoryAggregator
from preprocessing.advanced_features import AdvancedFeatureEngineer
from model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="リアルタイム予測を実行します")
    parser.add_argument('--date', type=str, help='予測対象日 (YYYYMMDD)', default=None)
    args = parser.parse_args()

    target_date = args.date
    if not target_date:
        target_date = datetime.now().strftime('%Y%m%d')

    logger.info(f"予測対象日: {target_date}")

    # 1. 過去データのロード (特徴量生成用)
    # 直近1年分程度あればラグ特徴量の生成には十分
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    if not os.path.exists(data_path):
        logger.error("過去データが見つかりません。先に学習用前処理を実行してください。")
        return

    logger.info("過去データをロード中...")
    # pandasのread_parquetではfilters引数で読み込み期間を絞れる (pyarrowが必要)
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    try:
        df_history = pd.read_parquet(data_path, filters=[('date', '>=', one_year_ago)])
    except Exception as e:
        logger.warning(f"フィルタ付きロードに失敗しました (pyarrowバージョン依存の可能性): {e}")
        logger.warning("全件ロードします。")
        df_history = pd.read_parquet(data_path)

    # 2. 速報データのロード
    loader = SokuhoDataLoader()
    df_inference = loader.load(target_date=target_date)

    if df_inference.empty:
        logger.warning(f"指定日 ({target_date}) の速報データがありません。PC-KEIBAで「速報データの登録」が行われているか確認してください。")
        return

    logger.info(f"速報データ: {len(df_inference)} 件")

    # 3. データの結合
    # 歴史データと推論データを結合して、時系列順に並べる
    df_combined = pd.concat([df_history, df_inference], ignore_index=True)

    # 4. 前処理パイプラインの実行
    # DataCleanser (Inference Mode: rank欠損を許容)
    cleanser = DataCleanser()
    df_combined = cleanser.cleanse(df_combined, is_inference=True)

    # Feature Engineering
    fe = FeatureEngineer()
    df_combined = fe.add_features(df_combined)

    # Aggregators (History, Category, Advanced)
    # これらは shift(1) を使うため、未来行（推論対象）に対して過去データが集計される
    h_agg = HistoryAggregator()
    df_combined = h_agg.aggregate(df_combined)

    c_agg = CategoryAggregator()
    df_combined = c_agg.aggregate(df_combined)

    adv_fe = AdvancedFeatureEngineer()
    df_combined = adv_fe.add_features(df_combined)

    # 5. 推論対象データの抽出
    # dateがtarget_dateのデータを抽出
    target_dt = pd.to_datetime(target_date)
    df_target = df_combined[df_combined['date'] == target_dt].copy()

    if df_target.empty:
        logger.error(f"前処理後の推論対象データが空です。日付 {target_date} のデータが正しく処理されませんでした。")
        return

    # 6. 予測
    # モデルロード
    model_path = os.path.join(os.path.dirname(__file__), '../../models/ensemble_model.pkl')
    if not os.path.exists(model_path):
        logger.error("モデルファイルがありません。")
        return

    model = EnsembleModel()
    model.load_model(model_path)

    # 特徴量リストの取得 (学習時と同じカラムが必要)
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    feature_cols = datasets['train']['X'].columns.tolist()

    # 欠損している特徴量があれば埋める
    for col in feature_cols:
        if col not in df_target.columns:
            df_target[col] = 0

    X_pred = df_target[feature_cols]

    logger.info("予測を実行中...")
    scores = model.predict(X_pred)
    df_target['score'] = scores

    # 7. 結果表示
    logger.info("--- 予測結果 ---")

    disp_cols = ['race_number', 'horse_number', 'horse_name', 'score', 'jockey_id', 'nige_rate']

    for race_id, group in df_target.groupby('race_id'):
        venue_name = group['venue'].iloc[0] # コードのままかもしれないが
        race_num = group['race_number'].iloc[0]

        logger.info(f"Race ID: {race_id} ({venue_name} {race_num}R)")

        # スコア降順ソート
        group = group.sort_values('score', ascending=False)

        # コンソール出力
        print(group[disp_cols].to_string(index=False))
        print("-" * 50)

if __name__ == "__main__":
    main()
