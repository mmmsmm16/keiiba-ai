import sys
import os
import logging
import pandas as pd

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.loader import RawDataLoader
from preprocessing.cleansing import DataCleanser
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.aggregators import HistoryAggregator
from preprocessing.category_aggregators import CategoryAggregator
from preprocessing.dataset import DatasetSplitter

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. データのロード
        logger.info("Step 1: データのロード")
        loader = RawDataLoader()
        df = loader.load() # 全データをロード

        # 2. データクレンジング
        logger.info("Step 2: データクレンジング")
        cleanser = DataCleanser()
        df = cleanser.cleanse(df)

        # 3. 特徴量生成
        logger.info("Step 3: 基本特徴量生成")
        engineer = FeatureEngineer()
        df = engineer.add_features(df)

        # 4. 過去走特徴量生成
        logger.info("Step 4: 過去走特徴量生成")
        aggregator = HistoryAggregator()
        df = aggregator.aggregate(df)

        # 5. カテゴリ集計特徴量生成
        logger.info("Step 5: カテゴリ集計特徴量生成")
        cat_aggregator = CategoryAggregator()
        df = cat_aggregator.aggregate(df)

        # 6. データの保存 (全データ)
        output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'preprocessed_data.parquet')

        logger.info(f"Step 6: 中間データの保存 ({output_path})")
        df.to_parquet(output_path, index=False)

        # 7. データセット分割 (Train/Valid/Test)
        logger.info("Step 7: データセット分割と作成")
        splitter = DatasetSplitter()
        datasets = splitter.split_and_create_dataset(df)

        # 8. データセットの保存 (Pickle)
        # 辞書形式 (X, y, group) を保存
        dataset_path = os.path.join(output_dir, 'lgbm_datasets.pkl')
        logger.info(f"Step 8: 学習用データセットの保存 ({dataset_path})")
        pd.to_pickle(datasets, dataset_path)

        logger.info("前処理パイプラインが正常に完了しました。")

    except Exception as e:
        logger.error(f"前処理パイプライン中にエラーが発生しました: {e}")
        # DB接続エラーなどで失敗しても、動作確認用にモックデータでテストするオプションがあれば良いが、
        # ここでは本番用スクリプトとしてエラーで終了させる。

if __name__ == "__main__":
    main()
