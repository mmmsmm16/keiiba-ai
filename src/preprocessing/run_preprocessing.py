import sys
import os
import logging
import pandas as pd

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.cleansing import DataCleanser
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.aggregators import HistoryAggregator
from preprocessing.category_aggregators import CategoryAggregator
from preprocessing.dataset import DatasetSplitter
from preprocessing.advanced_features import AdvancedFeatureEngineer
from preprocessing.disadvantage_detector import DisadvantageDetector
from preprocessing.relative_features import RelativeFeatureEngineer

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='前回のStep 5完了時点から再開します')
    parser.add_argument('--jra_only', action='store_true', help='JRA会場のみにフィルタリングします (Model v5用)')
    args = parser.parse_args()

    try:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '../../data/interim')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # チェックポイントや出力ファイルのパスをモードによって切り替え
        suffix = "_jra_v5" if args.jra_only else ""
        checkpoint_name = f'step5_checkpoint{suffix}.parquet'
        output_parquet_name = f'preprocessed_data{suffix}.parquet'
        output_dataset_name = f'lgbm_datasets{suffix}.pkl'
        
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        df = None

        if args.resume and os.path.exists(checkpoint_path):
            logger.info(f"チェックポイントからデータをロードします: {checkpoint_path}")
            df = pd.read_parquet(checkpoint_path)
            logger.info("Step 1-5: スキップ (完了済み)")
        else:
            # 1. データのロード
            logger.info(f"Step 1: データのロード (JRA-VAN) [JRA Only Mode: {args.jra_only}]")
            loader = JraVanDataLoader()
            df = loader.load(jra_only=args.jra_only) # 引数を渡す
 
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
            
            # チェックポイント保存
            logger.info(f"Step 5完了: チェックポイントを保存します ({checkpoint_path})")
            df.to_parquet(checkpoint_path, index=False)

        # 5b. 血統特徴量生成 (Bloodline)
        # NOTE: BloodlineFeatureEngineerは内部でjvd_umをロードします
        from preprocessing.bloodline_features import BloodlineFeatureEngineer
        logger.info("Step 5b: 血統特徴量生成")
        bloodline_engineer = BloodlineFeatureEngineer()
        df = bloodline_engineer.add_features(df)

        # 6. 高度特徴量生成 (展開予測など)
        logger.info("Step 6: 高度特徴量生成")
        adv_engineer = AdvancedFeatureEngineer()
        df = adv_engineer.add_features(df)

        # 6.5. 不利検出特徴量生成 (Phase 11.1新規)
        logger.info("Step 6.5: 不利検出特徴量生成")
        disadv_detector = DisadvantageDetector()
        df = disadv_detector.add_features(df)

        # 6.6. 相対的特徴量生成 (Phase 11.1新規)
        logger.info("Step 6.6: 相対的特徴量生成")
        relative_engineer = RelativeFeatureEngineer()
        df = relative_engineer.add_features(df)

        # 6.7. リアルタイム特徴量生成 (v9新規 - 当日の傾向)
        from preprocessing.realtime_features import RealTimeFeatureEngineer
        logger.info("Step 6.7: リアルタイム特徴量生成 (当日の傾向)")
        # JRAのみの場合、リアルタイム特徴量は前レースの結果を使うため
        # 同じ会場の過去レース数が減るわけではない（同日同会場なので影響なし）
        realtime_engineer = RealTimeFeatureEngineer()
        df = realtime_engineer.add_features(df)

        # 6.8. Deep Learning Embedding Features (Phase 12)
        from preprocessing.embedding_features import EmbeddingFeatureEngineer
        logger.info("Step 6.8: Embedding特徴量生成 (Entity Embeddings)")
        emb_engineer = EmbeddingFeatureEngineer()
        df = emb_engineer.add_features(df)

        # 7. データの保存 (全データ)
        output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_parquet_name)

        logger.info(f"Step 7: 中間データの保存 ({output_path})")
        df.to_parquet(output_path, index=False)

        # 8. データセット分割 (Train/Valid/Test)
        logger.info("Step 8: データセット分割と作成")
        splitter = DatasetSplitter()
        datasets = splitter.split_and_create_dataset(df)

        # 9. データセットの保存 (Pickle)
        # 辞書形式 (X, y, group) を保存
        dataset_path = os.path.join(output_dir, output_dataset_name)
        logger.info(f"Step 9: 学習用データセットの保存 ({dataset_path})")
        pd.to_pickle(datasets, dataset_path)

        logger.info("前処理パイプラインが正常に完了しました。")

    except Exception as e:
        logger.error(f"前処理パイプライン中にエラーが発生しました: {e}")
        # DB接続エラーなどで失敗しても、動作確認用にモックデータでテストするオプションがあれば良いが、
        # ここでは本番用スクリプトとしてエラーで終了させる。

if __name__ == "__main__":
    main()
