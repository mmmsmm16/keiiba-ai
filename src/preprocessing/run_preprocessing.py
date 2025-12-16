"""
run_preprocessing.py - 前処理パイプライン実行スクリプト

[変更履歴]
- 2025-12-15 v11 Extended: 
  - Phase 0: --history_start_date, --start_date 追加（データ期間制約）
  - A4対応: --use_realtime_features フラグ
  - A6対応: --no_embedding_features フラグ
  - データ品質バリデーション追加
"""

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
from preprocessing.experience_features import ExperienceFeatureEngineer
from preprocessing.race_level_features import RaceLevelFeatureEngineer
from preprocessing.validation_utils import validate_data_quality

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse

def main():
    parser = argparse.ArgumentParser(description='競馬AI前処理パイプライン (v11 Extended)')
    parser.add_argument('--resume', action='store_true', help='前回のStep 5完了時点から再開します')
    parser.add_argument('--jra_only', action='store_true', help='JRA会場のみにフィルタリングします')
    parser.add_argument('--suffix', type=str, default='', help='出力ファイル名のサフィックス (例: _v11)')
    
    # [v11 Extended Phase 0] データ期間制約
    parser.add_argument('--history_start_date', type=str, default='2014-01-01',
                       help='生データ読み込み開始日（ウォームアップ用）')
    parser.add_argument('--start_date', type=str, default='2014-01-01',
                       help='学習対象の開始日（これより前の行は除外）')
    
    # [v11 Extended] 特徴量フラグ（統一名）
    parser.add_argument('--use_realtime_features', action='store_true', 
                       help='trend_*特徴量を使用（逐次更新モード）')
    parser.add_argument('--no_embedding_features', action='store_false',
                       dest='use_embedding_features', default=True,
                       help='embedding特徴量を無効化')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("競馬AI 前処理パイプライン v11 Extended")
    logger.info(f"  history_start_date: {args.history_start_date}")
    logger.info(f"  start_date: {args.start_date}")
    logger.info(f"  use_realtime_features: {args.use_realtime_features}")
    logger.info(f"  use_embedding_features: {args.use_embedding_features}")
    logger.info("=" * 60)

    try:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '../../data/interim')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # チェックポイントや出力ファイルのパスをモードによって切り替え
        # suffix優先、なければjra_onlyチェック
        file_suffix = args.suffix
        if not file_suffix and args.jra_only:
            file_suffix = "_jra_v5"
            
        checkpoint_name = f'step5_checkpoint{file_suffix}.parquet'
        output_parquet_name = f'preprocessed_data{file_suffix}.parquet'
        output_dataset_name = f'lgbm_datasets{file_suffix}.pkl'
        
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        df = None

        if args.resume and os.path.exists(checkpoint_path):
            logger.info(f"チェックポイントからデータをロードします: {checkpoint_path}")
            df = pd.read_parquet(checkpoint_path)
            logger.info("Step 1-5: スキップ (完了済み)")
        else:
            # 1. データのロード
            logger.info(f"Step 1: データのロード (JRA-VAN)")
            logger.info(f"  history_start_date={args.history_start_date}, jra_only={args.jra_only}")
            loader = JraVanDataLoader()
            df = loader.load(
                jra_only=args.jra_only,
                history_start_date=args.history_start_date
            )
 
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
        # [v11 Extended] use_realtime_featuresフラグで運用モード切替
        from preprocessing.realtime_features import RealTimeFeatureEngineer
        mode_str = "逐次更新モード" if args.use_realtime_features else "事前予測モード"
        logger.info(f"Step 6.7: リアルタイム特徴量生成 [{mode_str}]")
        realtime_engineer = RealTimeFeatureEngineer()
        df = realtime_engineer.add_features(df, use_realtime=args.use_realtime_features)

        # 6.8. Deep Learning Embedding Features (Phase 12)
        # [v11 Extended] --no_embedding_featuresフラグでスキップ可能
        if args.use_embedding_features:
            from preprocessing.embedding_features import EmbeddingFeatureEngineer
            logger.info("Step 6.8: Embedding特徴量生成 (Entity Embeddings)")
            emb_engineer = EmbeddingFeatureEngineer()
            df = emb_engineer.add_features(df)
        else:
            logger.info("Step 6.8: Embedding特徴量生成 [スキップ: --no_embedding_features指定]")

        # 6.9. 経験特徴量生成 (v7新規 - コース経験・距離経験・騎手乗り替わり)
        logger.info("Step 6.9: 経験特徴量生成 (コース経験・距離経験・初条件フラグ)")
        exp_engineer = ExperienceFeatureEngineer()
        df = exp_engineer.add_features(df)

        # 6.10. レースレベル特徴量生成 (v7新規 - 前走レースの強さ評価)
        logger.info("Step 6.10: レースレベル特徴量生成 (メンバー強度・パフォーマンス価値)")
        race_level_engineer = RaceLevelFeatureEngineer()
        df = race_level_engineer.add_features(df)

        # 6.11. [v11 Extended N3] 対戦レベル特徴量生成
        from preprocessing.opposition_features import OppositionFeatureEngineer
        logger.info("Step 6.11: 対戦レベル特徴量生成 (opponent_strength, relative_strength)")
        opp_engineer = OppositionFeatureEngineer()
        df = opp_engineer.add_features(df)

        # ================================================================
        # [v11 Extended Phase 0] ウォームアップ期間の除外
        # ================================================================
        # history_start_date < start_date の場合、ウォームアップ期間のデータを除外
        if args.start_date:
            start_date_pd = pd.to_datetime(args.start_date)
            before_count = len(df)
            df = df[df['date'] >= start_date_pd].copy()
            after_count = len(df)
            warmup_excluded = before_count - after_count
            
            if warmup_excluded > 0:
                logger.info(f"[Phase 0] ウォームアップ期間除外: {warmup_excluded:,}件 (学習対象: {after_count:,}件)")
            
            logger.info(f"学習対象期間: {df['date'].min()} ~ {df['date'].max()}")
            logger.info(f"レース数: {df['race_id'].nunique():,}件, レコード数: {len(df):,}件")

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

        # [v11] データ品質バリデーション
        logger.info("Step 10: データ品質バリデーション")
        validation_result = validate_data_quality(df, stage="Final")
        if validation_result['warnings']:
            logger.warning(f"品質チェックで {len(validation_result['warnings'])} 件の警告があります")

        # [v11 Extended V6] リーク検査
        from preprocessing.validation_utils import check_feature_leak
        logger.info("Step 10b: リーク検査 (FORBIDDEN_COLUMNS)")
        feature_cols = list(datasets['train']['X'].columns)
        check_feature_leak(feature_cols, raise_error=False)  # 警告のみ（エラーにしない）

        logger.info("=" * 60)
        logger.info("前処理パイプラインが正常に完了しました。")
        logger.info(f"  出力ファイル: {output_path}")
        logger.info(f"  データセット: {dataset_path}")
        logger.info(f"  レコード数: {len(df)}")
        logger.info(f"  特徴量数: {len(datasets['train']['X'].columns)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"前処理パイプライン中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
