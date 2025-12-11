import os
import shutil
import logging
import pandas as pd
from src.pipeline.config import ExperimentConfig

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.aggregators import HistoryAggregator
from src.preprocessing.category_aggregators import CategoryAggregator
from src.preprocessing.dataset import DatasetSplitter
from src.preprocessing.advanced_features import AdvancedFeatureEngineer
from src.preprocessing.disadvantage_detector import DisadvantageDetector
from src.preprocessing.relative_features import RelativeFeatureEngineer
from src.preprocessing.bloodline_features import BloodlineFeatureEngineer
from src.preprocessing.realtime_features import RealTimeFeatureEngineer
from src.preprocessing.embedding_features import EmbeddingFeatureEngineer

logger = logging.getLogger(__name__)

def prepare_data(config: ExperimentConfig, run_dir: str):
    """
    データセットを作成し、run_dir/data/lgbm_datasets.pkl に保存する。
    use_cache=True の場合、既存のキャッシュを探してコピーする。
    """
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_dataset_path = os.path.join(data_dir, "lgbm_datasets.pkl")

    # キャッシュチェック
    # 優先順位: 1. config.data.cache_path  2. 自動生成パス
    cache_path = None
    cache_base_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    
    if config.data.cache_path:
        # 明示的に指定されたキャッシュパスを使用
        cache_path = config.data.cache_path
        # 相対パスの場合はプロジェクトルートからの相対パスとして解釈
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(os.path.dirname(__file__), '../..', cache_path)
    else:
        # 自動生成: features名とjra_onlyフラグからパス生成
        suffix = "_jra" if config.data.jra_only else ""
        cache_filename = f"lgbm_datasets_{config.data.features}{suffix}.pkl"
        cache_path = os.path.join(cache_base_dir, cache_filename)

    # 評価用データのパス
    eval_data_path = os.path.join(data_dir, "preprocessed_data.parquet")
    
    # キャッシュ使用条件: lgbm_datasets.pkl AND preprocessed_data.parquet の両方が必要
    if config.data.use_cache and cache_path and os.path.exists(cache_path):
        # 評価用データも存在するかチェック
        if not os.path.exists(eval_data_path):
            # キャッシュディレクトリから対応するParquetファイルを探す
            # 命名規則: lgbm_datasets_SUFFIX.pkl -> preprocessed_data_SUFFIX.parquet
            cache_dir = os.path.dirname(cache_path)
            cache_fname = os.path.basename(cache_path)
            if cache_fname.startswith('lgbm_datasets'):
                parquet_fname = cache_fname.replace('lgbm_datasets', 'preprocessed_data').replace('.pkl', '.parquet')
                cache_parquet_path = os.path.join(cache_dir, parquet_fname)
                if os.path.exists(cache_parquet_path):
                    try:
                        shutil.copy(cache_parquet_path, eval_data_path)
                        logger.info(f"💾 Copied cached evaluation data: {cache_parquet_path}")
                    except Exception as e:
                        logger.warning(f"Failed to copy cached parquet: {e}")

        if os.path.exists(eval_data_path):
            logger.info(f"💾 Using cached dataset: {cache_path}")
            try:
                shutil.copy(cache_path, output_dataset_path)
                logger.info(f"✅ Copied to {output_dataset_path}")
                return
            except Exception as e:
                logger.warning(f"Cache copy failed: {e}. Running full preprocessing.")
        else:
            logger.info(f"⚠️ Evaluation data not found ({eval_data_path}). Running Step 6+ to generate it.")

    logger.info("⚙️ Starting Full Preprocessing Pipeline...")
    
    # -----------------------------------------------------------------
    # 増分前処理: base_features (Step 1-5) がキャッシュにあればスキップ
    # -----------------------------------------------------------------
    # jra_only フラグでキャッシュを分ける
    base_suffix = "_jra" if config.data.jra_only else "_all"
    base_features_path = os.path.join(cache_base_dir, f"base_features{base_suffix}.parquet")
    df = None
    
    if config.data.use_cache and os.path.exists(base_features_path):
        logger.info(f"💾 Found base features cache: {base_features_path}")
        try:
            df = pd.read_parquet(base_features_path)
            logger.info(f"✅ Loaded base features: {len(df)} rows, {len(df.columns)} columns")
            logger.info("⏭️ Skipping Steps 1-5 (using cached base features)")
        except Exception as e:
            logger.warning(f"Failed to load base features: {e}. Running full pipeline.")
            df = None
    
    if df is None:
        # 1. Load
        logger.info("Step 1: Loading Data (JRA-VAN)")
        loader = JraVanDataLoader()
        df = loader.load(jra_only=config.data.jra_only)

        # 2. Cleanse
        logger.info("Step 2: Cleansing")
        cleanser = DataCleanser()
        df = cleanser.cleanse(df)

        # 3. Basic Features
        logger.info("Step 3: Basic Feature Engineering")
        engineer = FeatureEngineer()
        df = engineer.add_features(df)

        # 4. History Aggregation
        logger.info("Step 4: History Aggregation")
        aggregator = HistoryAggregator()
        df = aggregator.aggregate(df)

        # 5. Category Aggregation
        logger.info("Step 5: Category Aggregation")
        cat_aggregator = CategoryAggregator()
        df = cat_aggregator.aggregate(df)

        # 5b. Bloodline
        logger.info("Step 5b: Bloodline Features")
        bloodline_engineer = BloodlineFeatureEngineer()
        df = bloodline_engineer.add_features(df)
        
        # 💾 Save base features for future incremental runs
        logger.info(f"💾 Saving base features to: {base_features_path}")
        try:
            os.makedirs(os.path.dirname(base_features_path), exist_ok=True)
            df.to_parquet(base_features_path, index=False)
            logger.info(f"✅ Base features saved: {len(df.columns)} columns")
        except Exception as e:
            logger.warning(f"Failed to save base features: {e}")

    # -----------------------------------------------------------------
    # Step 6+: 高度特徴量（常に実行 - 新しい特徴量を追加可能）
    # -----------------------------------------------------------------
    
    # 6. Advanced Features
    logger.info("Step 6: Advanced Features")
    adv_engineer = AdvancedFeatureEngineer()
    df = adv_engineer.add_features(df)

    # 6.5 Disadvantage
    logger.info("Step 6.5: Disadvantage Detection")
    disadv_detector = DisadvantageDetector()
    df = disadv_detector.add_features(df)

    # 6.6 Relative
    logger.info("Step 6.6: Relative Features")
    relative_engineer = RelativeFeatureEngineer()
    df = relative_engineer.add_features(df)

    # 6.7 Realtime
    logger.info("Step 6.7: Realtime Features")
    realtime_engineer = RealTimeFeatureEngineer()
    df = realtime_engineer.add_features(df)

    # 6.8 Embeddings
    logger.info("Step 6.8: Embedding Features")
    emb_engineer = EmbeddingFeatureEngineer()
    df = emb_engineer.add_features(df)

    # 6.9 Experience Features (v7新規)
    from src.preprocessing.experience_features import ExperienceFeatureEngineer
    logger.info("Step 6.9: Experience Features (v7)")
    exp_engineer = ExperienceFeatureEngineer()
    df = exp_engineer.add_features(df)

    # 6.10 Race Level Features (v7新規)
    from src.preprocessing.race_level_features import RaceLevelFeatureEngineer
    logger.info("Step 6.10: Race Level Features (v7)")
    race_level_engineer = RaceLevelFeatureEngineer()
    df = race_level_engineer.add_features(df)

    # 7. Save Intermediate (Optional - skipping for speed)
    # output_parquet = os.path.join(data_dir, "preprocessed.parquet")
    # df.to_parquet(output_parquet, index=False)

    # 8. Split & Create Dataset
    logger.info(f"Step 8: Splitting Dataset (Valid Year: {config.data.valid_year})")
    
    # Filter by train_years + valid_year + future (test)
    # To strictly follow config.data.train_years, we might want to filter training data specifically.
    # DatasetSplitter uses < valid_year for training.
    # If config.train_years is [2020, ..., 2024] and valid is 2025, then < 2025 covers it.
    # But if config.train_years is [2024] only, DatasetSplitter would still take everything < 2025.
    # We should apply filtering here if we want strict adherence.
    
    # Strict filtering for training data
    valid_year = config.data.valid_year
    train_years = set(config.data.train_years)
    
    # Keep Test data (> valid_year) and Valid data (== valid_year)
    # Filter Train data (< valid_year) to only include train_years
    
    mask_test = df['year'] > valid_year
    mask_valid = df['year'] == valid_year
    mask_train = (df['year'] < valid_year) & (df['year'].isin(train_years))
    
    df_filtered = df[mask_test | mask_valid | mask_train].copy()
    logger.info(f"Filtered rows: {len(df)} -> {len(df_filtered)}")

    splitter = DatasetSplitter()
    target_type = getattr(config.data, 'target_type', 'ranking')
    datasets = splitter.split_and_create_dataset(df_filtered, valid_year=valid_year, target_type=target_type)

    # 8.5. 評価用のrawデータを保存（odds, rank等を含む）
    eval_data_path = os.path.join(data_dir, "preprocessed_data.parquet")
    logger.info(f"Step 8.5: Saving evaluation data to {eval_data_path}")
    df_filtered.to_parquet(eval_data_path, index=False)

    # 9. Save
    logger.info(f"Step 9: Saving Dataset to {output_dataset_path}")
    pd.to_pickle(datasets, output_dataset_path)
    
    # 10. キャッシュにも保存（次回以降の再利用のため）
    if cache_path and not os.path.exists(cache_path):
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            shutil.copy(output_dataset_path, cache_path)
            logger.info(f"💾 Cached dataset saved to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
