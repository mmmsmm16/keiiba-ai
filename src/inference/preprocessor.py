import pandas as pd
import numpy as np
import logging
import os
import sys

# src pass
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.aggregators import HistoryAggregator
from preprocessing.category_aggregators import CategoryAggregator
from preprocessing.advanced_features import AdvancedFeatureEngineer
from preprocessing.cleansing import DataCleanser

logger = logging.getLogger(__name__)

class InferencePreprocessor:
    """
    推論用：データの前処理を行うクラス。
    過去データと結合して特徴量を再生成し、学習時と同じフォーマットの入力を作成します。
    """
    def __init__(self, history_path: str = None):
        if history_path is None:
            # デフォルトは src/../../data/processed/preprocessed_data.parquet
            base_dir = os.path.dirname(__file__)
            history_path = os.path.join(base_dir, '../../data/processed/preprocessed_data.parquet')
        
        self.history_path = history_path

    def preprocess(self, new_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        推論用データを前処理します。

        Args:
            new_df (pd.DataFrame): InferenceDataLoaderから取得した未処理の推論対象データ。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                1. 推論用特徴量データ (X) - モデル入力用
                2. 推論対象データのメタデータ (IDs) - 結果表示用 (race_id, horse_number, horse_name, etc.)
        """
        logger.info("推論用前処理を開始します...")

        # 1. 過去データのロード
        if not os.path.exists(self.history_path):
             raise FileNotFoundError(f"過去データが見つかりません: {self.history_path}")
        
        history_df = pd.read_parquet(self.history_path)
        logger.info(f"過去データをロードしました: {len(history_df)} 件")

        # 2. データの結合
        # new_df には rank などのカラムがない(NaN)ため、単純結合でOK
        # 共通のカラムのみを持つように調整するか、concatに任せる
        # history_df は既に特徴量を持っているが、再計算するため問題ない。
        # ただし、raw columns (rank, horse_id, etc.) が history_df に残っている必要がある。
        
        # 新しいデータの簡易クレンジング (DataCleanserはrank=NaNを消すので使わない)
        new_df['date'] = pd.to_datetime(new_df['date'])
        
        # 結合
        # reset_indexしないとindexが重複する可能性あり
        combined_df = pd.concat([history_df, new_df], axis=0, ignore_index=True)
        logger.info(f"データを結合しました: 合計 {len(combined_df)} 件 (新規 {len(new_df)} 件)")

        # 3. 特徴量生成パイプラインの実行
        # 既存の特徴量カラムがあっても上書きされる想定
        
        # 3.1 基本特徴量
        feat_engineer = FeatureEngineer()
        combined_df = feat_engineer.add_features(combined_df)
        
        # 3.2 過去走特徴量 (HistoryAggregator)
        # 内部で sort_values(['horse_id', 'date']) される
        hist_agg = HistoryAggregator()
        combined_df = hist_agg.aggregate(combined_df)
        
        # 3.3 カテゴリ集計 (CategoryAggregator)
        cat_agg = CategoryAggregator()
        combined_df = cat_agg.aggregate(combined_df)
        
        # 3.4 高度特徴量 (AdvancedFeatureEngineer)
        adv_engineer = AdvancedFeatureEngineer()
        combined_df = adv_engineer.add_features(combined_df)

        # 4. 推論対象行の抽出
        # 今回追加した new_df に対応する行のみを取り出す
        # race_id と horse_number で特定する
        # (結合前に new_df にフラグを立てておくと楽だが、race_idリストでフィルタリングする)
        
        target_race_ids = new_df['race_id'].unique()
        inference_df = combined_df[combined_df['race_id'].isin(target_race_ids)].copy()
        
        # new_df に含まれていたものだけに絞る (historyにも同じrace_idがある可能性は低いが念のため)
        # dateもチェックしたほうが安全？ -> 今回はrace_idがユニークと仮定
        
        # 順序を戻す (race_id, horse_number)
        inference_df = inference_df.sort_values(['race_id', 'horse_number'])
        
        logger.info(f"特徴量生成完了。推論対象: {len(inference_df)} 件")

        # 5. カラム選択 (学習時と同じ入力形式にする)
        # DatasetSplitter._create_lgbm_dataset のロジックを参照
        
        drop_cols = [
            # ID・メタデータ
            'race_id', 'date', 'title', 'horse_id', 'horse_name',
            'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
            # 目的変数 (存在すれば)
            'rank', 'target', 'rank_str',
            # 未来情報 
            'time', 'raw_time',
            'passing_rank',
            'last_3f',
            'odds', 'popularity',
            'weight', 'weight_diff', # weightは補完していないので削除対象（学習時も削除している）
            'weight_diff_val', 'weight_diff_sign',
            'winning_numbers', 'payout', 'ticket_type',
            # PC-KEIBA specific
            'pass_1', 'pass_2', 'pass_3', 'pass_4',
            # Temporary
            'is_nige_temp'
        ]

        # ID情報は返却用に確保
        id_cols = ['race_id', 'date', 'venue', 'race_number', 'horse_number', 'horse_name', 'jockey_id']
        ids = inference_df[id_cols].copy()

        # X の作成
        X = inference_df.drop(columns=drop_cols, errors='ignore')
        
        # カテゴリ変数の除外 (数値のみ)
        X = X.select_dtypes(exclude=['object'])

        # 欠損値処理 (LightGBM/CatBoostはOKだが、TabNetやStandardScalerのためにFillNa推奨)
        # ここでは0埋め等はせず、モデル側(TabNet)の仕様に任せるか、最小限の処理を行う
        # DatasetSplitterではfillnaしていないが、TabNetクラス内で np.nan_to_num している。
        # なのでこのまま返す。

        return X, ids
