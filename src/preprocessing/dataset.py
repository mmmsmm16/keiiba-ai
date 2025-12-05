import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DatasetSplitter:
    """
    データセットを学習用・検証用・テスト用に分割し、
    LightGBM (Ranking) で学習可能な形式に整形するクラス。
    """
    def split_and_create_dataset(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        データを分割してデータセットを作成します。

        Args:
            df (pd.DataFrame): 前処理済みの全データ。

        Returns:
            Dict: train, valid, test それぞれの {'X', 'y', 'group'} を含む辞書。
        """
        logger.info("データセットの分割と作成を開始...")

        # ターゲット変数の作成 (Relevance Score)
        # 1着=3, 2着=2, 3着=1, 着外=0
        df['target'] = df['rank'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else (1 if x == 3 else 0)))

        # 時系列分割
        # Train: 2015-2022
        # Valid: 2023
        # Test: 2024
        # (日付型であることを前提。feature_engineeringでyearを作成済み)
        train_df = df[df['year'].between(2015, 2022)].copy()
        valid_df = df[df['year'] == 2023].copy()
        test_df = df[df['year'] == 2024].copy()

        logger.info(f"分割完了: Train({len(train_df)}), Valid({len(valid_df)}), Test({len(test_df)})")

        return {
            'train': self._create_lgbm_dataset(train_df),
            'valid': self._create_lgbm_dataset(valid_df),
            'test': self._create_lgbm_dataset(test_df)
        }

    def _create_lgbm_dataset(self, df: pd.DataFrame) -> Dict:
        """
        DataFrameからLightGBM用の X, y, group を作成します。
        """
        if df.empty:
            return {'X': pd.DataFrame(), 'y': pd.Series(), 'group': np.array([])}

        # LambdaRankのためには、クエリ（レースID）ごとにデータがまとまっている必要がある
        # race_id でソート
        df = df.sort_values('race_id')

        # グループ情報 (各レースの出走馬数)
        group = df.groupby('race_id').size().to_numpy()

        # 特徴量 (X) と ターゲット (y) の分離
        # AGENTS.md の規定により、未来情報（レース結果のオッズ、体重など）は削除する
        # id系や日付、レース名なども削除
        drop_cols = [
            # ID・メタデータ
            'race_id', 'date', 'title', 'horse_id', 'horse_name',
            'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
            # 目的変数そのもの
            'rank', 'target',
            # 未来情報 (Result)
            'time', 'passing_rank', 'last_3f',
            'odds', 'popularity', # 確定オッズ・人気は禁止
            'weight', 'weight_diff', # 当日馬体重も禁止 (AGENTS.md準拠)
            'winning_numbers', 'payout', 'ticket_type' # 払い戻し情報
        ]

        # 存在しないカラムをdropしようとしてもエラーにならないように errors='ignore'
        X = df.drop(columns=drop_cols, errors='ignore')

        # カテゴリ変数がobject型のままだとLightGBMで扱いにくい場合があるが、
        # 基本的にはfeature_engineeringで数値化済み (sex_num, weather_num...)
        # 残っているobject型があれば除外するかcategory型にする
        X = X.select_dtypes(exclude=['object'])

        y = df['target']

        return {'X': X, 'y': y, 'group': group}
