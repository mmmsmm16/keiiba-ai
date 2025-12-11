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
    
    @staticmethod
    def _create_ranking_target(rank: int) -> int:
        """v12互換: ランキング用ターゲット (1着=3, 2着=2, 3着=1, 着外=0)"""
        if rank == 1:
            return 3
        elif rank == 2:
            return 2
        elif rank == 3:
            return 1
        else:
            return 0
    
    @staticmethod
    def _create_v13_graded_target(rank: int) -> float:
        """v13用: 複勝圏グレード付きターゲット (1着=1.0, 2着=0.5, 3着=0.25, 着外=0)"""
        if rank == 1:
            return 1.0
        elif rank == 2:
            return 0.5
        elif rank == 3:
            return 0.25
        else:
            return 0.0
    
    def split_and_create_dataset(self, df: pd.DataFrame, valid_year: int = 2025,
                                  target_type: str = "ranking") -> Dict[str, Dict]:
        """
        データを分割してデータセットを作成します。

        Args:
            df (pd.DataFrame): 前処理済みの全データ。
            valid_year (int): 検証に使用する年。Trainはこれより前の年、Testはこれより後の年になる。
            target_type (str): ターゲット種別 ("ranking" or "v13_graded")

        Returns:
            Dict: train, valid, test それぞれの {'X', 'y', 'group'} を含む辞書。
        """
        logger.info(f"データセットの分割と作成を開始 (Valid Year: {valid_year}, Target Type: {target_type})...")

        # ターゲット変数の作成
        if 'target' not in df.columns:
            if target_type == "v13_graded":
                # v13堅実モデル用: 複勝圏グレード付き
                logger.info("📊 v13用ターゲット生成: 1着=1.0, 2着=0.5, 3着=0.25, 着外=0")
                df['target'] = df['rank'].apply(self._create_v13_graded_target)
            else:
                # v12互換: ランキング用 (1着=3, 2着=2, 3着=1, 着外=0)
                df['target'] = df['rank'].apply(self._create_ranking_target)

        # 時系列分割
        # Train: 2010 ~ valid_year - 1 (Expanded start range)
        # Valid: valid_year
        # Test: valid_year + 1 ~
        train_df = df[df['year'] < valid_year].copy()
        valid_df = df[df['year'] == valid_year].copy()
        test_df = df[df['year'] > valid_year].copy()

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
        df = df.sort_values('race_id')

        # グループ情報
        group = df.groupby('race_id').size().to_numpy()

        # 特徴量 (X) と ターゲット (y) の分離
        # 【重要】未来情報（レース結果）を含むカラムは全て削除する
        drop_cols = [
            # ID・メタデータ
            'race_id', 'date', 'title', 'horse_id', 'horse_name',
            'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
            # 目的変数
            'rank', 'target', 'rank_str',
            # 未来情報 (Result)
            'time', 'raw_time',       # ← raw_time (1355など) が残っていると即リーク
            'passing_rank',           # 通過順
            'last_3f',                # 上がり3F
            'odds', 'popularity',     # オッズ・人気
            'weight',                 # 当日馬体重
            # 'weight_diff',          # ← 有効化 (Advanced Featuresで生成)
            'weight_diff_val', 'weight_diff_sign', # 元データにある場合は削除（重複回避）
            'winning_numbers', 'payout', 'ticket_type', # 払い戻し
            # PC-KEIBA特有のカラム（もしあれば）
            'pass_1', 'pass_2', 'pass_3', 'pass_4',
            
            # --- Leakage Features to Drop (Phase 11.1 fix) ---
            # These are derived from current race result or future odds
            'slow_start_recovery', 'pace_disadvantage', 'wide_run',
            'track_bias_disadvantage', 'outer_frame_disadv',
            'odds_race_rank', 'popularity_race_rank',
            'odds_deviation', 'popularity_deviation',
            
            # --- Low Impact Features to Drop (v5 Feature Selection) ---
            # 今回(v8)は再評価のため残す
            # 'race_avg_prize',         # 重要度 0
            # 'race_pace_cat',          # 重要度 0
            # 'total_prize',            # 重要度 0
            # 'is_long_break',          # 重要度 0
            # 'race_nige_horse_count',  # 重要度 9
            # 'race_nige_bias',         # 重要度 46
            # 'horse_pace_disadv_rate', # 重要度 74
            # 'weather_num',            # 重要度 92
            # 'weekday',                # 重要度 119
            
            # --- v6 Ineffective Features (重要度 0) ---
            # 'frame_zone',             # 重要度 0
            # 'distance_category',      # 重要度 0
            # 'state_num',              # 重要度 0
            # 'surface_num',            # 重要度 0
            
            # --- v7 Market Features (馬の能力と無関係) ---
            'lag1_odds',              # 前走オッズ（市場評価）
            'lag1_popularity',        # 前走人気（市場評価）
        ]
        # Sample Weights for Odds-Weighted Loss (Phase 15)
        # Use log1p(odds) to prioritize high-value winners without excessive noise sensitivity
        # Default weight = 1.0
        # Winner (Target > 0) weight = 1.0 + np.log1p(odds)
        w = np.ones(len(df))
        if 'odds' in df.columns:
            # fillna(1.0) and use log1p
            odds = df['odds'].fillna(1.0)
            # Apply weight only for Top 3 (target > 0)
            # w[df['target'] > 0] = 1.0 + np.log1p(odds[df['target'] > 0])
            # Wait, log1p of 1.0 is 0.7. log1p of 100 is 4.6.
            # Base weight 1.0. Bonus is log1p(odds).
            is_winner = df['target'] > 0
            w[is_winner] = 1.0 + np.log1p(odds[is_winner])

        # 存在しないカラムをdropしようとしてもエラーにならないように errors='ignore'
        X = df.drop(columns=drop_cols, errors='ignore')

        # カテゴリ変数がobject型のままだとLightGBMで扱いにくい場合があるため数値型のみ選択
        # (feature_engineeringで数値化済み前提)
        X = X.select_dtypes(exclude=['object'])

        y = df['target']

        return {'X': X, 'y': y, 'group': group, 'w': w}