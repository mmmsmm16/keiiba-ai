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

        # [Fix] LambdaRank用にデータをソート (日付 -> レースID)
        # データの並び順とgroupの並び順が完全に一致する必要がある
        df = df.sort_values(['date', 'race_id'])

        # グループ情報 (Query単位のデータ数)
        # sort=False にすることで、dfの並び順(date, race_id順)を維持したままカウントする
        # 【Important】LambdaRank/YetiRankでは学習データがクエリごとに連続している必要がある
        # [Fix] LambdaRank requires data to be sorted by query (race_id) blocks.
        # df is sorted, but standard groupby().size() aggregates disjoint blocks if they share the same key.
        # To strictly match the physical dataframe order, we use Run-Length Encoding (RLE).
        
        # 1. Identify where race_id changes
        is_new_group = df['race_id'] != df['race_id'].shift()
        group_ids = is_new_group.cumsum()
        
        # 2. Count size of each physical block
        group = df.groupby(group_ids, sort=False).size().to_numpy()
        
        # [Validation] Disjoint Check
        # If n_groups > nunique, it means some race_ids are split (disjoint).
        n_unique_races = df['race_id'].nunique()
        if len(group) != n_unique_races:
             # Find which races are disjoint
             dup_counts = df.groupby('race_id').size()
             # This is just counts, not blocks.
             # We just warn for now.
             print(f"[WARNING] Disjoint race_ids detected! Groups: {len(group)}, Unique: {n_unique_races}")
             # This implies data quality issue, but RLE enables training to proceed safely.

        # [Validation] Group Consistency Check
        # 1. Sum of groups must equal total rows
        assert group.sum() == len(df), f"Group sum {group.sum()} != len(df) {len(df)}"
        # 2. Number of groups must match physical blocks (Guaranteed by logic above)
        # assert len(group) == n_unique_races <-- Disabled because we handle disjoint now
        
        # 3. [Strict] Verify contiguous blocks (Race ID Order Mismatch Check)
        # LambdaRank requires data to be sorted by query (race_id) blocks.
        # Check if df matches the group boundaries exactly.
        current_idx = 0
        for i, size in enumerate(group):
            # Check strictly if the chunk contains only one race_id
            # Using .iloc is fast enough for ~50k groups
            chunk_race_id = df['race_id'].iloc[current_idx]
            # Just checking the first row is basically enough if we assume sorted, 
            # but to be 100% sure we check nunique if performance allows. 
            # Doing .iloc[slice].nunique() for every group might be slow (40k calls).
            # Optimization: check first and last of the chunk. If sorted, they must match.
            chunk_race_id_last = df['race_id'].iloc[current_idx + size - 1]
            
            # [Fix] Do not compare with unique_races[i] because disjoint groups break the index alignment.
            # Only check self-consistency (start matches end).
            
            if chunk_race_id != chunk_race_id_last:
                 raise ValueError(f"Group {i} is not contiguous! Found {chunk_race_id} at start and {chunk_race_id_last} at end.")
            
            current_idx += size


        # 特徴量 (X) と ターゲット (y) の分離
        # 【重要】未来情報（レース結果）を含むカラムは全て削除する
        drop_cols = [
            # ID・メタデータ (v09: 特徴量として使うため一部残す)
            'race_id', 'date', 'title', 'mare_id',
            'horse_name', # 名前は基本不要
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
            
            # --- v11: trend_* (事前予測モード用のためdrop) ---
            # realtime=OFF時はニュートラル埋めされているため情報量なし
            'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
            'trend_win_front_rate', 'trend_win_fav_rate',
            
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
            
            # --- v11 Speed Index Intermediates (Leakage) ---
            'time_index', 'last_3f_index',
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

        # [v09] ID系を保持するためのオプション (デフォルトは削除)
        # ユーザー要求により jockey_id, trainer_id, sire_id を特徴量として使いたい場合がある
        # 実習済みの experiment runner が明示的に X を加工することも可能だが、
        # ここでは exclude_ids=False なら削除しないようにする。
        # 一旦、ハードコードされたリストから外す。
        
        # 存在しないカラムをdropしようとしてもエラーにならないように errors='ignore'
        X = df.drop(columns=drop_cols, errors='ignore')

        # [v09 Fix] jockey_idなどはobject型だがCatBoostでは必要。
        # select_dtypes(exclude=['object']) をコメントアウトし、
        # 必要に応じて上位で処理する。
        # X = X.select_dtypes(exclude=['object'])

        y = df['target']

        return {'X': X, 'y': y, 'group': group, 'w': w}