
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Callable, Dict, Optional
import hashlib
import gc
from tqdm import tqdm

from .features import temporal_stats, class_stats, segment_stats, head_to_head, training_detail, horse_gear, race_attributes, impost_features, form_trend, race_conditions, race_structure, lap_fit, rating_elo, deep_lag_extended, stable_form, horse_events, aptitude_smoothing, relative_expansion

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    特徴量ブロックの作成、キャッシュ、ロードを管理するクラス
    """
    def __init__(self, cache_dir: str = "data/features"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.registry: Dict[str, Callable] = {}
        
        # デフォルトのブロックを登録
        self._register_default_blocks()

    def _register_default_blocks(self):
        """標準の特徴量ブロックを登録する"""
        self.registry["base_attributes"] = self._compute_base_attributes
        self.registry["history_stats"] = self._compute_history_stats
        self.registry["jockey_stats"] = self._compute_jockey_stats
        self.registry["pace_stats"] = self._compute_pace_stats
        self.registry["bloodline_stats"] = self._compute_bloodline_stats
        self.registry["odds_features"] = self._compute_odds_features
        self.registry["training_stats"] = self._compute_training_stats
        self.registry["burden_stats"] = self._compute_burden_stats
        self.registry["changes_stats"] = self._compute_change_stats
        self.registry["aptitude_stats"] = self._compute_aptitude_stats
        self.registry["speed_index_stats"] = self._compute_speed_index_stats
        self.registry["pace_pressure_stats"] = self._compute_pace_pressure_stats
        self.registry["relative_stats"] = self._compute_relative_stats
        self.registry["jockey_trainer_stats"] = self._compute_jockey_trainer_stats
        
        # M4 New Blocks
        self.registry["temporal_jockey_stats"] = self._compute_temporal_jockey_stats
        self.registry["temporal_trainer_stats"] = self._compute_temporal_trainer_stats
        self.registry["class_stats"] = self._compute_class_stats
        self.registry["segment_stats"] = self._compute_segment_stats
        self.registry["risk_stats"] = self._compute_risk_stats
        self.registry["course_aptitude"] = self._compute_course_aptitude
        self.registry["extended_aptitude"] = self._compute_extended_aptitude
        self.registry["runstyle_fit"] = self._compute_runstyle_fit
        self.registry["jockey_trainer_compatibility"] = self._compute_jockey_trainer_compatibility
        self.registry["interval_aptitude"] = self._compute_interval_aptitude
        self.registry["physique_training"] = self._compute_physique_training
        self.registry["jockey_strategy"] = self._compute_jockey_strategy
        self.registry["race_dynamics"] = self._compute_race_dynamics
        self.registry["sire_aptitude"] = self._compute_sire_aptitude
        
        # Phase T: Odds Fluctuation
        # Phase T: Odds & Bias
        self.registry["odds_fluctuation"] = self._compute_odds_fluctuation
        self.registry["track_bias"] = self._compute_track_bias
        
        # Phase T2: New Feature Blocks (2026-01)
        self.registry["frame_bias"] = self._compute_frame_bias
        self.registry["weight_pattern"] = self._compute_weight_pattern
        self.registry["rest_pattern"] = self._compute_rest_pattern
        self.registry["corner_dynamics"] = self._compute_corner_dynamics
        self.registry["head_to_head"] = self._compute_head_to_head
        self.registry["training_detail"] = self._compute_training_detail
        self.registry["training_detail"] = self._compute_training_detail
        self.registry["bloodline_detail"] = self._compute_bloodline_detail
        self.registry["strategy_pattern"] = self._compute_strategy_pattern
        self.registry["horse_gear"] = self._compute_horse_gear

        # T2 Refined v3
        self.registry["race_attributes"] = self._compute_race_attributes
        self.registry["impost_features"] = self._compute_impost_features
        self.registry["form_trend"] = self._compute_form_trend
        self.registry["race_conditions"] = self._compute_race_conditions
        self.registry["race_structure"] = self._compute_race_structure
        self.registry["lap_fit"] = self._compute_lap_fit
        self.registry["rating_elo"] = self._compute_rating_elo
        self.registry["deep_lag_extended"] = self._compute_deep_lag_extended
        self.registry["stable_form"] = self._compute_stable_form
        self.registry["horse_events"] = self._compute_horse_events
        self.registry["aptitude_smoothing"] = self._compute_aptitude_smoothing
        self.registry["aptitude_smoothing"] = self._compute_aptitude_smoothing
        self.registry["relative_expansion"] = self._compute_relative_expansion
        
        # Batch 1 (Quick Win)
        self.registry["mining_features"] = self._compute_mining_features
        self.registry["track_aptitude"] = self._compute_track_aptitude
        
        # Batch 2 (Jockey/Trainer Attributes & Logistics)
        self.registry["attribute_features"] = self._compute_attribute_features
        self.registry["logistics_features"] = self._compute_logistics_features
        
        # Batch 3 (Lap Analysis & Pace Features)
        self.registry["pace_features"] = self._compute_pace_features
        
        # Batch 4 (Corner Position Features)
        self.registry["corner_features"] = self._compute_corner_features

    def _compute_risk_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import risk_stats
        return risk_stats.compute_risk_stats(df)

    def _compute_sire_aptitude(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import sire_aptitude
        return sire_aptitude.compute_sire_aptitude(df)
        
    def _compute_race_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import race_dynamics
        return race_dynamics.compute_race_dynamics(df)
        
    def _compute_jockey_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import jockey_strategy
        return jockey_strategy.compute_jockey_strategy(df)

    def _compute_physique_training(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import physique_training
        return physique_training.compute_physique_training(df)
        
    def _compute_interval_aptitude(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import interval_aptitude
        return interval_aptitude.compute_interval_aptitude(df)

    def _compute_course_aptitude(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import course_aptitude
        return course_aptitude.compute_course_aptitude(df)
    
    def _compute_extended_aptitude(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import extended_aptitude
        return extended_aptitude.compute_extended_aptitude(df)

    def _compute_runstyle_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import runstyle_fit
        return runstyle_fit.compute_runstyle_fit(df)
        
    def _compute_jockey_trainer_compatibility(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import jockey_trainer_compatibility
        return jockey_trainer_compatibility.compute_jockey_trainer_compatibility(df)

    def _compute_odds_fluctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import odds_fluctuation
        return odds_fluctuation.compute_odds_fluctuation(df)

    def _compute_track_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import track_bias
        return track_bias.calculate_track_bias_features(df)

    def _compute_frame_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import frame_bias
        return frame_bias.compute_frame_bias(df)

    def _compute_weight_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import weight_pattern
        return weight_pattern.compute_weight_pattern(df)

    def _compute_rest_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import rest_pattern
        return rest_pattern.compute_rest_pattern(df)

    def _compute_corner_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import corner_dynamics
        return corner_dynamics.compute_corner_dynamics(df)

    def _compute_bloodline_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import bloodline_detail
        return bloodline_detail.compute_nicks_stats(df)

    def _compute_horse_gear(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.horse_gear import HorseGearFeatures
        return HorseGearFeatures().transform(df)

    # --- T2 Refined v3 Methods ---
    def _compute_race_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import race_attributes
        return race_attributes.compute(df)

    def _compute_impost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import impost_features
        return impost_features.compute(df)

    def _compute_form_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import form_trend
        return form_trend.compute(df)

    def _compute_race_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import race_conditions
        return race_conditions.compute(df)

    def _compute_race_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import race_structure
        return race_structure.compute(df)
        
    def _compute_lap_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import lap_fit
        return lap_fit.compute(df)

    def _compute_rating_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import rating_elo
        return rating_elo.compute(df)

    def _compute_deep_lag_extended(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import deep_lag_extended
        return deep_lag_extended.compute(df)

    def _compute_stable_form(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import stable_form
        return stable_form.compute(df)

    def _compute_horse_events(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import horse_events
        return horse_events.compute(df)

    def _compute_aptitude_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import aptitude_smoothing
        return aptitude_smoothing.compute(df)

    def _compute_relative_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import relative_expansion
        return relative_expansion.compute(df)

    def _compute_mining_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.mining_features import MiningFeatureGenerator
        return MiningFeatureGenerator().transform(df)

    def _compute_track_aptitude(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.track_aptitude import TrackAptitudeFeatureGenerator
        return TrackAptitudeFeatureGenerator().transform(df)

    def _compute_attribute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.attribute_features import AttributeFeatureGenerator
        return AttributeFeatureGenerator().transform(df)

    def _compute_logistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.logistics_features import LogisticsFeatureGenerator
        return LogisticsFeatureGenerator().transform(df)

    def _compute_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.pace_features import PaceFeatureGenerator
        return PaceFeatureGenerator().transform(df)

    def _compute_corner_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features.corner_features import CornerPositionFeatureGenerator
        return CornerPositionFeatureGenerator().transform(df)

    def _compute_strategy_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 戦略パターン (Strategy & Interactions)
        - is_dist_short_nige: 距離短縮 x 逃げ
        - is_jockey_change_dist_change: 騎手変更 x 距離変更
        - lag1_bad_luck_proxy: 前走不利 (上位人気/3F上位 + 敗北 + 小差)
        """
        logger.info("ブロック計算中: strategy_pattern")
        
        # 必要なカラムの確認
        req_cols = ['horse_id', 'date', 'distance', 'jockey_id', 'rank', 'time_diff', 'last_3f', 'pass_1']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        df_sorted = df.copy()
        if 'date' in df_sorted.columns:
            df_sorted = df_sorted.sort_values(['horse_id', 'date'])
        
        grp = df_sorted.groupby('horse_id')
        
        # 1. Dist Change & Jockey Change logic
        df_sorted['prev_dist'] = grp['distance'].shift(1)
        df_sorted['dist_change_val'] = df_sorted['distance'] - df_sorted['prev_dist']
        df_sorted['is_dist_short'] = (df_sorted['dist_change_val'] < 0).astype(int)
        df_sorted['is_dist_change'] = (df_sorted['dist_change_val'] != 0).astype(int)
        
        df_sorted['prev_jockey'] = grp['jockey_id'].shift(1)
        df_sorted['is_jockey_change'] = (df_sorted['jockey_id'] != df_sorted['prev_jockey']).astype(int)
        
        # 2. is_dist_short_nige
        # 条件: 距離短縮 かつ (前走4角1番手 OR 今回逃げ予測)
        # 今回逃げ予測は難しいので「前走4角1番手」で代用
        # pass_1: "1" or "1-1" etc.
        def is_nige_func(val):
             try:
                 # [Fix] Convert to int to handle "01" vs "1"
                 return 1 if int(str(val).split('-')[0]) == 1 else 0
             except: return 0
             
        df_sorted['is_nige_instant'] = df_sorted['pass_1'].apply(is_nige_func)
        df_sorted['lag1_is_nige'] = grp['is_nige_instant'].shift(1).fillna(0)
        
        df_sorted['is_dist_short_nige'] = (df_sorted['is_dist_short'] == 1) & (df_sorted['lag1_is_nige'] == 1)
        df_sorted['is_dist_short_nige'] = df_sorted['is_dist_short_nige'].astype(int)
        
        # 3. is_jockey_change_dist_change
        df_sorted['is_jockey_change_dist_change'] = (df_sorted['is_jockey_change'] == 1) & (df_sorted['is_dist_change'] == 1)
        df_sorted['is_jockey_change_dist_change'] = df_sorted['is_jockey_change_dist_change'].astype(int)
        
        # 4. lag1_bad_luck_proxy
        # 前走: 3F順位 <= 3, 着順 >= 6, 着差 <= 0.5
        
        # 3F Rank calculation per race
        # Note: df contains all history. Ideally grouping by race_id first to calc rank, 
        # but this block gets sorted by horse_id.
        # Need to re-sort or use transform on original df if possible.
        # We can calculate 3F rank globally first? No, per race.
        # To avoid re-sorting repeatedly, let's assume we can do it.
        
        # Sort by race_id temporarily to calc 3F rank
        # Need to keep index to restore?
        df_sorted['temp_idx'] = df_sorted.index
        df_race = df_sorted[['race_id', 'last_3f', 'temp_idx']].sort_values(['race_id'])
        
        # Group by race, rank last_3f (ascending=True, smaller is faster)
        df_race['last_3f_rank'] = df_race.groupby('race_id')['last_3f'].rank(method='min', ascending=True)
        
        # Merge back 3F rank
        # Use pandas merge or map? Map is faster if unique index.
        # df_race index is shuffled.
        # Join on temp_idx
        df_sorted = pd.merge(df_sorted, df_race[['temp_idx', 'last_3f_rank']], on='temp_idx', how='left')
        
        # Now groupby horse again to shift
        # Re-sort to be sure
        df_sorted = df_sorted.sort_values(['horse_id', 'date'])
        grp2 = df_sorted.groupby('horse_id')
        
        df_sorted['lag1_rank'] = grp2['rank'].shift(1)
        df_sorted['lag1_time_diff'] = grp2['time_diff'].shift(1)
        df_sorted['lag1_3f_rank'] = grp2['last_3f_rank'].shift(1)
        
        mask_bad_luck = (
            (df_sorted['lag1_3f_rank'] <= 3) & 
            (df_sorted['lag1_rank'] >= 6) & 
            (df_sorted['lag1_time_diff'] <= 0.5)
        )
        df_sorted['lag1_bad_luck_proxy'] = mask_bad_luck.astype(int)
        
        feats = ['is_dist_short_nige', 'is_jockey_change_dist_change', 'lag1_bad_luck_proxy']
        
        return df_sorted[keys + feats].copy()


    def _compute_base_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 基本属性 (Raw Data)
        """
        logger.info("ブロック計算中: base_attributes")
        # 必要なカラムを厳密に定義
        cols = [
            'horse_number', 'bracket_number', 'age', 'sex', 
            'weight', 'weight_diff', 'jockey_id', 'trainer_id',
            'sire_id', 'mare_id',
            # v10: Race Context
            'grade_code', 'kyoso_joken_code', 'distance', 'surface', 'venue'
        ]
        # 存在確認
        avail_cols = [c for c in cols if c in df.columns]
        
        # マージ用キー: race_id, horse_number, horse_id
        keys = ['race_id', 'horse_number', 'horse_id']
        
        # keysに含まれるカラムはfeaturesから除外して重複を防ぐ
        final_cols = keys + [c for c in avail_cols if c not in keys]
        
        return df[final_cols].copy()

    def _compute_history_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 過去走集計 (History Stats)
        - interval: 前走からの間隔
        - lag1-5_rank: 過去1-5走の着順
        - lag1-5_time_diff: 過去1-5走のタイム差
        - mean_rank_5: 過去5走平均着順
        - mean_time_diff_5: 過去5走平均タイム差
        """
        logger.info("ブロック計算中: history_stats")
        
        # 必要なカラムの確認
        req_cols = ['horse_id', 'date', 'rank', 'time_diff', 'race_id', 'horse_number', 'honshokin', 'fukashokin', 'grade_code']
        for c in req_cols:
            if c not in df.columns:
                # v10: honshokin, fukashokin, grade_code are optional for compatibility but required for v10
                logger.warning(f"history_stats: {c} not found. 0 filling or skipping.")
                
        # ソート: horse_id, date
        df_sorted = df.sort_values(['horse_id', 'date']).copy()
        
        # グループ化
        grp = df_sorted.groupby('horse_id')
        
        # 0. Run Count (出走回数 - 0 start)
        # modelが「初出走」を区別できるようにする
        df_sorted['run_count'] = grp.cumcount()

        # --- v10: Competitor Level Logic ---
        # 1. Calculate Cumulative Prize for each horse (Time-aware)
        # その時点までの獲得総賞金 (今回の賞金も含む: "勝った後の強さ"を表すため)
        df_sorted['daily_prize'] = df_sorted['honshokin'].fillna(0) + df_sorted['fukashokin'].fillna(0)
        df_sorted['cumulative_prize'] = grp['daily_prize'].cumsum()
        
        # 2. Race-level Aggregation
        # レースごとの平均レベル(avg_prize)と、勝者のレベル(winner_prize)
        # 内部計算用に一時的なDataFrameを作る
        race_stats = df_sorted.groupby('race_id')['cumulative_prize'].mean().rename('race_level_avg')
        
        # 勝者 (rank=1) の賞金
        # 同着(Dead Heat)がある場合、race_idが重複して map でエラーになるため、groupby().mean() で一意にする
        winners = df_sorted[df_sorted['rank'] == 1].groupby('race_id')['cumulative_prize'].mean().rename('race_winner_level')
        # ここでは後でfillna(0)する。
        
        # Merge back
        # sort順を維持するために merge ではなく map を使用 (高速化)
        df_sorted['race_level_avg'] = df_sorted['race_id'].map(race_stats)
        df_sorted['race_winner_level'] = df_sorted['race_id'].map(winners)
        
        # 欠損補完 (勝者データ取得漏れなど)
        df_sorted['race_level_avg'] = df_sorted['race_level_avg'].fillna(0)
        # 自分が勝因でなければ不明 -> 0
        df_sorted['race_winner_level'] = df_sorted['race_winner_level'].fillna(0)
        
        # 1. Interval
        # 前回のdateを取得 (lag1)
        df_sorted['prev_date'] = grp['date'].shift(1)
        # 日数差
        df_sorted['interval'] = (pd.to_datetime(df_sorted['date']) - pd.to_datetime(df_sorted['prev_date'])).dt.days
        
        # 2. Deep Lag Features (Lag 1 to 5)
        # 過去1走〜5走までの各値を特徴量にする
        
        lag_feats = []
        for i in range(1, 6):
            # Rank
            col_rank = f"lag{i}_rank"
            df_sorted[col_rank] = grp['rank'].shift(i)
            lag_feats.append(col_rank)
            
            # Time Diff (Fill with 3.0 if missing)
            col_td = f"lag{i}_time_diff"
            df_sorted[col_td] = grp['time_diff'].shift(i).fillna(3.0)
            lag_feats.append(col_td)
            
            # Competitor Levels
            col_rl = f"lag{i}_race_level"
            df_sorted[col_rl] = grp['race_level_avg'].shift(i).fillna(0)
            lag_feats.append(col_rl)
            
            col_wl = f"lag{i}_winner_level"
            df_sorted[col_wl] = grp['race_winner_level'].shift(i).fillna(0)
            lag_feats.append(col_wl)
            
            # Grade (Categorical)
            col_gr = f"lag{i}_grade"
            df_sorted[col_gr] = grp['grade_code'].shift(i).fillna('missing')
            lag_feats.append(col_gr)

            # Compatibility: 'lag1_rank' exists in lag_feats. 
            # 'prev_race_level' was lag1_race_level.
            # To maintain compatibility with existing feature names if needed?
            # Existing specific names: lag1_rank, lag1_time_diff, prev_race_level, prev_winner_level, prev_grade
            # Let's add alias for lag1 to keep backward compat if downstream models rely on specific names?
            # Or just update model config? The user is re-training, so new names are fine.
            # However, 'prev_race_level' etc. might be used in other blocks?
            # No, blocks are independent.
            # But let's verify if I should remove old 'prev_' names or keep lag1 as aliases.
            # I will simple generate lag1...5.
            # And for lag1 specifically, I can also keep 'prev_race_level' = lag1_race_level for safety or just replace.
            
        # Add alias for lag1 specific names previously used, just in case (though I will simply use lag1_xxx in feats)
        # The prompt asked for "lag5までの各特徴量".
        # I will remove 'prev_race_level' etc from the output list and replace with 'lag1_race_level' etc.
        # But 'lag1_rank' and 'lag1_time_diff' were already named so.
            
        
        # 3. Rolling Features (過去5走)
        # shift(1)してからrollingすることで当日のリークを防ぐ
        # min_periods=1で、1回でも過去走があれば算出
        # [Check Leakage] Safe: transform applies within group. shift(1) excludes current race.
        df_sorted['mean_rank_5'] = grp['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df_sorted['mean_time_diff_5'] = grp['time_diff'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # 欠損値処理
        # mean_time_diff_5 の欠損も 3.0 で埋めておく（一貫性のため）
        df_sorted['mean_time_diff_5'] = df_sorted['mean_time_diff_5'].fillna(3.0)
        
        # 抽出カラム
        feats = [
            'run_count', 'interval', 'mean_rank_5', 'mean_time_diff_5'
        ] + lag_feats
        
        keys = ['race_id', 'horse_number', 'horse_id'] # horse_idもキーに含める
        
        return df_sorted[keys + feats].copy()

    def _compute_jockey_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 騎手過去成績 (Jockey Stats)
        - 過去1年間の成績 (Win Rate, Top3 Rate, Avg Rank)
        - リーク防止のため、当該レース日付より前のデータのみを使用
        """
        logger.info("ブロック計算中: jockey_stats")
        
        req_cols = ['race_id', 'horse_number', 'horse_id', 'jockey_id', 'date', 'rank']
        for c in req_cols:
            if c not in df.columns:
                raise ValueError(f"jockey_statsにはカラム {c} が必要です。")
                
        # 必要なカラムに絞り、ソート
        df_sorted = df[req_cols].copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
        # NaN handling for IDs to prevent groupby dropping rows
        df_sorted['jockey_id'] = df_sorted['jockey_id'].fillna('99999').astype(str)
        
        df_sorted = df_sorted.sort_values(['jockey_id', 'date'])
        
        # ターゲット生成
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
        df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)
        
        # Groupby Jockey
        grp = df_sorted.groupby('jockey_id')
        
        # 過去1年間の集計 (Rolling '365D')
        # closed='left' で「当日を含まない」過去データを指定
        # on='date' で日付ベースの窓
        # min_periods=0 (初騎乗は0)
        
        # 注: groupby().rolling() はMultiIndexを返すため、reset_index等で調整が必要
        logger.info("  Calculing rolling 365D stats...")
        
        rolling = df_sorted.set_index('date').groupby('jockey_id')['is_win'].rolling('365D', closed='left')
        df_sorted['jockey_n_races'] = rolling.count().values
        df_sorted['jockey_win_count'] = rolling.sum().values
        
        rolling_top3 = df_sorted.set_index('date').groupby('jockey_id')['is_top3'].rolling('365D', closed='left')
        df_sorted['jockey_top3_count'] = rolling_top3.sum().values
        
        rolling_rank = df_sorted.set_index('date').groupby('jockey_id')['rank'].rolling('365D', closed='left')
        df_sorted['jockey_sum_rank'] = rolling_rank.sum().values
        
        # 特徴量計算 (レート)
        # 0除算回避のため fillna(0)
        df_sorted['jockey_win_rate'] = (df_sorted['jockey_win_count'] / df_sorted['jockey_n_races']).fillna(0)
        df_sorted['jockey_top3_rate'] = (df_sorted['jockey_top3_count'] / df_sorted['jockey_n_races']).fillna(0)
        df_sorted['jockey_avg_rank'] = (df_sorted['jockey_sum_rank'] / df_sorted['jockey_n_races']).fillna(0)
        
        # 欠損値は0埋め (初騎乗など)
        cols_to_fill = ['jockey_n_races', 'jockey_win_rate', 'jockey_top3_rate', 'jockey_avg_rank']
        df_sorted[cols_to_fill] = df_sorted[cols_to_fill].fillna(0)
        
        # 抽出
        feats = ['jockey_n_races', 'jockey_win_rate', 'jockey_top3_rate', 'jockey_avg_rank']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        return df_sorted[keys + feats].copy()



    def _compute_pace_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 展開・脚質特徴量 (Pace Stats)
        - 過去5走のコーナー通過順、Nige Rate、PCI
        """
        logger.info("ブロック計算中: pace_stats")
        
        req_cols = ['race_id', 'horse_number', 'horse_id', 'date', 'corner_1', 'passage_rate', 'time', 'last_3f']
        # data loaderによっては corner_1 ではなく passage_rate (1-1-1) のみの場合がある
        # JRA-VANの場合は `passage_rate` カラムがある想定
        if 'passage_rate' not in df.columns:
            # 必須ではないが、ないと脚質が計算できない
            # jvd_seにあるはず
            pass

        # 必要なカラムの存在確認 (time_seconds, last_3f はPCI用)
        # passage_rate はコーナー通過順
        check_cols = ['horse_id', 'date', 'race_id', 'horse_number']
        for c in check_cols:
            if c not in df.columns:
                raise ValueError(f"pace_statsにはカラム {c} が必要です。")
        
        df_sorted = df.copy() # 全カラム使う可能性があるのでコピー
        df_sorted = df_sorted.sort_values(['horse_id', 'date'])
        
        # 1. 最初のコーナー通過順 (First Corner Rank) の抽出
        # passage_rate: "1-1-1" or "10-10" or "NULL"
        def extract_first_corner(val):
            if pd.isna(val) or val == '':
                return np.nan
            try:
                # "-" で分割して最初の要素
                # "10-10" -> "10"
                # "05" -> "5"
                # まれに "*" などが入る場合があるかも？ -> try-exceptでNaNへ
                parts = str(val).split('-')
                if len(parts) > 0:
                    return float(parts[0])
            except:
                return np.nan
            return np.nan
            
        if 'passage_rate' in df_sorted.columns:
            df_sorted['first_corner_rank'] = df_sorted['passage_rate'].apply(extract_first_corner)
        elif 'pass_1' in df_sorted.columns:
             # pass_1 is aliased corner_1 (Rank) in Loader
             # if string "01", float conversion works same as corner_1
             # Use extract check just in case it is "01" string
             df_sorted['first_corner_rank'] = df_sorted['pass_1'].apply(extract_first_corner)
        elif 'corner_1' in df_sorted.columns:
            df_sorted['first_corner_rank'] = df_sorted['corner_1']
        else:
            # カラムがない場合はNaN
            df_sorted['first_corner_rank'] = np.nan

        # 2. PCI (Pace Change Index) 計算
        # (Last3F / (Time - Last3F)) * 100
        # Time - Last3F が前半(Total - 3F)
        def calc_pci(row):
            t = row.get('time')
            l3 = row.get('last_3f')
            # Convert to numeric if string
            try:
                t = float(t) if t is not None and not pd.isna(t) else np.nan
                l3 = float(l3) if l3 is not None and not pd.isna(l3) else np.nan
            except (ValueError, TypeError):
                return np.nan
            if pd.isna(t) or pd.isna(l3) or t <= l3 or l3 <= 0:
                return np.nan
            prev_dist_time = t - l3
            if prev_dist_time <= 0:
                return np.nan
            return (l3 / prev_dist_time) * 100
            
        df_sorted['pci'] = df_sorted.apply(calc_pci, axis=1)

        
        # 3. 集計用フラグ
        # 逃げ判定 (最初のコーナーが1番手)
        df_sorted['is_nige'] = (df_sorted['first_corner_rank'] == 1).astype(int)
        
        # 通過順正規化 (通過順 / 頭数) -> 頭数がない場合は概算(Nan)
        # weightなどで代用はできない。horse_countが欲しいが、base_attributesにはない。
        # ここでは簡易的に rank のmaxをそのレースの頭数とする手もあるが、Leakになる。
        # あるいは passage_rate があればそのまま rank を使う (絶対値でも傾向は出る)
        # User request: "avg_first_corner_norm" (通過順 / 出走頭数)
        # 出走頭数を取得するには race_id で groupby countするしかない
        # しかし pipeline は base_df しか受け取らない。base_df は全データなので計算可能。
        race_counts = df.groupby('race_id')['horse_number'].count().to_dict()
        df_sorted['n_horses'] = df_sorted['race_id'].map(race_counts)
        df_sorted['first_corner_norm'] = df_sorted['first_corner_rank'] / df_sorted['n_horses']
        
        # 4. 過去集計 (Shift 1 -> Rolling 5)
        # [Fix Leakage] transform(lambda) でグループ内完結
        grp = df_sorted.groupby('horse_id')
        
        # last_nige_rate (過去5走の逃げ率)
        # [Check Leakage] Safe: transform applies within group.
        df_sorted['last_nige_rate'] = grp['is_nige'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # avg_first_corner_norm (過去5走の平均通過順正規化)
        df_sorted['avg_first_corner_norm'] = grp['first_corner_norm'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # avg_pci (過去5走PCI平均)
        df_sorted['avg_pci'] = grp['pci'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # 欠損補完
        # nige_rate -> 0
        df_sorted['last_nige_rate'] = df_sorted['last_nige_rate'].fillna(0)
        # norm -> 0.5 (真ん中)
        df_sorted['avg_first_corner_norm'] = df_sorted['avg_first_corner_norm'].fillna(0.5)
        # pci -> 50 (平均的ペース?? 計算式によるが、1.0*100=100??)
        # Last3F=35, First=35 -> 1.0 * 100 = 100.
        # Last3F=34, First=36 (Slow Pace) -> 34/36 = 0.94 * 100 = 94.
        # Last3F=38, First=34 (Fast Pace) -> 38/34 = 1.11 * 100 = 111.
        # ユーザー数式の値域確認: 600m(3F) vs (Total-600m).
        # 1200m: 3F vs 3F. Even=100.
        # 2400m: 3F vs 9F. Even means 3F is 1/3 of time? No.
        # 2400m Total=150s. Last3f=35s. First=115s. 35/115 = 30.
        # なので距離によってPCIのベースラインが全然違う。
        # 過去5走の平均をとるなら、距離適性が混ざる。
        # しかしユーザー指示は「過去5走の平均」かつ「欠損は50で埋める」。
        # 50という値が適切かは不明だが、指示通り実装する。
        df_sorted['avg_pci'] = df_sorted['avg_pci'].fillna(50)
        
        feats = ['last_nige_rate', 'avg_first_corner_norm', 'avg_pci']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        return df_sorted[keys + feats].copy()

    def _compute_pace_pressure_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 展開圧 (Pace Pressure)
        - レース内の逃げ候補数と、その相互作用を計算。
        - 注意: 前走までの逃げ率(last_nige_rate)を利用するため、リークはない。
        """
        logger.info("ブロック計算中: pace_pressure_stats")
        
        # 必要なカラムの準備: last_nige_rate は pace_stats で計算されるが、
        # ここではBlock依存を避けるため、feature_pipeline内で完結させるか、
        # あるいは _compute_pace_stats のロジックを再利用/依存する。
        # 今回は独立性を保つため、pace_stats と同様のロジックで last_nige_rate を計算する。
        
        if 'pass_1' not in df.columns and 'passing_rank' not in df.columns:
            # カラムがない -> 計算不可
            keys = ['race_id', 'horse_number', 'horse_id']
            return df[keys].copy()

        df_sorted = df.sort_values(['date', 'race_id']).copy()
        
        # 1. is_nige (逃げ判定)
        # pass_1 (Corner 1) が 1 なら逃げ
        # '00' は通過順データなしを意味するので除外
        def is_nige_parse(s):
            if pd.isna(s) or s == '' or s == '00' or s == 0:
                return 0
            # Try splitting "1-1-1" or handling single value
            try:
                parts = str(s).split('-')
                if len(parts) > 0:
                    first_corner = float(parts[0])
                    # 1位通過 = 逃げ
                    return 1 if first_corner == 1 else 0
            except:
                pass
            return 0

        if 'running_style' in df_sorted.columns:
            # 1: 逃げ (Nige) in JRA-VAN
            df_sorted['is_nige'] = (pd.to_numeric(df_sorted['running_style'], errors='coerce') == 1).astype(int)
        elif 'pass_1' in df_sorted.columns:
            df_sorted['is_nige'] = df_sorted['pass_1'].apply(is_nige_parse)
        elif 'passing_rank' in df_sorted.columns:
            df_sorted['is_nige'] = df_sorted['passing_rank'].apply(is_nige_parse)
        else:
             df_sorted['is_nige'] = 0
             
        # 2. last_nige_rate (過去5走の逃げ率) - Shift(1) 必須
        grp_horse = df_sorted.groupby('horse_id')
        df_sorted['last_nige_rate'] = grp_horse['is_nige'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df_sorted['last_nige_rate'] = df_sorted['last_nige_rate'].fillna(0.0)
        df_sorted['nige_experience_n'] = grp_horse['is_nige'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=1).count()
        ).fillna(0.0)
        
        # Reliability-weighted nige score (threshold-free core)
        c = 5.0
        df_sorted['nige_score'] = df_sorted['last_nige_rate'] * np.sqrt(
            df_sorted['nige_experience_n'] / (df_sorted['nige_experience_n'] + c)
        )
        
        # 3. Race集計: レース内の逃げ圧力総和
        # last_nige_rate は "当該レース以前" の情報なので、レース内で集計してもリークしない。
        
        # a) 閾値版 (逃げ候補数)
        # [Fix] Lowered threshold from 0.5 to 0.2 - data shows max is ~0.4
        df_sorted['is_candidate'] = (df_sorted['last_nige_rate'] > 0.2).astype(int)
        
        # b) 連続値版 (圧力総和)
        grp_race = df_sorted.groupby('race_id')
        
        # transform でレースごとの総和を各行に付与
        df_sorted['race_nige_count_bin'] = grp_race['is_candidate'].transform('sum')
        df_sorted['race_nige_pressure_sum'] = grp_race['last_nige_rate'].transform('sum')
        df_sorted['race_nige_pressure_score_sum'] = grp_race['nige_score'].transform('sum')
        
        df_sorted['is_candidate_weighted'] = (df_sorted['nige_score'] > 0.15).astype(int)
        df_sorted['race_nige_count_weighted'] = grp_race['is_candidate_weighted'].transform('sum')
        
        # 4. Self Exclusion (自分を除く)
        df_sorted['race_nige_count_excl'] = df_sorted['race_nige_count_bin'] - df_sorted['is_candidate']
        df_sorted['is_nige_interaction'] = df_sorted['last_nige_rate'] * df_sorted['race_nige_count_excl']
        
        df_sorted['race_pressure_excl'] = df_sorted['race_nige_pressure_sum'] - df_sorted['last_nige_rate']
        df_sorted['nige_pressure_interaction'] = df_sorted['last_nige_rate'] * df_sorted['race_pressure_excl']
        df_sorted['race_pressure_score_excl'] = df_sorted['race_nige_pressure_score_sum'] - df_sorted['nige_score']
        df_sorted['nige_score_interaction'] = df_sorted['nige_score'] * df_sorted['race_pressure_score_excl']
        
        feats = [
            # 'last_nige_rate', # Computed in pace_stats. Don't overwrite.
            'race_nige_count_bin', 'race_nige_pressure_sum',
            'is_nige_interaction', 'nige_pressure_interaction',
            'nige_score', 'race_nige_pressure_score_sum',
            'race_nige_count_weighted', 'nige_score_interaction'
        ]
        

        
        keys = ['race_id', 'horse_number', 'horse_id']
        return df_sorted[keys + feats].copy()

    def _compute_relative_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 相対化特徴量 (Relative Stats)
        - レース内での相対的な数値を計算する (Z-score, Percentile)
        - 対象: 実力(SpeedIndex), 末脚(Last3F), 展開(Pace), 間隔(Interval), 斤量(Impost)
        - 注意: 全て race_id 単位で処理するためリークしない。
        -       ただし元データは「過去走の平均」などである必要がある。
        """
        logger.info("ブロック計算中: relative_stats")
        
        # ソート
        df_sorted = df.sort_values(['date', 'race_id']).copy()
        
        # --- 1. Base Features Calculation (Local Safe Re-impl) ---
        
        # A. Speed Index (Past Avg)
        if 'time' in df_sorted.columns and 'distance' in df_sorted.columns and 'venue' in df_sorted.columns:
            # Grouping for Standard Time
            # timeが欠損の場合は計算不可
            # std=0 fix: タイムの変動がないケースは稀だが、念のため
            
            # [Safe Expanding]
            # Time Standardization: (Time - Mean) / Std?? No, SpeedIndex logic is better.
            # Use simplified Speed Index: (BaseTime - Time) + 80.
            # BaseTime = Group Expanding Mean.
            
            keys_si = ['venue', 'distance', 'surface']
            grp_si = df_sorted.groupby(keys_si)
            
            # 1. Base Time (Past Avg of similar races)
            base_time = grp_si['time'].transform(lambda x: x.expanding().mean().shift(1))
            
            # 2. Local Speed Index (Current Race)
            # 距離係数は簡易的に 1.0 とする（厳密な指数ロジックは speed_index_stats にあるが、ここでは相対比較用なので線形変換で十分）
            # SI = (Base - Time) + 80
            # Timeが小さい（速い）ほど SIは大きくなる
            df_sorted['temp_si'] = (base_time - df_sorted['time']) + 80
            
            # 3. Lag1 Past Avg Speed Index
            # これが「この馬の実力値」
            grp_horse = df_sorted.groupby('horse_id')
            df_sorted['base_speed_index'] = grp_horse['temp_si'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            df_sorted['base_speed_index'] = df_sorted['base_speed_index'].fillna(50) # Default
            
        else:
            df_sorted['base_speed_index'] = 50

        # B. Last 3F (Past Avg)
        if 'last_3f' in df_sorted.columns:
            grp_horse = df_sorted.groupby('horse_id')
            df_sorted['base_last_3f'] = grp_horse['last_3f'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            # 欠損は平均35.0くらい？
            df_sorted['base_last_3f'] = df_sorted['base_last_3f'].fillna(35.0)
        else:
            df_sorted['base_last_3f'] = 35.0

        # C. Last Nige Rate (Past Avg)
        # これは pace_pressure_stats と同じロジック
        if 'pass_1' in df_sorted.columns:
            s_pass = pd.to_numeric(df_sorted['pass_1'], errors='coerce')
            df_sorted['is_nige'] = (s_pass == 1).astype(int)
        elif 'passing_rank' in df_sorted.columns:
            def is_nige_parse(s):
                if not isinstance(s, str): return 0
                parts = s.split('-')
                if len(parts) > 0:
                    try: return 1 if float(parts[0]) == 1 else 0
                    except: return 0
                return 0
            df_sorted['is_nige'] = df_sorted['passing_rank'].apply(is_nige_parse)
        else:
            df_sorted['is_nige'] = 0
            
        grp_horse = df_sorted.groupby('horse_id')
        df_sorted['base_nige_rate'] = grp_horse['is_nige'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df_sorted['base_nige_rate'] = df_sorted['base_nige_rate'].fillna(0)

        # D. Days Since Last
        df_sorted['prev_date'] = grp_horse['date'].shift(1)
        df_sorted['base_interval'] = (df_sorted['date'] - df_sorted['prev_date']).dt.days
        df_sorted['base_interval'] = df_sorted['base_interval'].fillna(999) # First time -> Long interval

        # E. Impost (Current)
        # Raw value
        if 'impost' not in df_sorted.columns:
            df_sorted['impost'] = 55.0
        df_sorted['base_impost'] = df_sorted['impost'].fillna(55.0)


        # --- 2. Relative Stats Calculation (Race-level) ---
        target_cols = {
            'base_speed_index': 'speed_index',
            'base_last_3f': 'last_3f',
            'base_nige_rate': 'nige_rate',
            'base_interval': 'interval',
            'base_impost': 'impost'
        }
        
        grp_race = df_sorted.groupby('race_id')
        
        feats = []
        for col, alias in target_cols.items():
            # Mean / Std
            r_mean = grp_race[col].transform('mean')
            r_std = grp_race[col].transform('std').fillna(0) # 1頭だけなどの場合はNaN->0
            
            # Z-score: (x - mean) / std
            # std=0 の場合は 0 とする
            # numpy.where is faster or pandas apply
            # Division by zero handled by numpy usually logic?
            # Safe division:
            z_col = f"relative_{alias}_z"
            
            # std=0 detection
            # If std=0, then all values are same (or 1 record). Z should be 0.
            # Using simple math: (x - mean) / (std + 1e-6) ?
            # Better: explicit handle
            
            # Note: fillna(1) for std to avoid division by zero, then mask 0s?
            # Or just replace 0 with NaN then fillna?
            
            # (x - mean)
            diff = df_sorted[col] - r_mean
            
            # z = diff / std. If std==0, result is inf.
            # Handle std=0:
            r_std_safe = r_std.replace(0, 1.0)
            z_score = diff / r_std_safe
            
            # If original std was 0, force z-score to 0
            is_std_zero = (r_std == 0)
            z_score = z_score.mask(is_std_zero, 0.0)
            
            df_sorted[z_col] = z_score.fillna(0.0)
            feats.append(z_col)
            
            # Percentile (Rank pct)
            # rank(pct=True) returns 0..1
            p_col = f"relative_{alias}_pct"
            df_sorted[p_col] = grp_race[col].transform(lambda x: x.rank(pct=True, method='average'))
            df_sorted[p_col] = df_sorted[p_col].fillna(0.5)
            feats.append(p_col)

            # [New] Raw Diff (Value - Mean)
            # Z-score depends on std. If std is small (tight race), Z-score explodes.
            # Raw diff provides absolute difference context.
            d_col = f"relative_{alias}_diff"
            df_sorted[d_col] = diff.fillna(0.0)
            feats.append(d_col)

        keys = ['race_id', 'horse_number', 'horse_id']
        return df_sorted[keys + feats].copy()
 


    def _compute_jockey_trainer_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 騎手×厩舎 特徴量 (Jockey x Trainer Stats)
        - 騎手と厩舎のペアごとの累積成績
        - [Sorting]: jockey_id, trainer_id, date, race_id で順序を固定
        - [Smoothing]: Laplace Smoothing (prior=20) を適用
        - [Leakage Fix]: shift(1) により当該レースを除外
        """
        logger.info("ブロック計算中: jockey_trainer_stats")
        
        req_cols = ['race_id', 'horse_number', 'horse_id', 'date', 'jockey_id', 'trainer_id', 'rank']
        for c in req_cols:
            if c not in df.columns:
                raise ValueError(f"jockey_trainer_statsにはカラム {c} が必要です。")
                
        # 1. 厳密なソート (同一日・同一ペアの順序を固定)
        df_sorted = df[req_cols].copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        df_sorted = df_sorted.sort_values(['jockey_id', 'trainer_id', 'date', 'race_id'])
        
        # ターゲット作成
        df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
        
        # 2. GroupByで累積集計
        grp = df_sorted.groupby(['jockey_id', 'trainer_id'])
        
        # cumcount()は 0, 1, 2... と増えるため、そのまま「過去の走数」として使える (shift(1)相当)
        df_sorted['jt_run_count'] = grp.cumcount()
        
        # cumsum().shift(1) で「過去の合計」を取得
        df_sorted['jt_top3_sum'] = grp['is_top3'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df_sorted['jt_win_sum'] = grp['is_win'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df_sorted['jt_rank_sum'] = grp['rank'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        
        # 3. 平滑化 (Laplace Smoothing)
        # prior=20, global_rate は全件平均を使用
        prior = 20
        global_top3_rate = df['rank'].le(3).mean()
        global_win_rate = df['rank'].eq(1).mean()
        global_avg_rank = df['rank'].mean()
        
        df_sorted['jt_top3_rate_smoothed'] = (df_sorted['jt_top3_sum'] + prior * global_top3_rate) / (df_sorted['jt_run_count'] + prior)
        df_sorted['jt_win_rate_smoothed'] = (df_sorted['jt_win_sum'] + prior * global_win_rate) / (df_sorted['jt_run_count'] + prior)
        
        # 平均着順 (0除算回避)
        df_sorted['jt_avg_rank'] = np.where(
            df_sorted['jt_run_count'] > 0,
            df_sorted['jt_rank_sum'] / df_sorted['jt_run_count'],
            global_avg_rank
        )
        
        # 特徴量リスト
        feats = ['jt_run_count', 'jt_top3_rate_smoothed', 'jt_win_rate_smoothed', 'jt_avg_rank']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        return df_sorted[keys + feats].copy()

    def _compute_bloodline_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 血統特徴量 (Bloodline Stats)
        - 種牡馬(sire_id)ごとの累積成績 (勝率、芝勝率、ダート勝率)
        - 累積データなので expanding (cumsum)を使用。Shift(1)でリーク回避。
        """
        logger.info("ブロック計算中: bloodline_stats")
        
        # 必要なカラム: sire_id, surface(またはtrack_code), rank
        # 必要なカラム: sire_id, surface(またはtrack_code), rank, state
        req_cols = ['race_id', 'horse_number', 'horse_id', 'date', 'sire_id', 'surface', 'rank', 'state']
        for c in req_cols:
            if c not in df.columns:
                # surfaceは 'track_code' から cleansing で作られる想定だが
                # loader.pyによると 'surface' カラムが存在する (map_surface適用済)
                raise ValueError(f"bloodline_statsにはカラム {c} が必要です。")
                
        df_sorted = df[req_cols].copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        # 種牡馬ごとの時系列順
        df_sorted = df_sorted.sort_values(['sire_id', 'date'])
        
        # ターゲットと条件フラグ作成
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
        df_sorted['is_turf'] = (df_sorted['surface'] == '芝').astype(int)
        df_sorted['is_dirt'] = (df_sorted['surface'] == 'ダート').astype(int)
        
        # 条件付き勝利
        df_sorted['win_turf'] = df_sorted['is_win'] & df_sorted['is_turf']
        df_sorted['win_dirt'] = df_sorted['is_win'] & df_sorted['is_dirt']
        
        # GroupByで累積集計 (cumsum)
        # shift(1)で「今回のレースを含まない」過去の累積にする
        grp = df_sorted.groupby('sire_id')
        
        # 1. 出走回数
        df_sorted['sire_n_races'] = grp.cumcount() # cumcountは0 start (0, 1, 2...) -> shift不要でOK?
        # いや、cumcountの結果: 1走目は0. 2走目は1.
        # 特徴量として欲しいのは「過去N戦」。
        # 1走目の時、過去は0戦。=> cumcount()の値そのものが「過去の出走数」になる。
        # 例: [race1, race2, race3] -> cumcount=[0, 1, 2]
        # race1の時: 過去0戦 (OK)
        # race2の時: 過去1戦 (OK)
        # なので shift不要。cumcountそのものがshift相当。
        
        # 2. 勝利数など (cumsum) -> これはshift必要
        # race1: cumsum=1(if win). But we need 0 for race1 features.
        # So we need shift(1).
        # [Fix Leakage] cumsum().shift(1) must be within group.
        # Flat shift causes crossover leak.
        
        # [Check Leakage] Safe: cumsum includes current, shift(1) excludes current. transform keeps within group.
        df_sorted['sire_win_sum'] = grp['is_win'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        
        df_sorted['sire_turf_races'] = grp['is_turf'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df_sorted['sire_turf_wins'] = grp['win_turf'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        
        df_sorted['sire_dirt_races'] = grp['is_dirt'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df_sorted['sire_dirt_wins'] = grp['win_dirt'].transform(lambda x: x.cumsum().shift(1)).fillna(0)

        # For Heavy Track Calculation (Prepare here)
        # State column needed. Loader checks r.baba_jotai_code.
        # Ensure 'state' is available or derive. Loader guarantees 'state' column.
        if 'state' not in df_sorted.columns:
             df_sorted['state'] = 'Unknown'
        # 3. レート計算
        # 0除算注意
        def safe_div(a, b):
            return np.where(b > 0, a / b, 0.0)
            
        df_sorted['sire_win_rate'] = safe_div(df_sorted['sire_win_sum'], df_sorted['sire_n_races'])
        df_sorted['sire_turf_win_rate'] = safe_div(df_sorted['sire_turf_wins'], df_sorted['sire_turf_races'])
        df_sorted['sire_dirt_win_rate'] = safe_div(df_sorted['sire_dirt_wins'], df_sorted['sire_dirt_races'])
        
        # 補完 (一応)
        cols_to_fill = ['sire_n_races', 'sire_win_rate', 'sire_turf_win_rate', 'sire_dirt_win_rate']
        df_sorted[cols_to_fill] = df_sorted[cols_to_fill].fillna(0)
        
        # 4. Heavy Track Aptitude (重・不良)
        # state: '重' or '不良' OR codes '03', '04', 3, 4
        # Loader might deliver codes.
        heavy_codes = ['重', '不良', '03', '04', 3, 4, '3', '4']
        df_sorted['is_heavy'] = df_sorted['state'].isin(heavy_codes).astype(int)
        df_sorted['win_heavy'] = df_sorted['is_win'] & df_sorted['is_heavy']
        
        df_sorted['sire_heavy_races'] = grp['is_heavy'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df_sorted['sire_heavy_wins'] = grp['win_heavy'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        df_sorted['sire_heavy_win_rate'] = safe_div(df_sorted['sire_heavy_wins'], df_sorted['sire_heavy_races'])
        
        feats = ['sire_n_races', 'sire_win_rate', 'sire_turf_win_rate', 'sire_dirt_win_rate', 'sire_heavy_win_rate']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        return df_sorted[keys + feats].copy() 

    def _compute_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 時系列オッズ特徴量 (Odds Features)
        - odds_10min: 10分前単勝オッズ
        - 確定オッズ(odds)はリークになるため含めない
        """
        logger.info("ブロック計算中: odds_features")
        
        req_cols = ['race_id', 'horse_number', 'horse_id']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        df_feats = df[req_cols].copy()
        
        if 'odds_10min' in df.columns:
            df_feats['odds_10min'] = df['odds_10min']
        else:
            # 存在しない場合はNaN (LightGBMが処理)
            # 古いデータや取得失敗時は欠損
            df_feats['odds_10min'] = np.nan
            
        return df_feats 

    def _compute_training_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 調教特徴量 (Training Stats)
        """
        logger.info("ブロック計算中: training_stats")
        cols = [
            'training_course', 'training_time_4f', 'training_time_3f', 
            'training_time_last1f', 'training_level'
        ]
        keys = ['race_id', 'horse_number', 'horse_id']
        avail_cols = [c for c in cols if c in df.columns]
        
        return df[keys + avail_cols].copy()

    def _compute_burden_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 斤量特徴量 (Burden Stats)
        """
        logger.info("ブロック計算中: burden_stats")
        cols = ['impost', 'weight_diff', 'weight_ratio']
        keys = ['race_id', 'horse_number', 'horse_id']
        avail_cols = [c for c in cols if c in df.columns]
        
        return df[keys + avail_cols].copy()

    def _compute_corner_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] コーナーダイナミクス (Corner Dynamics)
        """
        logger.info("特徴量ブロックを計算中: corner_dynamics")
        from .features import corner_dynamics
        return corner_dynamics.compute_corner_dynamics(df)

    def _compute_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 対戦成績 (Head-to-Head)
        """
        logger.info("特徴量ブロックを計算中: head_to_head")
        from .features import head_to_head
        return head_to_head.compute_head_to_head(df)

    def _compute_training_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 調教詳細 (Training Detail)
        """
        logger.info("特徴量ブロックを計算中: training_detail")
        from .features import training_detail
        return training_detail.compute_training_detail(df)

    def _compute_change_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 変化特徴量 (Change Stats)
        - jockey_change: 乗り替わりフラグ
        - jockey_class_diff: 鞍上強化度合い
        - dist_change_category: 距離延長/短縮/同距離
        - interval_category: 出走間隔カテゴリ
        """
        logger.info("ブロック計算中: changes_stats")
        df_sorted = df.sort_values(['horse_id', 'date']).copy()
        grp = df_sorted.groupby('horse_id')
        
        # 1. Jockey Change
        df_sorted['prev_jockey_id'] = grp['jockey_id'].shift(1)
        df_sorted['jockey_change'] = (df_sorted['jockey_id'] != df_sorted['prev_jockey_id']).astype(int)
        
        # 2. Jockey Class Diff (Simple version: use expanding win rate)
        # Note: Detailed jockey stats might be better, but we compute a simple one here for self-containment
        df_tmp = df[['jockey_id', 'date', 'rank', 'race_id', 'horse_number']].copy()
        df_tmp['is_win'] = (df_tmp['rank'] == 1).astype(int)
        j_grp = df_tmp.sort_values(['jockey_id', 'date']).groupby('jockey_id')
        df_tmp['temp_j_win_rate'] = j_grp['is_win'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        
        # Map back to sorted df
        df_sorted = pd.merge(df_sorted, df_tmp[['race_id', 'horse_number', 'temp_j_win_rate']], on=['race_id', 'horse_number'], how='left')
        
        # Re-sort and re-group to ensure temp_j_win_rate is accessible in grp
        df_sorted = df_sorted.sort_values(['horse_id', 'date'])
        grp = df_sorted.groupby('horse_id')
        
        df_sorted['prev_j_win_rate'] = grp['temp_j_win_rate'].shift(1).fillna(0)
        df_sorted['jockey_class_diff'] = df_sorted['temp_j_win_rate'] - df_sorted['prev_j_win_rate']

        # 3. Distance Change
        df_sorted['prev_distance'] = grp['distance'].shift(1)
        def categorize_dist_change(row):
            curr = row['distance']
            prev = row['prev_distance']
            if pd.isna(prev): return 'Same'
            if curr > prev: return 'Lengthening'
            elif curr < prev: return 'Shortening'
            else: return 'Same'
        df_sorted['dist_change_category'] = df_sorted.apply(categorize_dist_change, axis=1)

        # 4. Interval Category
        df_sorted['prev_date'] = grp['date'].shift(1)
        df_sorted['interval'] = (pd.to_datetime(df_sorted['date']) - pd.to_datetime(df_sorted['prev_date'])).dt.days
        def categorize_interval(val):
            if pd.isna(val): return 'First'
            if val <= 7: return 'Rento'
            if val <= 30: return 'Short'
            if val <= 90: return 'Medium'
            if val <= 180: return 'Long'
            return 'Rest'
        df_sorted['interval_category'] = df_sorted['interval'].apply(categorize_interval)

        feats = ['jockey_change', 'jockey_class_diff', 'dist_change_category', 'interval_category']
        keys = ['race_id', 'horse_number', 'horse_id']
        return df_sorted[keys + feats].copy()

    def _compute_aptitude_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] 適性特徴量 (Aptitude Stats)
        - course_win_rate: 競馬場適性
        - dist_win_rate: 距離適性
        - surface_win_rate: 馬場適性
        """
        logger.info("ブロック計算中: aptitude_stats")
        df_sorted = df.sort_values(['horse_id', 'date']).copy()
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
        
        # 距離区分 (1200m, 1600m ...) の簡易ビン
        def bin_dist(d):
            if d <= 1400: return 'Sprint'
            if d <= 1800: return 'Mile'
            if d <= 2200: return 'Intermediate'
            if d <= 2800: return 'Long'
            return 'Stay'
        df_sorted['dist_bin'] = df_sorted['distance'].apply(bin_dist)

        # 競馬場別
        df_sorted['course_win_rate'] = df_sorted.groupby(['horse_id', 'venue'])['is_win'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        # 距離別
        df_sorted['dist_win_rate'] = df_sorted.groupby(['horse_id', 'dist_bin'])['is_win'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        # 馬場別
        df_sorted['surface_win_rate'] = df_sorted.groupby(['horse_id', 'surface'])['is_win'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)

        feats = ['course_win_rate', 'dist_win_rate', 'surface_win_rate']
        keys = ['race_id', 'horse_number', 'horse_id']
        return df_sorted[keys + feats].copy()

    def _compute_speed_index_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] スピード指数 (Speed Index)
        - 標準化した走破タイムをベースにする
        - [Fix Leakage] 全期間集計(Future Leak)を廃止し、Expanding(As-of)集計に変更
        """
        logger.info("ブロック計算中: speed_index_stats")
        df_sorted = df.copy()
        
        # 時系列順にソート (Expanding計算のため必須)
        if 'date' in df_sorted.columns:
            df_sorted['date'] = pd.to_datetime(df_sorted['date'])
            df_sorted = df_sorted.sort_values(['date', 'race_id'])
        
        # 条件キー: venue, distance, surface
        # [Refinement] User feedback: Limit keys to ensure stability. Drop track_condition.
        # "cond_keys を増やしすぎない（最初は venue×distance×surface で十分）"
        cond_keys = ['venue', 'distance', 'surface']
        
        logger.info(f"  Speed Index Grouping Keys: {cond_keys}")

        # [Changed]: Global Mean/Std (Leak) -> Expanding Mean/Std (Safe)
        # グループごとの累積統計を計算
        # shift(1)することで「当該レース以前」の統計を使う（自己除外＆未来除外）
        grp = df_sorted.groupby(cond_keys)
        
        # タイムの平均と標準偏差 (Expanding)
        # transformの結果はdf_sortedのindexと整合する
        expanding_mean = grp['time'].transform(lambda x: x.expanding().mean().shift(1))
        expanding_std = grp['time'].transform(lambda x: x.expanding().std().shift(1))
        expanding_count = grp['time'].transform(lambda x: x.expanding().count().shift(1))
        
        # スコア計算
        # 値が小さいほど速い -> (Mean - Time) / Std
        # StdがNaN(データ不足)または0の場合は計算不可 -> 0 (平均並み) とする
        # [Refinement] Handle std=0 explicitly
        expanding_std = expanding_std.replace(0, np.nan)
        
        speed_index = (expanding_mean - df_sorted['time']) / expanding_std
        
        # [Refinement] Fallback for insufficient samples
        # min_periods未満の場合は不安定なので0.0 (平均) に倒す
        min_samples = 30
        speed_index = speed_index.mask(expanding_count < min_samples, 0.0)
        
        # 欠損処理: 初出走や同条件の過去データがない場合は 0.0
        df_sorted['speed_index'] = speed_index.fillna(0.0)
        
        # 無限大などの異常値クリップ (念のため -10 ~ 10)
        df_sorted['speed_index'] = df_sorted['speed_index'].clip(-10, 10)

        # 過去走集計 (Horse ID単位)
        # ここも GroupBy + transform(shift...rolling) なので安全
        # 再度 horse_id, date でソートが必要（さっきは date, race_id でソートしたため順序が変わっている可能性）
        df_sorted = df_sorted.sort_values(['horse_id', 'date'])
        h_grp = df_sorted.groupby('horse_id')
        
        df_sorted['avg_speed_index'] = h_grp['speed_index'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
        df_sorted['max_speed_index'] = h_grp['speed_index'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()).fillna(0)

        # [New] Redefine Race Level: Previous Race's Rival Average Speed Index
        # 1. Calculate Race Average of avg_speed_index (Entering Ability)
        # Note: avg_speed_index is per horse entering the race.
        r_grp_si = df_sorted.groupby('race_id')
        df_sorted['race_avg_speed_index'] = r_grp_si['avg_speed_index'].transform('mean')
        
        # 2. Shift(1) to get Previous Race's Level for specific horse
        # Needs repeat sort to be safe
        df_sorted = df_sorted.sort_values(['horse_id', 'date'])
        h_side_grp = df_sorted.groupby('horse_id')
        
        df_sorted['lag1_rival_speed_mean'] = h_side_grp['race_avg_speed_index'].shift(1).fillna(50) # Default 50

        feats = ['avg_speed_index', 'max_speed_index', 'lag1_rival_speed_mean']
        keys = ['race_id', 'horse_number', 'horse_id']
        return df_sorted[keys + feats].copy()


    def _compute_temporal_jockey_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] M4-A: Temporal Jockey Stats
        - Rolling 180D/365D stats
        - Relative stats (Z-score vs Global Trend)
        """
        logger.info("ブロック計算中: temporal_jockey_stats")
        
        req_cols = ['jockey_id', 'date', 'rank']
        if not all(c in df.columns for c in req_cols):
             msg = f"Missing columns for temporal_jockey: {[c for c in req_cols if c not in df.columns]}"
             raise ValueError(msg)
             
        df_target = df.copy()
        if not np.issubdtype(df_target['date'].dtype, np.datetime64):
            df_target['date'] = pd.to_datetime(df_target['date'])
            
        # Target creation
        df_target['is_win'] = (df_target['rank'] == 1).astype(int)
        df_target['is_top3'] = (df_target['rank'] <= 3).astype(int)
        
        # 1. Rolling Stats (180D, 365D)
        # Using the safe/optimized utility
        rolled = temporal_stats.compute_rolling_stats(
            df_target,
            group_col='jockey_id',
            target_cols=['is_win', 'is_top3'],
            time_col='date',
            windows=['180D', '365D']
        )
        # rolled has columns like 'jockey_id_n_races_180d', 'jockey_id_is_win_sum_180d'
        
        # Calculate Rates
        for w in ['180d', '365d']:
            n_col = f"jockey_id_n_races_{w}"
            w_sum = f"jockey_id_is_win_sum_{w}"
            t_sum = f"jockey_id_is_top3_sum_{w}"
            
            w_rate = f"jockey_win_rate_{w}"
            t_rate = f"jockey_top3_rate_{w}"
            
            rolled[w_rate] = (rolled[w_sum] / rolled[n_col]).fillna(0)
            rolled[t_rate] = (rolled[t_sum] / rolled[n_col]).fillna(0)
            
            rolled.rename(columns={n_col: f"jockey_n_races_{w}"}, inplace=True)
            
        # 2. Relative Stats
        # Target: n_races_365d, win_rate_365d
        rel_targets = ['jockey_n_races_365d', 'jockey_win_rate_365d', 'jockey_top3_rate_365d']
        rel = temporal_stats.compute_relative_stats(
            rolled,
            target_cols=[c for c in rel_targets if c in rolled.columns],
            time_col='date',
            window=10000,
            use_fixed_baseline=True,
            baseline_cutoff_date='2024-12-31'
        )
        
        # Merge relative columns back to rolled
        new_rel_cols = [c for c in rel.columns if c not in rolled.columns]
        rolled[new_rel_cols] = rel[new_rel_cols]
        
        # 3. Momentum (180d - 365d)
        rolled['jockey_win_rate_momentum'] = rolled['jockey_win_rate_180d'] - rolled['jockey_win_rate_365d']
        rolled['jockey_top3_rate_momentum'] = rolled['jockey_top3_rate_180d'] - rolled['jockey_top3_rate_365d']
        
        # Select final columns
        final_feats = [c for c in rolled.columns if 'jockey_' in c and c not in df.columns]
        
        keys = ['race_id', 'horse_number', 'horse_id']
        return rolled[keys + final_feats].copy()

    def _compute_temporal_trainer_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] M4-A: Temporal Trainer Stats
        - New block for trainer (previously missing)
        """
        logger.info("ブロック計算中: temporal_trainer_stats")
        
        req_cols = ['trainer_id', 'date', 'rank']
        if not all(c in df.columns for c in req_cols):
             raise ValueError("Missing columns for trainer stats")
             
        df_target = df.copy()
        if not np.issubdtype(df_target['date'].dtype, np.datetime64):
            df_target['date'] = pd.to_datetime(df_target['date'])
            
        df_target['is_win'] = (df_target['rank'] == 1).astype(int)
        df_target['is_top3'] = (df_target['rank'] <= 3).astype(int)
        
        # 1. Rolling Stats
        rolled = temporal_stats.compute_rolling_stats(
            df_target,
            group_col='trainer_id',
            target_cols=['is_win', 'is_top3'],
            time_col='date',
            windows=['180D', '365D']
        )
        
        for w in ['180d', '365d']:
            n_col = f"trainer_id_n_races_{w}"
            w_sum = f"trainer_id_is_win_sum_{w}"
            t_sum = f"trainer_id_is_top3_sum_{w}"
            
            w_rate = f"trainer_win_rate_{w}"
            t_rate = f"trainer_top3_rate_{w}"
            
            rolled[w_rate] = (rolled[w_sum] / rolled[n_col]).fillna(0)
            rolled[t_rate] = (rolled[t_sum] / rolled[n_col]).fillna(0)
            rolled.rename(columns={n_col: f"trainer_n_races_{w}"}, inplace=True)
            
        # 2. Relative Stats
        rel_targets = ['trainer_n_races_365d', 'trainer_win_rate_365d', 'trainer_top3_rate_365d']
        rel = temporal_stats.compute_relative_stats(
            rolled,
            target_cols=[c for c in rel_targets if c in rolled.columns],
            time_col='date',
            window=10000,
            use_fixed_baseline=True,
            baseline_cutoff_date='2024-12-31'
        )
        
        new_rel_cols = [c for c in rel.columns if c not in rolled.columns]
        rolled[new_rel_cols] = rel[new_rel_cols]
        
        # 3. Momentum (180d - 365d)
        rolled['trainer_win_rate_momentum'] = rolled['trainer_win_rate_180d'] - rolled['trainer_win_rate_365d']
        rolled['trainer_top3_rate_momentum'] = rolled['trainer_top3_rate_180d'] - rolled['trainer_top3_rate_365d']

        final_feats = [c for c in rolled.columns if 'trainer_' in c and c not in df.columns]
        keys = ['race_id', 'horse_number', 'horse_id']
        
        return rolled[keys + final_feats].copy()

    def _compute_class_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] M4-B: Class Stats
        - Recent Class Experience & Trend
        """
        logger.info("ブロック計算中: class_stats")
        return class_stats.compute_class_stats(df)

    def _compute_segment_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block] M4-C: Segment Stats
        - Small Field & Metric Mile Reinforcement
        """
        logger.info("ブロック計算中: segment_stats")
        return segment_stats.compute_segment_stats(df)

    def create_block(self, block_name: str, func: Callable[[pd.DataFrame], pd.DataFrame], 
                     base_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """
        特徴量ブロックを作成する（キャッシュ対応）
        
        Args:
            block_name: 特徴量ブロックの一意な名前
            func: base_dfを受け取り特徴量DFを返す関数
            base_df: ベースとなるデータフレーム
            force: Trueならキャッシュを無視して再計算
        """
        cache_path = os.path.join(self.cache_dir, f"{block_name}.parquet")
        
        if not force and os.path.exists(cache_path):
            logger.debug(f"キャッシュされた特徴量ブロックを使用: {block_name}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"{block_name} のキャッシュ読み込み失敗: {e}. 再計算します。")
        
        logger.debug(f"特徴量ブロックを計算中: {block_name}")
        block_df = func(base_df)
        
        # キャッシュ保存
        logger.debug(f"ブロックをキャッシュに保存: {cache_path}")
        block_df.to_parquet(cache_path)
        
        return block_df

    def load_features(self, base_df: pd.DataFrame, block_names: List[str], force: bool = False) -> pd.DataFrame:
        """
        複数の特徴量ブロックをロードし、結合して返す (Disk-Based Merge to avoid OOM)
        依存関係のあるブロックのため、累積されたカラムを後続ブロックに渡す
        """
        # ベースのキー情報から開始 (horse_idを追加)
        key_cols = ['race_id', 'horse_number', 'horse_id']
        
        # キー確認
        for k in key_cols:
            if k not in base_df.columns:
                raise ValueError(f"Base DataFrameにキー列がありません: {k}")

        # Initialize current features with keys (Unique per PID to allow concurrency)
        temp_merge_path = os.path.join(self.cache_dir, f"temp_merge_{os.getpid()}.parquet")
        base_df[key_cols].to_parquet(temp_merge_path)
        
        logger.info(f"Initialized merge buffer at {temp_merge_path}")
        
        # ブロック依存マップ: 他ブロックの出力に依存するブロック
        # これらはキャッシュを無効化し、累積dfを使用する
        dependent_blocks = {
            'race_structure',      # needs: last_nige_rate from pace_stats
            'pace_pressure_stats', # needs: last_nige_rate from pace_stats
            'lap_fit',             # needs: avg_pci, pace_expectation_proxy
            'relative_stats',      # needs: speed_index from speed_index_stats
            'relative_expansion',  # needs: horse_elo from rating_elo
            'aptitude_smoothing',  # needs: jockey_n_races from jockey_stats
        }
        
        try:
            # Temporarily set logging level to WARNING to suppress "compute block" logs
            old_level = logger.level
            logger.setLevel(logging.WARNING)
            
            pbar = tqdm(block_names, desc="Feature Processing", unit="block")
            for name in pbar:
                pbar.set_postfix(block=name)
                gc.collect()
                if name not in self.registry:
                    raise ValueError(f"未知の特徴量ブロックです: {name}. 登録済み: {list(self.registry.keys())}")
                
                func = self.registry[name]
                try:
                    # Load or Compute Block
                    # logger.info(f"Preparing Block: {name}") # Suppressed as per user request
                    
                    # 依存ブロックの場合は累積dfを使用してキャッシュを無効化
                    if name in dependent_blocks:
                        # Load Current buffer to see what features we already have
                        current_df = pd.read_parquet(temp_merge_path)
                        # Get feature columns (those not in base_df)
                        # [Fix] Exclude columns that are already in base_df to avoid _x/_y duplicates
                        feature_cols = [c for c in current_df.columns if c not in base_df.columns and c not in key_cols]
                        
                        # Merge only features into base_df for the block function
                        accumulated_df = pd.merge(
                            base_df, 
                            current_df[key_cols + feature_cols],
                            on=key_cols, how='left'
                        )
                        # Use silent mode for create_block if possible or just let it log to debug
                        block_df = self.create_block(name, func, accumulated_df, force=True)
                    else:
                        block_df = self.create_block(name, func, base_df, force=force)
                    
                    # Load Current buffer
                    current_df = pd.read_parquet(temp_merge_path)
                    
                    # Identify new columns
                    cols_to_use = [c for c in block_df.columns if c not in current_df.columns]
                    # logger.debug(f"New cols for {name}: {cols_to_use}")

                    # Check keys
                    if not all(k in block_df.columns for k in key_cols):
                         raise ValueError(f"ブロック {name} にはマージ用のキー {key_cols} が必要です。")

                    # Merge into buffer by keys
                    # logger.info(f"Merging Block: {name} (Adding {len(cols_to_use)} cols)")
                    current_df = pd.merge(
                        current_df,
                        block_df[key_cols + cols_to_use],
                        on=key_cols, how='left'
                    )
                    
                    current_df.to_parquet(temp_merge_path)
                    
                except Exception as e:
                    logger.error(f"Error processing block {name}: {e}")
                    raise e

            # Restore logging level
            logger.setLevel(old_level)
            
            # Load Final result
            return pd.read_parquet(temp_merge_path)
            
        finally:
            if os.path.exists(temp_merge_path):
                try:
                    os.remove(temp_merge_path)
                    logger.info(f"Cleaned up merge buffer at {temp_merge_path}")
                except: pass

