"""
realtime_features.py - リアルタイム特徴量生成モジュール

当日のレース結果から計算されるトラックバイアス指標（trend_*）を生成する。

[変更履歴]
- 2025-12-15 v11: A4対応 - use_realtimeフラグで事前予測/逐次更新モード切替
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# デフォルト値（事前予測モード用 / 第1レース用）
REALTIME_DEFAULTS = {
    'trend_win_inner_rate': 0.25,
    'trend_win_mid_rate': 0.50,
    'trend_win_outer_rate': 0.25,
    'trend_win_front_rate': 0.20,
    'trend_win_fav_rate': 0.33,
}


class RealTimeFeatureEngineer:
    """
    当日のリアルタイム傾向（トラックバイアス）を特徴量化するクラス。
    各レースについて、その日の「それまでのレース」の結果を集計して特徴量とします。
    
    [v11更新]
    - use_realtimeフラグで運用モード切替
      - True: 逐次更新モード（trend_*を計算）
      - False: 事前予測モード（trend_*をニュートラル値で埋める）
    - デフォルトは False（事前予測モード）
    """
    
    def add_features(self, df: pd.DataFrame, use_realtime: bool = False) -> pd.DataFrame:
        """
        リアルタイム特徴量を追加します。
        
        Args:
            df: 入力DataFrame
            use_realtime: True=逐次更新モード（trend_*を計算）、
                         False=事前予測モード（ニュートラル値で埋める）
                         
        Returns:
            trend_*特徴量が追加されたDataFrame
        """
        # ----------------------------------------------------------------
        # [v11 A4] 事前予測モード: ニュートラル値で埋めるだけ
        # ----------------------------------------------------------------
        if not use_realtime:
            logger.info("リアルタイム特徴量: 事前予測モード（trend_*はニュートラル値で埋めます）")
            for col, val in REALTIME_DEFAULTS.items():
                df[col] = val
            return df
        
        # ----------------------------------------------------------------
        # 逐次更新モード: 当日の傾向を計算
        # ----------------------------------------------------------------
        logger.info("リアルタイム特徴量（逐次更新モード）: 当日の傾向を生成中...")
        
        # 必要なカラムの確認
        required = ['date', 'venue', 'race_number', 'frame_number', 'rank', 'passing_rank', 'popularity']
        for col in required:
            if col not in df.columns:
                logger.error(f"必須カラムがありません: {col}")
                # 欠損の場合はデフォルト値で埋める
                for c, v in REALTIME_DEFAULTS.items():
                    df[c] = v
                return df

        if 'surface' not in df.columns:
            logger.warning("surfaceカラムがないため、コース別のバイアス集計ができません。全体集計を行います。")
            df['surface'] = 'All'

        df = df.sort_values(['date', 'venue', 'race_number']).copy()
        
        # グループ化: 日付 + 開催地 + サーフェス (芝/ダート別)
        
        # 1. レース単位の「勝った枠」「逃げ切りか」「1番人気が勝ったか」を判定
        winners = df[df['rank'] == 1].copy()
        
        # 枠番定義 (Inner: 1-2, Mid: 3-6, Outer: 7-8)
        winners['win_inner'] = winners['frame_number'].isin([1, 2]).astype(int)
        winners['win_mid']   = winners['frame_number'].isin([3, 4, 5, 6]).astype(int)
        winners['win_outer'] = winners['frame_number'].isin([7, 8]).astype(int)

        # 逃げ(通過順位の最初が1)が勝ったか
        def is_nige(s):
            if not isinstance(s, str): return 0
            try:
                first_pos = s.split('-')[0]
                return 1 if first_pos == '1' else 0
            except: return 0
        winners['win_front'] = winners['passing_rank'].apply(is_nige)
        
        # 1番人気が勝ったか
        winners['win_fav'] = (winners['popularity'] == 1).astype(int)

        # レースIDごとの結果辞書
        race_results = winners.set_index(['date', 'venue', 'race_number'])[[
            'win_inner', 'win_mid', 'win_outer', 'win_front', 'win_fav'
        ]]
        
        # Join surface back to race_results
        race_meta = df[['date', 'venue', 'race_number', 'surface']].drop_duplicates()
        race_stats = pd.merge(race_meta, race_results, on=['date', 'venue', 'race_number'], how='left').fillna(0)
        
        # Sort
        race_stats = race_stats.sort_values(['date', 'venue', 'surface', 'race_number'])
        
        # 2. 累積和 (Cumsum) と 平均 (Running Average) の計算
        grouped = race_stats.groupby(['date', 'venue', 'surface'], group_keys=False)
        
        cols = ['win_inner', 'win_mid', 'win_outer', 'win_front', 'win_fav']
        
        for col in cols:
            # Shifted expanding mean (Today's trend so far)
            race_stats[f"trend_{col}_rate"] = grouped[col].apply(lambda x: x.shift(1).expanding().mean())
            
        # Select for merge
        trend_cols = [f"trend_{c}_rate" for c in cols]
        trend_df = race_stats[['date', 'venue', 'race_number'] + trend_cols].copy()

        # 3. 元のDataFrameに結合
        df_out = df.copy() 
        
        if trend_df.empty:
            logger.warning("Trend DF is empty. Creating empty trend columns.")
        else:
            df_out = pd.merge(df, trend_df, on=['date', 'venue', 'race_number'], how='left')
        
        # 欠損値埋め（第1レースなど）
        for col, val in REALTIME_DEFAULTS.items():
            if col not in df_out.columns:
                df_out[col] = val
            else:
                df_out[col] = df_out[col].fillna(val)
        
        logger.info(f"リアルタイム特徴量生成完了: {len(trend_cols)} features processed.")
        return df_out
