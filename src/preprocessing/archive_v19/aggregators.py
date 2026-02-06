"""
aggregators.py - 過去走特徴量生成モジュール

[変更履歴]
- 2025-12-15 v11: A1対応 - rank系の0埋め廃止 → ニュートラル補完(8.0) + 欠損フラグ追加
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# v11: ニュートラル値定義（16頭立て中位 = 8.0）
RANK_NEUTRAL_VALUE = 8.0
LAST_3F_NEUTRAL_VALUE = 35.0  # 上がり3Fの平均的な値
TIME_DIFF_NEUTRAL_VALUE = 1.0 # タイム差の平均的な値 (v16.1)


class HistoryAggregator:
    """
    馬の過去走情報（ラグ特徴量）を生成するクラス。
    
    [v11更新]
    - rank系の0埋め廃止 → 中央値（8.0）で補完
    - 欠損フラグ（*_is_missing）を追加
    """
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        過去走データを集計して特徴量を追加します。

        Args:
            df (pd.DataFrame): 前処理済みの全データ。

        Returns:
            pd.DataFrame: 過去走特徴量が追加されたデータ。
        """
        logger.info("過去走特徴量の生成を開始...")

        # 処理のために馬IDと日付でソート（必須）
        df = df.sort_values(['horse_id', 'date'])

        # 特徴量生成対象のカラム
        target_cols = ['rank', 'last_3f', 'odds', 'popularity', 'time_diff']

        # 1. 前走（Lag 1）の特徴量
        # 単純に1つずらす
        logger.info("前走データの生成中...")
        grouped = df.groupby('horse_id')
        for col in target_cols:
            if col in df.columns:
                df[f'lag1_{col}'] = grouped[col].shift(1)

        # 2. 近5走の平均（Rolling Mean）
        # 現在のレースを含まないように、shift(1)した上でrollingする
        logger.info("近5走平均データの生成中...")
        for col in ['rank', 'last_3f']:
            if col in df.columns:
                df[f'mean_{col}_5'] = grouped[col].transform(lambda x: x.shift(1).rolling(5).mean())
        
        # [v16.1] time_diffは min_periods=1 で計算（1走でもあれば平均を採用、欠損値は無視）
        if 'time_diff' in df.columns:
            df['mean_time_diff_5'] = grouped['time_diff'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # ----------------------------------------------------------------
        # [v11 A1] rank系の欠損フラグ生成 + ニュートラル補完
        # ----------------------------------------------------------------
        logger.info("v11: rank系ニュートラル補完を適用中...")
        
        # 欠損フラグを先に生成
        rank_cols = ['lag1_rank', 'mean_rank_5', 'mean_rank_all']
        for col in rank_cols:
            if col in df.columns:
                flag_col = f'{col}_is_missing'
                df[flag_col] = df[col].isna().astype('int8')
        
        # mean_last_3f_5 の欠損フラグ
        if 'mean_last_3f_5' in df.columns:
            df['mean_last_3f_5_is_missing'] = df['mean_last_3f_5'].isna().astype('int8')
            
        # [v16.1] time_diff系の欠損フラグ
        for col in ['lag1_time_diff', 'mean_time_diff_5']:
            if col in df.columns:
                df[f'{col}_is_missing'] = df[col].isna().astype('int8')

        # 3. コース適性（同馬場状態・同距離区分での過去成績）
        # Expanding mean (累積平均) を使用
        logger.info("通算成績データの生成中...")
        df['total_races'] = grouped['race_id'].cumcount() # 過去出走数

        # 過去の平均着順 (Expanding Mean)
        df['mean_rank_all'] = grouped['rank'].transform(lambda x: x.shift(1).expanding().mean())
        
        # mean_rank_all の欠損フラグを生成（上で定義済みの変数より後に生成されるため）
        df['mean_rank_all_is_missing'] = df['mean_rank_all'].isna().astype('int8')

        # 過去の勝利数
        df['wins_all'] = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum())

        # 勝率
        df['win_rate_all'] = df['wins_all'] / df['total_races']
        df['win_rate_all'] = df['win_rate_all'].fillna(0)

        # 獲得賞金 (Total Prize)
        if 'honshokin' in df.columns:
            df['honshokin_filled'] = pd.to_numeric(df['honshokin'], errors='coerce').fillna(0)
            df['total_prize'] = grouped['honshokin_filled'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            df.drop(columns=['honshokin_filled'], inplace=True)
        else:
            df['total_prize'] = 0

        # ----------------------------------------------------------------
        # [v11 A1] 欠損補完（0ではなくニュートラル値を使用）
        # ----------------------------------------------------------------
        # 生rank派生: 動的ニュートラル = (n_horses + 1) / 2
        # 正規化派生: ニュートラル = 0.5
        
        # n_horses がない場合は計算
        if 'n_horses' not in df.columns:
            df['n_horses'] = df.groupby('race_id')['horse_number'].transform('count')
        
        # 動的ニュートラル値（レースごとに異なる）
        dynamic_neutral = (df['n_horses'] + 1) / 2
        
        # rank系の補完（生rank派生）
        for col in ['lag1_rank', 'mean_rank_5', 'mean_rank_all']:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    # 動的ニュートラル値で補完
                    df[col] = df[col].fillna(dynamic_neutral).astype('float32')
                    logger.debug(f"{col}: {nan_count}件を動的ニュートラル値で補完")
        
        # last_3f系の補完（固定ニュートラル値）
        if 'mean_last_3f_5' in df.columns:
            df['mean_last_3f_5'] = df['mean_last_3f_5'].fillna(LAST_3F_NEUTRAL_VALUE).astype('float32')
        
        # lag1_last_3f は平均的な値で補完（存在する場合）
        if 'lag1_last_3f' in df.columns:
            df['lag1_last_3f_is_missing'] = df['lag1_last_3f'].isna().astype('int8')
            df['lag1_last_3f'] = df['lag1_last_3f'].fillna(LAST_3F_NEUTRAL_VALUE).astype('float32')

        # [v16.1] time_diff系は補完しない (NaNのままモデルに入力)
        # 理由: 競走中止などを安易な値で埋めるとリスク評価を誤るため。
        
        # その他のlag特徴（odds, popularity）は0埋めのまま（市場情報なので0=情報なしで適切）
        if 'lag1_odds' in df.columns:
            df['lag1_odds'] = df['lag1_odds'].fillna(0)
        if 'lag1_popularity' in df.columns:
            df['lag1_popularity'] = df['lag1_popularity'].fillna(0)
        
        # wins_all は0埋め（新馬は0勝で正しい）
        df['wins_all'] = df['wins_all'].fillna(0)

        # ================================================================
        # [v11 Extended] rank_norm: 正規化された着順（中間生成物）
        # ================================================================
        # rank_norm = 1 - (rank - 1) / (n_horses - 1)
        # 注: これは中間生成物であり、最終特徴量には含めない（リーク）
        logger.info("v11: rank_norm（正規化着順）を生成中...")
        
        df['rank_norm'] = 1 - (df['rank'] - 1) / (df['n_horses'] - 1).clip(lower=1)
        
        # shift済み派生のみ特徴量化
        df['lag1_rank_norm'] = df.groupby('horse_id')['rank_norm'].shift(1)
        df['mean_rank_norm_5'] = df.groupby('horse_id')['rank_norm'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        # 欠損フラグ + ニュートラル補完（正規化なので0.5）
        df['lag1_rank_norm_is_missing'] = df['lag1_rank_norm'].isna().astype('int8')
        df['lag1_rank_norm'] = df['lag1_rank_norm'].fillna(0.5).astype('float32')
        
        df['mean_rank_norm_5_is_missing'] = df['mean_rank_norm_5'].isna().astype('int8')
        df['mean_rank_norm_5'] = df['mean_rank_norm_5'].fillna(0.5).astype('float32')
        
        # 中間生成物の削除（重要！リーク防止）
        logger.info("v11: rank_norm（中間生成物）を削除")
        df.drop(columns=['rank_norm'], inplace=True)

        # ----------------------------------------------------------------
        # 検証ログ
        # ----------------------------------------------------------------
        self._log_fill_stats(df)

        logger.info("過去走特徴量の生成完了")
        return df
    
    def _log_fill_stats(self, df: pd.DataFrame) -> None:
        """
        [v11] 補完後の統計をログ出力する。
        """
        # 欠損フラグの合計をログ
        flag_cols = [c for c in df.columns if c.endswith('_is_missing')]
        if flag_cols:
            flag_stats = {c: df[c].sum() for c in flag_cols if df[c].sum() > 0}
            if flag_stats:
                logger.info(f"欠損フラグ統計: {flag_stats}")
        
        # rank系が0に偏っていないことを確認
        for col in ['mean_rank_5', 'lag1_rank', 'mean_rank_all']:
            if col in df.columns:
                zero_rate = (df[col] == 0).mean()
                if zero_rate > 0.01:
                    logger.warning(f"警告: {col} の0率が {zero_rate:.1%} です（ニュートラル補完が適用されていない可能性）")

