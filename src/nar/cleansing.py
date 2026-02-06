"""
cleansing.py - データクレンジングモジュール

[変更履歴]
- 2025-12-15 v11: A2対応 - 番兵値のNaN化・異常値処理追加
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataCleanser:
    """
    生データのクレンジングを行うクラス。
    
    [v11更新]
    - 番兵値（odds=0, weight=999等）の検出・変換処理を追加
    - 異常値バリデーションログを追加
    """
    
    def cleanse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームのクレンジングを行います。

        Args:
            df (pd.DataFrame): RawDataLoaderから取得した生データ。

        Returns:
            pd.DataFrame: クレンジング済みのデータ。
        """
        logger.info("データクレンジングを開始...")
        original_len = len(df)

        # 1. 順位(rank)がないデータ（取消、中止など）を削除
        # [v11] rank=0 も削除対象に追加
        df = df.dropna(subset=['rank'])
        df = df[df['rank'] != 0]  # A2: rank=0は無効データ
        df['rank'] = df['rank'].astype(int)

        # 2. [v11 A2] 番兵値の処理
        df = self._handle_sentinel_values(df)

        # 3. 欠損値の処理
        # 体重(weight)の欠損: 既に_handle_sentinel_valuesで処理済み
        # ただし追加の保険として残す
        if 'weight' in df.columns and df['weight'].isnull().any():
            median_weight = df['weight'].median()
            if pd.isna(median_weight):
                median_weight = 470  # デフォルト
            df['weight'] = df['weight'].fillna(median_weight)

        # 体重増減(weight_diff): 0で埋める
        if 'weight_diff' in df.columns:
            df['weight_diff'] = df['weight_diff'].fillna(0)

        # 4. 型変換
        # 日付をdatetime型に
        df['date'] = pd.to_datetime(df['date'])

        # 5. [v11] バリデーションログ
        self._log_validation_stats(df)

        logger.info(f"クレンジング完了: {original_len} -> {len(df)} 件 (削除: {original_len - len(df)})")
        return df
    
    def _handle_sentinel_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [v11 A2] 番兵値（ありえない値）をNaN化または適切な値に変換する。
        
        対象:
        - odds=0 → NaN化
        - weight=0 or weight>=999 → 中央値で補完
        - weight_diff <= -99 or >= 999 → 0
        - impost=0 → 平均値で補完
        - frame_number=0, horse_number=0 → NaN化（後で除外される可能性）
        """
        logger.info("番兵値の検出・変換中...")
        stats = {}
        
        # odds=0 → NaN
        if 'odds' in df.columns:
            mask = df['odds'] == 0
            count = mask.sum()
            if count > 0:
                df.loc[mask, 'odds'] = np.nan
                stats['odds=0'] = count
        
        # weight 異常値
        if 'weight' in df.columns:
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            mask = (df['weight'] == 0) | (df['weight'] >= 999) | (df['weight'] < 300)
            count = mask.sum()
            if count > 0:
                # 異常値以外の中央値で補完
                median_val = df.loc[~mask, 'weight'].median()
                if pd.isna(median_val):
                    median_val = 470
                df.loc[mask, 'weight'] = median_val
                stats['weight異常値'] = count
        
        # weight_diff 異常値
        if 'weight_diff' in df.columns:
            df['weight_diff'] = pd.to_numeric(df['weight_diff'], errors='coerce')
            mask = (df['weight_diff'] <= -99) | (df['weight_diff'] >= 999)
            count = mask.sum()
            if count > 0:
                df.loc[mask, 'weight_diff'] = 0
                stats['weight_diff異常値'] = count
        
        # impost=0 → 平均値
        if 'impost' in df.columns:
            df['impost'] = pd.to_numeric(df['impost'], errors='coerce')
            mask = (df['impost'] == 0) | (df['impost'].isna())
            count = mask.sum()
            if count > 0:
                mean_val = df.loc[~mask, 'impost'].mean()
                if pd.isna(mean_val):
                    mean_val = 55.0
                df.loc[mask, 'impost'] = mean_val
                stats['impost異常値'] = count
        
        # [v11] last_3f=0 → NaN (上がり3Fが0は欠損)
        if 'last_3f' in df.columns:
            df['last_3f'] = pd.to_numeric(df['last_3f'], errors='coerce')
            mask = df['last_3f'] == 0
            count = mask.sum()
            if count > 0:
                df.loc[mask, 'last_3f'] = np.nan
                stats['last_3f=0'] = count
        
        # frame_number=0, horse_number=0 → 警告のみ（削除はしない）
        for col in ['frame_number', 'horse_number']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mask = df[col] == 0
                count = mask.sum()
                if count > 0:
                    # 削除ではなくNaN化（後続処理で扱う）
                    df.loc[mask, col] = np.nan
                    stats[f'{col}=0'] = count
        
        if stats:
            logger.info(f"番兵値を変換しました: {stats}")
        else:
            logger.info("番兵値: 変換対象なし")
            
        return df
    
    def _log_validation_stats(self, df: pd.DataFrame) -> None:
        """
        [v11] クレンジング後のデータ統計をログ出力する。
        """
        # 主要カラムのNaN率
        key_cols = ['odds', 'weight', 'weight_diff', 'impost', 'frame_number', 'horse_number']
        nan_stats = {}
        for col in key_cols:
            if col in df.columns:
                nan_rate = df[col].isna().mean()
                if nan_rate > 0:
                    nan_stats[col] = f"{nan_rate:.2%}"
        
        if nan_stats:
            logger.info(f"クレンジング後のNaN率: {nan_stats}")
        
        # 数値カラムの基本統計（異常値チェック用）
        if 'weight' in df.columns:
            w_min, w_max = df['weight'].min(), df['weight'].max()
            if w_min < 350 or w_max > 600:
                logger.warning(f"weight範囲が異常: {w_min} - {w_max}")
        
        if 'impost' in df.columns:
            i_min, i_max = df['impost'].min(), df['impost'].max()
            if i_min < 40 or i_max > 70:
                logger.warning(f"impost範囲が異常: {i_min} - {i_max}")
