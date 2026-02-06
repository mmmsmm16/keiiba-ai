"""
pace_features.py - ペース分析特徴量生成モジュール

[Phase 17] Pace Analysis (Re-implementation)
レース展開（ペース）を分析し、各馬の展開適性を特徴量化する。
未来情報のリークを厳格に排除するため、当該レースの指標は中間計算のみに使用し、
モデル入力には「過去の傾向」と「出走メンバーから予測される展開」のみを使用する。
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PaceFeatureEngineer:
    """
    ペース分析特徴量生成クラス
    
    Features:
    1. PCI (Pace Change Index): 個人のペース配分 (前半速度/後半速度)
    2. RPCI (Race PCI): レース全体のペース (上位馬のPCI平均)
    3. ERPCI (Expected Race PCI): 今回のメンバーから予測される想定ペース
    4. Mismatch: 各馬の適性と想定ペースの乖離
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ペース関連特徴量を追加する。
        
        Args:
            df: 前処理済みDataFrame (must have: race_id, horse_id, date, raw_time, last_3f, distance)
            
        Returns:
            df: 特徴量追加後のDataFrame
        """
        logger.info("ペース分析特徴量の生成を開始...")
        
        # 必要なカラムのチェック
        required_cols = ['race_id', 'horse_id', 'date', 'raw_time', 'last_3f', 'distance', 'rank']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"必須カラムが不足しているため、Pace Featuresをスキップします: {missing}")
            return df

        # データ型変換とソート
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'race_id', 'horse_number']).reset_index(drop=True)
        
        # ----------------------------------------------------------------
        # 1. Raw PCI / RPCI 計算 (中間生成物)
        # ----------------------------------------------------------------
        logger.info("  1. Raw PCI/RPCI 計算...")
        
        # 数値変換と基本的バリデーション
        # raw_time, last_3f, distance は数値型
        df['raw_time'] = pd.to_numeric(df['raw_time'], errors='coerce')
        df['last_3f'] = pd.to_numeric(df['last_3f'], errors='coerce')
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        
        # PCI 計算
        # PCI = (Avg Speed First) / (Avg Speed Last)
        # Avg Speed Last = 600 / last_3f (m/s)
        # Avg Speed First = (distance - 600) / (raw_time - last_3f) (m/s)
        
        # 分母0や異常値対策
        valid_mask = (
            (df['distance'] > 600) & 
            (df['last_3f'] > 0) & 
            (df['raw_time'] > df['last_3f'])
        )
        
        # Pre-allocate
        df['raw_pci'] = np.nan
        
        # Vectorized calculation
        dist_first = df.loc[valid_mask, 'distance'] - 600
        time_first = df.loc[valid_mask, 'raw_time'] - df.loc[valid_mask, 'last_3f']
        speed_first = dist_first / time_first
        
        speed_last = 600 / df.loc[valid_mask, 'last_3f']
        
        df.loc[valid_mask, 'raw_pci'] = speed_first / speed_last
        
        # RPCI 計算 (Race Pace Change Index)
        # そのレースの上位3頭(rank <= 3)のPCI平均
        # rankが欠損している場合(除外等)は計算対象外
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        
        # 上位3頭のPCIを抽出
        top3_mask = (df['rank'] <= 3) & (df['raw_pci'].notna())
        
        # レースごとの平均を計算
        # transformにより全行に値を割り当て
        rpci_series = df[top3_mask].groupby('race_id')['raw_pci'].transform('mean')
        
        # groupby('race_id')で結合 (上位3頭がいないレースはNaN)
        # 上記のtransformはtop3_maskの行にしか値がないため、一度DataFrameにしてマージする
        rpci_df = df.loc[top3_mask, ['race_id', 'raw_pci']].groupby('race_id')['raw_pci'].mean().reset_index()
        rpci_df.columns = ['race_id', 'raw_rpci']
        
        # マージ
        df = pd.merge(df, rpci_df, on='race_id', how='left')
        
        # ----------------------------------------------------------------
        # 2. 過去特徴量の生成 (Horse History)
        # ----------------------------------------------------------------
        logger.info("  2. 過去傾向(History)の集計...")
        
        # 時系列ソート (horse_id, date)
        df = df.sort_values(['horse_id', 'date'])
        
        # GroupBy object
        grouped = df.groupby('horse_id')
        
        # lag1_pci (前走PCI)
        df['lag1_pci'] = grouped['raw_pci'].shift(1).astype('float32')
        
        # mean_pci_5 (近5走平均)
        # min_periods=1: 1走でもあれば計算する
        df['mean_pci_5'] = grouped['raw_pci'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).astype('float32')
        
        # pci_trend (近3走 - 近5走)
        df['mean_pci_3'] = grouped['raw_pci'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        ).astype('float32')
        df['pci_trend'] = (df['mean_pci_3'] - df['mean_pci_5']).astype('float32')
        
        # RPCI履歴
        df['lag1_rpci'] = grouped['raw_rpci'].shift(1).astype('float32')
        df['mean_rpci_5'] = grouped['raw_rpci'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).astype('float32')
        
        # 不要な一時列削除
        del df['mean_pci_3']
        
        # ----------------------------------------------------------------
        # 3. 展開予測 (ERPCI) とミスマッチ
        # ----------------------------------------------------------------
        logger.info("  3. 展開予測(ERPCI)とミスマッチ計算...")
        
        # レースごとの出走メンバーの mean_pci_5 を集計
        # これが「今回のレースの想定ペース」になる
        
        # groupby object (race level)
        # mean_pci_5 がNaNの馬もいるので考慮
        race_grouped = df.groupby('race_id')
        
        # erpci_mean: 平均的なペース嗜好
        # transformで各行に割り当て
        df['erpci_mean'] = race_grouped['mean_pci_5'].transform('mean').astype('float32')
        
        # erpci_std: メンバーの脚質のバラつき
        df['erpci_std'] = race_grouped['mean_pci_5'].transform('std').fillna(0).astype('float32')
        
        # pace_mismatch: 自分の適性とレース想定の乖離
        # +: 自分はハイペース型だがレースはスロー想定（合いにくい？）
        # -: 自分はスロー型だがレースはハイ想定（ついていけない？）
        # 絶対値の方が使いやすいかも
        df['pace_mismatch'] = (df['mean_pci_5'] - df['erpci_mean']).astype('float32')
        df['abs_pace_mismatch'] = df['pace_mismatch'].abs().astype('float32')
        
        # ----------------------------------------------------------------
        # 4. クリーンアップ (リーク防止)
        # ----------------------------------------------------------------
        # 生の raw_pci, raw_rpci は当該レースの結果なので削除必須
        drop_cols = ['raw_pci', 'raw_rpci']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        logger.info("ペース分析特徴量の生成完了")
        return df

    def get_feature_names(self):
        return [
            'lag1_pci', 'mean_pci_5', 'pci_trend',
            'lag1_rpci', 'mean_rpci_5',
            'erpci_mean', 'erpci_std',
            'pace_mismatch', 'abs_pace_mismatch'
        ]
