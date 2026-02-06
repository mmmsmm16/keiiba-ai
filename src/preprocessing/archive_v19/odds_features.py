"""
Time-Series Odds Feature Engineering

Phase 16: 時系列オッズ特徴量を生成するモジュール。
T-30（発走30分前）とT-10（発走10分前）のオッズ差分から、
「大口投票（オッズ急落）」などのシグナルを検出する。
"""

import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OddsFeatureEngineer:
    """時系列オッズ特徴量を生成するクラス。"""
    
    def __init__(self, snapshot_dir: str = "data/odds_snapshots", odds_drop_threshold: float = 0.20):
        """
        Args:
            snapshot_dir: オッズスナップショットディレクトリ
            odds_drop_threshold: オッズ急落フラグの閾値（デフォルト: 20%下落）
        """
        self.snapshot_dir = snapshot_dir
        self.odds_drop_threshold = odds_drop_threshold
    
    def load_odds_snapshot(self, years, snapshot_type: str = "T-30") -> pd.DataFrame:
        """
        指定したスナップショットタイプのオッズをロード。
        
        Args:
            years: 年のリストまたはrange
            snapshot_type: "T-30", "T-10", "T-60" など
            
        Returns:
            pd.DataFrame: race_id, horse_number, odds_{type}, ninki_{type}
        """
        df_list = []
        for year in years:
            path = os.path.join(self.snapshot_dir, str(year), f"odds_{snapshot_type}.parquet")
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path, filters=[('ticket_type', '=', 'win')])
                    df = df[['race_id', 'combination', 'odds', 'ninki']].copy()
                    df['race_id'] = df['race_id'].astype(str)
                    df['horse_number'] = pd.to_numeric(df['combination'], errors='coerce')
                    
                    # リネーム
                    suffix = snapshot_type.replace('-', '_').lower()  # T-30 -> t_30
                    df.rename(columns={
                        'odds': f'odds_{suffix}',
                        'ninki': f'ninki_{suffix}'
                    }, inplace=True)
                    
                    df_list.append(df[['race_id', 'horse_number', f'odds_{suffix}', f'ninki_{suffix}']])
                    logger.debug(f"Loaded {snapshot_type} odds for {year}")
                except Exception as e:
                    logger.warning(f"Failed to load {snapshot_type} odds for {year}: {e}")
            else:
                logger.debug(f"{snapshot_type} odds not found for {year}: {path}")
                
        if not df_list:
            return pd.DataFrame()
        
        return pd.concat(df_list, ignore_index=True)
    
    def add_features(self, df: pd.DataFrame, t30_odds: pd.DataFrame, t10_odds: pd.DataFrame = None) -> pd.DataFrame:
        """
        時系列オッズ特徴量をメインDataFrameに追加。
        
        Args:
            df: メインのレースデータ（race_id, horse_number含む）
            t30_odds: T-30オッズデータ
            t10_odds: T-10オッズデータ（Noneの場合はdf内のoddsをT-10として使用）
            
        Returns:
            pd.DataFrame: 時系列オッズ特徴量が追加されたデータ
        """
        logger.info("時系列オッズ特徴量の生成を開始...")
        
        # キーの型を統一
        df['race_id'] = df['race_id'].astype(str)
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
        
        # 既存の時系列オッズカラムがあれば削除（再計算のため）
        existing_cols = ['odds_t_30', 'ninki_t_30', 'odds_t_10', 'ninki_t_10',
                         'odds_change_t30_t10', 'odds_diff_t30_t10', 'ninki_change_t30_t10',
                         'is_odds_drop', 'is_odds_surge', 'log_odds_t_30']
        for col in existing_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # T-30オッズをマージ
        if not t30_odds.empty:
            t30_odds = t30_odds.copy()
            t30_odds['race_id'] = t30_odds['race_id'].astype(str)
            t30_odds['horse_number'] = pd.to_numeric(t30_odds['horse_number'], errors='coerce')
            
            df = pd.merge(df, t30_odds, on=['race_id', 'horse_number'], how='left')
            n_matched = df['odds_t_30'].notna().sum() if 'odds_t_30' in df.columns else 0
            logger.info(f"T-30オッズをマージ: {n_matched:,} / {len(df):,} レコード")
        else:
            df['odds_t_30'] = np.nan
            df['ninki_t_30'] = np.nan
        
        # T-10オッズの取得（既存のoddsカラムまたは別途マージ）
        if t10_odds is not None and not t10_odds.empty:
            t10_odds = t10_odds.copy()
            t10_odds['race_id'] = t10_odds['race_id'].astype(str)
            t10_odds['horse_number'] = pd.to_numeric(t10_odds['horse_number'], errors='coerce')
            df = pd.merge(df, t10_odds, on=['race_id', 'horse_number'], how='left', suffixes=('', '_dup'))
            # 重複カラムがあれば削除
            df = df.loc[:, ~df.columns.duplicated()]
            n_matched = df['odds_t_10'].notna().sum() if 'odds_t_10' in df.columns else 0
            logger.info(f"T-10オッズをマージ: {n_matched:,} / {len(df):,} レコード")
            odds_t10_col = 'odds_t_10'
            ninki_t10_col = 'ninki_t_10'
        else:
            # 既存のoddsカラムをT-10として使用
            odds_t10_col = 'odds'
            ninki_t10_col = 'popularity'
        
        # ========================================
        # 時系列オッズ特徴量の計算
        # ========================================
        
        # 1. オッズ変化率: (T-30 - T-10) / T-30
        #    正の値 = オッズが下がった（買われた）
        df['odds_change_t30_t10'] = (df['odds_t_30'] - df[odds_t10_col]) / (df['odds_t_30'] + 1e-5)
        
        # 2. オッズ差（絶対値）
        df['odds_diff_t30_t10'] = df['odds_t_30'] - df[odds_t10_col]
        
        # 3. 人気順変動: T-30人気 - T-10人気
        #    正の値 = 人気が上がった（人気順位の数字が減った）
        df['ninki_change_t30_t10'] = df['ninki_t_30'] - df[ninki_t10_col]
        
        # 4. オッズ急落フラグ（大口投票シグナル）
        df['is_odds_drop'] = (df['odds_change_t30_t10'] >= self.odds_drop_threshold).astype(int)
        
        # 5. オッズ急騰フラグ（逆張りの可能性）
        df['is_odds_surge'] = (df['odds_change_t30_t10'] <= -self.odds_drop_threshold).astype(int)
        
        # 6. T-30オッズの対数（市場評価の初期値）
        df['log_odds_t_30'] = np.log1p(df['odds_t_30'].fillna(0))
        
        # NaN処理
        fill_cols = ['odds_change_t30_t10', 'odds_diff_t30_t10', 'ninki_change_t30_t10', 'log_odds_t_30']
        for col in fill_cols:
            df[col] = df[col].fillna(0)
        
        # 統計情報のログ
        n_drops = df['is_odds_drop'].sum()
        n_surges = df['is_odds_surge'].sum()
        logger.info(f"オッズ急落フラグ (>={self.odds_drop_threshold*100:.0f}%): {n_drops:,} 件")
        logger.info(f"オッズ急騰フラグ (<=-{self.odds_drop_threshold*100:.0f}%): {n_surges:,} 件")
        logger.info(f"オッズ変化率 平均: {df['odds_change_t30_t10'].mean():.4f}")
        
        logger.info("時系列オッズ特徴量の生成完了")
        return df


def create_odds_features(df: pd.DataFrame, 
                         years_range: range,
                         snapshot_dir: str = "data/odds_snapshots",
                         odds_drop_threshold: float = 0.20) -> pd.DataFrame:
    """
    便利関数: 時系列オッズ特徴量を一括で追加。
    
    Args:
        df: メインのレースデータ
        years_range: 対象年の範囲
        snapshot_dir: スナップショットディレクトリ
        odds_drop_threshold: 急落判定の閾値
        
    Returns:
        時系列オッズ特徴量が追加されたDataFrame
    """
    engineer = OddsFeatureEngineer(snapshot_dir, odds_drop_threshold)
    
    t30_odds = engineer.load_odds_snapshot(years_range, "T-30")
    t10_odds = engineer.load_odds_snapshot(years_range, "T-10")
    
    return engineer.add_features(df, t30_odds, t10_odds)
