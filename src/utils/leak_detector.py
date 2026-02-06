
import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def check_data_leakage(df: pd.DataFrame, target_col: str = 'target', threshold: float = 0.95):
    """
    DataFrame内のデータリークの可能性をチェックします。
    
    Args:
        df: 特徴量とターゲットを含むDataFrame
        target_col: ターゲットカラム名
        threshold: 失敗とみなす相関係数の閾値
        
    Raises:
        ValueError: リークが検出された場合
    """
    logger.info("🛡️ データリーク検知を実行中...")
    
    if target_col not in df.columns:
        logger.warning(f"ターゲット列 '{target_col}' が見つかりません。相関チェックをスキップします。")
        return

    # 1. 高相関チェック
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # ターゲット自体は除外
        features = [c for c in numeric_cols if c != target_col]
        
        # 相関計算
        # データサイズが大きい場合はサンプリング
        sample_size = 100000
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
            
        corrs = df_sample[features].corrwith(df_sample[target_col]).abs()
        
        # 閾値チェック
        leaks = corrs[corrs >= threshold]
        
        if not leaks.empty:
            msg = f"❌ リークを検出しました! 相関係数 >= {threshold} の特徴量:\n{leaks}"
            logger.error(msg)
            raise ValueError(msg)
            
    # 2. 禁止カラムチェック
    # レース前に知る由もないカラムが含まれていないかチェック
    forbidden_names = ['rank', 'finishing_position', 'time_seconds', 'payout']
    
    found_forbidden = [c for c in df.columns if any(bad in c.lower() for bad in forbidden_names) and c != target_col]
    
    if found_forbidden:
         logger.warning(f"⚠️  禁止ワードを含むカラムが見つかりました: {found_forbidden}. モデル入力(X)に含まれていないことを確認してください。")

    logger.info("✅ リーク検知を通過しました。")
