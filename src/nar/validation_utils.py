"""
validation_utils.py - Feature Engineering v11

番兵値検知・バリデーション・欠損フラグ生成のユーティリティモジュール。

[変更履歴]
- 2025-12-15: v11新規作成 - A2対応（番兵値のNaN化とバリデーション）
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 定数定義
# ============================================================================

# rank系欠損補完のニュートラル値（16頭立て中位 = 8.0）
RANK_NEUTRAL_VALUE = 8.0

# 番兵値として扱う値の定義
SENTINEL_RULES = {
    'odds': {'invalid': [0, 0.0], 'action': 'nan', 'default': None},
    'frame_number': {'invalid': [0], 'action': 'nan', 'default': None},
    'horse_number': {'invalid': [0], 'action': 'nan', 'default': None},
    'race_number': {'invalid': [0], 'action': 'nan', 'default': None},
    'weight': {'invalid_range': (0, 999), 'action': 'median', 'default': 470},
    'weight_diff': {'invalid_range': (-99, 999), 'action': 'zero', 'default': 0},
    'impost': {'invalid': [0], 'action': 'mean', 'default': 55},
    'time': {'invalid': [0], 'action': 'nan', 'default': None},
    'rank': {'invalid': [0], 'action': 'drop', 'default': None},  # A2: rank=0は削除（cleansing.pyで既に処理済み）
    'last_3f': {'invalid': [0, 0.0], 'action': 'nan', 'default': None},  # v11: last_3f=0はNaN化
}

# trend_*のデフォルト値（事前予測モード用）
REALTIME_DEFAULTS = {
    'trend_win_inner_rate': 0.25,
    'trend_win_mid_rate': 0.50,
    'trend_win_outer_rate': 0.25,
    'trend_win_front_rate': 0.20,
    'trend_win_fav_rate': 0.33,
}

# [v11 Extended V6] 禁止列（特徴量に含めてはいけない結果由来列）
FORBIDDEN_COLUMNS = [
    # 当該レースの結果
    'rank', 'time', 'rank_norm', 'passing_rank', 'last_3f',
    # 中間生成物（shift前）
    'time_index', 'last_3f_index',
    # 賞金（当該レースの結果）
    'honshokin', 'prize',
    # 内部ID/一時列
    'rank_str', 'raw_time', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
    'weight_diff_val', 'weight_diff_sign',
]


# ============================================================================
# 番兵値処理関数
# ============================================================================

def handle_sentinel_values(df: pd.DataFrame, log_stats: bool = True) -> pd.DataFrame:
    """
    番兵値（ありえない値）をNaN化または適切なデフォルト値に変換する。
    
    [A2対応] odds=0, weight=999 などを検出して処理
    
    Args:
        df: 入力DataFrame
        log_stats: True の場合、変換した件数をログ出力
        
    Returns:
        処理後のDataFrame
    """
    logger.info("番兵値の検出・変換を開始...")
    
    stats = {}
    
    for col, rule in SENTINEL_RULES.items():
        if col not in df.columns:
            continue
            
        original_nan_count = df[col].isna().sum()
        
        # 無効値リストによる検出
        if 'invalid' in rule:
            mask = df[col].isin(rule['invalid'])
            count = mask.sum()
            if count > 0:
                if rule['action'] == 'nan':
                    df.loc[mask, col] = np.nan
                elif rule['action'] == 'zero':
                    df.loc[mask, col] = 0
                elif rule['action'] == 'mean':
                    mean_val = df.loc[~mask, col].mean() if (~mask).any() else rule['default']
                    df.loc[mask, col] = mean_val
                elif rule['action'] == 'median':
                    median_val = df.loc[~mask, col].median() if (~mask).any() else rule['default']
                    df.loc[mask, col] = median_val
                stats[col] = count
                    
        # 無効範囲による検出
        if 'invalid_range' in rule:
            low, high = rule['invalid_range']
            mask = (df[col] <= low) | (df[col] >= high)
            count = mask.sum()
            if count > 0:
                if rule['action'] == 'nan':
                    df.loc[mask, col] = np.nan
                elif rule['action'] == 'zero':
                    df.loc[mask, col] = 0
                elif rule['action'] == 'mean':
                    mean_val = df.loc[~mask, col].mean() if (~mask).any() else rule['default']
                    df.loc[mask, col] = mean_val
                elif rule['action'] == 'median':
                    median_val = df.loc[~mask, col].median() if (~mask).any() else rule['default']
                    df.loc[mask, col] = median_val
                stats[col] = count
    
    if log_stats and stats:
        logger.info(f"番兵値の変換完了: {stats}")
    elif log_stats:
        logger.info("番兵値: 変換対象なし")
        
    return df


# ============================================================================
# 欠損フラグ生成関数
# ============================================================================

def generate_missing_flags(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    指定カラムの欠損フラグ（*_is_missing）を生成する。
    
    [A1対応] rank系の欠損を追跡するためのフラグ
    
    Args:
        df: 入力DataFrame
        columns: 欠損フラグを生成するカラム名リスト
        
    Returns:
        欠損フラグが追加されたDataFrame
    """
    for col in columns:
        if col in df.columns:
            flag_col = f"{col}_is_missing"
            df[flag_col] = df[col].isna().astype('int8')
            logger.debug(f"{flag_col} を生成: 欠損={df[flag_col].sum()}件")
    
    return df


def fill_with_neutral(df: pd.DataFrame, columns: list, neutral_value: float = RANK_NEUTRAL_VALUE) -> pd.DataFrame:
    """
    指定カラムの欠損値をニュートラル値で補完する。
    
    [A1対応] rank系は0ではなく中央値（8.0）で補完
    
    Args:
        df: 入力DataFrame
        columns: 補完対象カラム名リスト
        neutral_value: 補完値（デフォルト: RANK_NEUTRAL_VALUE = 8.0）
        
    Returns:
        補完後のDataFrame
    """
    for col in columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df[col] = df[col].fillna(neutral_value)
                logger.debug(f"{col}: {nan_count}件をニュートラル値({neutral_value})で補完")
    
    return df


# ============================================================================
# バリデーション関数
# ============================================================================

def validate_data_quality(df: pd.DataFrame, stage: str = "unknown") -> dict:
    """
    データ品質チェックを行い、警告をログ出力する。
    
    パイプライン終了時に呼び出してデータの健全性を確認する。
    
    Args:
        df: 検証対象DataFrame
        stage: パイプラインのステージ名（ログ用）
        
    Returns:
        検証結果の辞書 {'passed': bool, 'warnings': list, 'errors': list}
    """
    logger.info(f"[{stage}] データ品質チェックを開始...")
    
    result = {'passed': True, 'warnings': [], 'errors': []}
    
    # 1. rank系の0埋めチェック
    rank_cols = ['mean_rank_5', 'mean_last_3f_5', 'lag1_rank', 'mean_rank_all']
    for col in rank_cols:
        if col in df.columns:
            zero_rate = (df[col] == 0).mean()
            if zero_rate > 0.05:  # 5%以上が0なら警告
                msg = f"{col} の0率が高い: {zero_rate:.1%}"
                result['warnings'].append(msg)
                logger.warning(f"[{stage}] {msg}")
    
    # 2. 番兵値残存チェック
    if 'odds' in df.columns and (df['odds'] == 0).any():
        msg = f"odds=0 が残存: {(df['odds'] == 0).sum()}件"
        result['warnings'].append(msg)
        logger.warning(f"[{stage}] {msg}")
        
    if 'weight' in df.columns and ((df['weight'] == 999) | (df['weight'] == 0)).any():
        count = ((df['weight'] == 999) | (df['weight'] == 0)).sum()
        msg = f"weight番兵値が残存: {count}件"
        result['warnings'].append(msg)
        logger.warning(f"[{stage}] {msg}")
    
    # 3. unknown率チェック
    id_cols = ['jockey_id', 'trainer_id', 'sire_id', 'horse_id']
    for col in id_cols:
        if col in df.columns:
            unknown_rate = (df[col].astype(str).str.lower() == 'unknown').mean()
            if unknown_rate > 0.10:  # 10%以上がunknownなら警告
                msg = f"{col} のunknown率が高い: {unknown_rate:.1%}"
                result['warnings'].append(msg)
                logger.warning(f"[{stage}] {msg}")
    
    # 4. 全体NaN率チェック
    total_nan_rate = df.isna().mean().mean()
    if total_nan_rate > 0.05:
        msg = f"全体NaN率が高い: {total_nan_rate:.1%}"
        result['warnings'].append(msg)
        logger.warning(f"[{stage}] {msg}")
    
    if result['warnings']:
        logger.info(f"[{stage}] 品質チェック完了: {len(result['warnings'])}件の警告")
    else:
        logger.info(f"[{stage}] 品質チェック完了: 問題なし")
        
    return result


def log_top_categories(df: pd.DataFrame, col: str, n: int = 10) -> dict:
    """
    カテゴリカラムの上位N件の出現頻度をログ出力する。
    
    [A3対応] unknown/0が異常に多くないかを監視
    
    Args:
        df: 入力DataFrame
        col: カラム名
        n: 上位N件
        
    Returns:
        上位カテゴリの辞書
    """
    if col not in df.columns:
        return {}
        
    top = df[col].astype(str).value_counts().head(n)
    top_dict = top.to_dict()
    
    logger.info(f"Top {n} {col}: {top_dict}")
    
    # unknown/0 のチェック
    total = len(df)
    for suspicious in ['unknown', 'Unknown', 'UNKNOWN', '0', 0, 'nan', 'None']:
        suspicious_str = str(suspicious)
        if suspicious_str in top_dict:
            rate = top_dict[suspicious_str] / total
            if rate > 0.10:
                logger.warning(f"{col} の '{suspicious_str}' が {rate:.1%} を占めています（異常に高い可能性）")
    
    return top_dict


# ============================================================================
# [v11 Extended V6] リーク検査関数
# ============================================================================

def check_feature_leak(feature_cols: list, raise_error: bool = True) -> bool:
    """
    特徴量リストに禁止列（結果由来列）が含まれていないかチェックする。
    
    [V6対応] リーク防止の自動検査
    
    Args:
        feature_cols: 使用する特徴量カラムのリスト
        raise_error: True の場合、リーク検出時に例外を発生させる
        
    Returns:
        True: 問題なし, False: リーク検出
        
    Raises:
        ValueError: raise_error=True かつリーク検出時
    """
    leaked = set(feature_cols) & set(FORBIDDEN_COLUMNS)
    
    if leaked:
        msg = f"[LEAK DETECTED] 禁止列が特徴量に含まれています: {sorted(leaked)}"
        logger.error(msg)
        if raise_error:
            raise ValueError(msg)
        return False
    
    logger.info("[LEAK CHECK] OK - 禁止列の混入なし")
    return True


def get_allowed_feature_cols(df: pd.DataFrame, 
                             exclude_patterns: list = None,
                             include_embedding: bool = True) -> list:
    """
    allowlist方式で安全な特徴量リストを生成する。
    
    [v11 Extended] select_dtypes による自動生成は禁止。
    明示的に許可された列のみを返す。
    
    Args:
        df: 入力DataFrame
        exclude_patterns: 追加で除外するパターン（正規表現）
        include_embedding: embedding特徴量を含めるか
        
    Returns:
        安全な特徴量リスト
    """
    # 基本的な非特徴量列
    non_feature_cols = [
        'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
        'date', 'venue', 'horse_name', 'title', 'sex', 'weather', 'surface', 'state',
        'course_id', 'distance_cat', 'direction',  # カテゴリ型（別途処理が必要）
    ] + FORBIDDEN_COLUMNS
    
    # 許可される列を取得
    allowed = []
    for col in df.columns:
        if col in non_feature_cols:
            continue
        if col.endswith('_is_missing'):
            allowed.append(col)  # 欠損フラグは許可
            continue
        if not include_embedding and col.endswith('_emb_'):
            continue
        if exclude_patterns:
            import re
            skip = False
            for pattern in exclude_patterns:
                if re.search(pattern, col):
                    skip = True
                    break
            if skip:
                continue
        allowed.append(col)
    
    # 最終チェック
    check_feature_leak(allowed, raise_error=False)
    
    return allowed

