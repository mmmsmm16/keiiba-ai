"""
Race Filter Utilities
JRA/NAR/海外レースのフィルタリング

JRA判定ロジック:
- race_id の5文字目（0-indexed: 4）が '0' であればJRA
- venue code が 01-10 であればJRA（01=札幌, 02=函館, ..., 10=小倉）
- それ以外はNAR（地方）または海外

Usage:
    from utils.race_filter import filter_jra_only, add_jra_flag
    df = filter_jra_only(df)  # JRAのみに絞る
    df = add_jra_flag(df)     # is_jra列を追加
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# JRAの競馬場コード（01-10）
JRA_VENUE_CODES = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# 競馬場名マッピング（参考）
VENUE_NAMES = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
    # NAR（地方）は30-55など
}


def is_jra_race_id(race_id: str) -> bool:
    """
    race_idからJRAかどうかを判定
    
    JRA race_id pattern: YYYYXPPKKNNRR
    - YYYY: 年
    - X: 組織区分 ('0' = JRA, '3','4','5' = NAR, 'A'-'Z' = 海外等)
    - PP: 競馬場コード
    - KK: 開催回
    - NN: 日目
    - RR: レース番号
    
    Args:
        race_id: レースID文字列
    
    Returns:
        JRAならTrue
    """
    if not isinstance(race_id, str) or len(race_id) < 5:
        return False
    return race_id[4] == '0'


def is_jra_venue(venue: str) -> bool:
    """
    venueコードからJRAかどうかを判定
    
    Args:
        venue: 競馬場コード (01-10がJRA)
    
    Returns:
        JRAならTrue
    """
    return str(venue).zfill(2) in JRA_VENUE_CODES


def add_jra_flag(df: pd.DataFrame, race_id_col: str = 'race_id', venue_col: str = 'venue') -> pd.DataFrame:
    """
    DataFrameにis_jraフラグを追加
    
    Args:
        df: 対象DataFrame
        race_id_col: race_id列名
        venue_col: venue列名（存在しない場合はrace_idのみで判定）
    
    Returns:
        is_jra列を追加したDataFrame
    """
    result = df.copy()
    
    # race_idベースの判定（主判定）
    result['is_jra'] = result[race_id_col].astype(str).apply(is_jra_race_id)
    
    # venue列がある場合は交差確認用にフラグ追加
    if venue_col in result.columns:
        result['is_jra_venue'] = result[venue_col].astype(str).str.zfill(2).isin(JRA_VENUE_CODES)
    
    return result


def filter_jra_only(
    df: pd.DataFrame,
    race_id_col: str = 'race_id',
    log_stats: bool = True
) -> pd.DataFrame:
    """
    JRAレースのみにフィルタ
    
    Args:
        df: 対象DataFrame
        race_id_col: race_id列名
        log_stats: フィルタ前後の統計をログ出力するか
    
    Returns:
        JRAレースのみのDataFrame
    """
    before_rows = len(df)
    before_races = df[race_id_col].nunique() if race_id_col in df.columns else 0
    
    # race_id 5文字目が '0' のレースのみ
    mask = df[race_id_col].astype(str).str[4:5] == '0'
    result = df[mask].copy()
    
    after_rows = len(result)
    after_races = result[race_id_col].nunique() if race_id_col in result.columns else 0
    
    if log_stats:
        logger.info(f"JRA filter applied: {before_rows:,} → {after_rows:,} rows "
                    f"(-{before_rows - after_rows:,}), "
                    f"{before_races:,} → {after_races:,} races "
                    f"(-{before_races - after_races:,})")
    
    return result


def filter_races(
    df: pd.DataFrame,
    include_nar: bool = False,
    include_overseas: bool = False,
    race_id_col: str = 'race_id',
    log_stats: bool = True
) -> pd.DataFrame:
    """
    レースフィルタ（汎用）
    
    Args:
        df: 対象DataFrame
        include_nar: NARを含めるか（default: False = JRAのみ）
        include_overseas: 海外を含めるか（default: False）
        race_id_col: race_id列名
        log_stats: ログ出力するか
    
    Returns:
        フィルタ後のDataFrame
    """
    if not include_nar and not include_overseas:
        # JRAのみ
        return filter_jra_only(df, race_id_col, log_stats)
    
    before_rows = len(df)
    
    # race_id[4]で判定
    df_temp = df.copy()
    df_temp['_org_code'] = df_temp[race_id_col].astype(str).str[4:5]
    
    mask = pd.Series([False] * len(df_temp), index=df_temp.index)
    
    # JRA (prefix '0') は常に含む
    mask |= (df_temp['_org_code'] == '0')
    
    if include_nar:
        # NAR (prefix '3', '4', '5')
        mask |= df_temp['_org_code'].isin(['3', '4', '5'])
    
    if include_overseas:
        # 海外 (alphabet prefixes)
        mask |= df_temp['_org_code'].str.isalpha()
    
    result = df[mask].copy()
    
    if log_stats:
        after_rows = len(result)
        logger.info(f"Race filter (include_nar={include_nar}, include_overseas={include_overseas}): "
                    f"{before_rows:,} → {after_rows:,} rows")
    
    return result


def add_race_filter_args(parser):
    """ArgumentParserにフィルタオプションを追加"""
    parser.add_argument('--include_nar', action='store_true', default=False,
                        help='Include NAR (local racing) data (default: JRA only)')
    parser.add_argument('--include_overseas', action='store_true', default=False,
                        help='Include overseas race data (default: exclude)')
    return parser


def apply_race_filter_from_args(df: pd.DataFrame, args, race_id_col: str = 'race_id') -> pd.DataFrame:
    """argparse argsからフィルタを適用"""
    include_nar = getattr(args, 'include_nar', False)
    include_overseas = getattr(args, 'include_overseas', False)
    return filter_races(df, include_nar=include_nar, include_overseas=include_overseas, 
                        race_id_col=race_id_col)


def get_race_stats(df: pd.DataFrame, race_id_col: str = 'race_id') -> dict:
    """
    JRA/NAR/海外の内訳統計を取得
    """
    df_temp = df.copy()
    df_temp['_org_code'] = df_temp[race_id_col].astype(str).str[4:5]
    
    stats = {
        'total_rows': len(df),
        'total_races': df[race_id_col].nunique(),
        'jra_rows': (df_temp['_org_code'] == '0').sum(),
        'nar_rows': df_temp['_org_code'].isin(['3', '4', '5']).sum(),
        'overseas_rows': df_temp['_org_code'].str.isalpha().sum(),
    }
    
    jra_mask = df_temp['_org_code'] == '0'
    stats['jra_races'] = df[jra_mask][race_id_col].nunique()
    
    nar_mask = df_temp['_org_code'].isin(['3', '4', '5'])
    stats['nar_races'] = df[nar_mask][race_id_col].nunique()
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Race Filter Test")
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data_v11.parquet')
    add_race_filter_args(parser)
    
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    
    # Show stats before filter
    stats = get_race_stats(df)
    print("=== Before Filter ===")
    print(f"Total: {stats['total_rows']:,} rows, {stats['total_races']:,} races")
    print(f"JRA: {stats['jra_rows']:,} rows, {stats['jra_races']:,} races")
    print(f"NAR: {stats['nar_rows']:,} rows, {stats['nar_races']:,} races")
    print(f"Overseas: {stats['overseas_rows']:,} rows")
    
    # Apply filter
    print("\n=== After Filter ===")
    filtered = apply_race_filter_from_args(df, args)
    
    # Odds stats
    if 'odds' in filtered.columns:
        odds_null = filtered['odds'].isna().mean() * 100
        print(f"Odds null rate: {odds_null:.2f}%")
