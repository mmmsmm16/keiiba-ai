"""
Update Aggregates Cache
========================
日次バッチで集計統計を更新するスクリプト。
本番予測の高速化のため、事前に統計値を計算してキャッシュしておく。
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = "data/aggregates"
HISTORY_YEARS = 5  # 過去5年分のデータを使用 (Nicks/Sire用 - メモリ考慮)

def ensure_cache_dir():
    """キャッシュディレクトリを作成"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def compute_jockey_stats(df: pd.DataFrame) -> pd.DataFrame:
    """騎手別統計を計算"""
    logger.info("Computing jockey stats...")
    
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    stats = df.groupby('jockey_id').agg({
        'is_win': ['sum', 'count', 'mean'],
        'is_top3': 'mean',
        'rank': 'mean'
    }).reset_index()
    
    stats.columns = ['jockey_id', 'jockey_wins', 'jockey_rides', 'jockey_win_rate', 
                     'jockey_top3_rate', 'jockey_avg_rank']
    
    return stats

def compute_trainer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """調教師別統計を計算"""
    logger.info("Computing trainer stats...")
    
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    stats = df.groupby('trainer_id').agg({
        'is_win': ['sum', 'count', 'mean'],
        'is_top3': 'mean',
        'rank': 'mean'
    }).reset_index()
    
    stats.columns = ['trainer_id', 'trainer_wins', 'trainer_entries', 'trainer_win_rate',
                     'trainer_top3_rate', 'trainer_avg_rank']
    
    return stats

def compute_sire_stats(df: pd.DataFrame) -> pd.DataFrame:
    """種牡馬別統計を計算"""
    logger.info("Computing sire stats...")
    
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    stats = df.groupby('sire_id').agg({
        'is_win': ['sum', 'count', 'mean'],
        'is_top3': 'mean',
        'rank': 'mean'
    }).reset_index()
    
    stats.columns = ['sire_id', 'sire_wins', 'sire_entries', 'sire_win_rate',
                     'sire_top3_rate', 'sire_avg_rank']
    
    return stats

def compute_horse_stats(df: pd.DataFrame) -> pd.DataFrame:
    """馬別履歴統計と直近状態（Lag特徴量）を計算"""
    logger.info("Computing horse stats & lag features...")
    
    # 1. Basic Aggregates
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    basic_stats = df.groupby('horse_id').agg({
        'is_win': 'mean',
        'is_top3': 'mean',
        'rank': 'mean',
        'race_id': 'count'  # run_count
    }).reset_index()
    basic_stats.columns = ['horse_id', 'horse_win_rate', 'horse_top3_rate', 'horse_mean_rank', 'run_count']
    
    # 2. Lag / Recent Features (Latest status per horse)
    # Sort by horse and date
    df_sorted = df.sort_values(['horse_id', 'date'])
    
    # Rolling Stats (Last 5 races)
    # Calculate rolling mean for each horse
    cols_to_roll = ['rank']
    if 'time_diff' in df_sorted.columns:
        cols_to_roll.append('time_diff')
    
    for col in cols_to_roll:
        df_sorted[f'{col}_rolling_5'] = df_sorted.groupby('horse_id')[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Extract latest status
    latest = df_sorted.drop_duplicates(subset=['horse_id'], keep='last').copy()
    
    rename_map = {
        'date': 'last_race_date',
        'rank': 'lag1_rank',
        'rank_rolling_5': 'mean_rank_5'
    }
    
    if 'time_diff' in df_sorted.columns:
        rename_map['time_diff'] = 'lag1_time_diff'
        rename_map['time_diff_rolling_5'] = 'mean_time_diff_5'
        
    latest_features = latest[['horse_id', 'date'] + list(rename_map.keys())[2:]].rename(columns=rename_map)
    
    # Merge everything
    stats = pd.merge(basic_stats, latest_features, on='horse_id', how='left')
    
    # Ensure date is string for serialization
    stats['last_race_date'] = stats['last_race_date'].astype(str)
    
    return stats

def compute_jockey_trainer_combos(df: pd.DataFrame) -> pd.DataFrame:
    """騎手×調教師の組み合わせ統計を計算"""
    logger.info("Computing jockey-trainer combo stats...")
    
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    stats = df.groupby(['jockey_id', 'trainer_id']).agg({
        'is_win': ['sum', 'count', 'mean'],
        'is_top3': 'mean'
    }).reset_index()
    
    stats.columns = ['jockey_id', 'trainer_id', 'combo_wins', 'combo_rides', 
                     'combo_win_rate', 'combo_top3_rate']
    
    # 少なくとも3戦以上の組み合わせのみ
    stats = stats[stats['combo_rides'] >= 3]
    
    return stats

def compute_course_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """コース適性統計を計算（馬×コース）"""
    logger.info("Computing course aptitude stats...")
    
    # Check for required columns
    if 'keibajo_code' not in df.columns:
        if 'race_id' in df.columns:
            logger.info("keibajo_code missing, deriving from race_id...")
            # race_id format: YYYY(4) + Venue(2) + ...
            # Assuming race_id is string. If not, cast it.
            df['keibajo_code'] = df['race_id'].astype(str).str[4:6]
        else:
            logger.warning("keibajo_code and race_id columns not found, skipping course aptitude")
            return pd.DataFrame()
    
    # コースキーを作成（競馬場 + 芝/ダ + 距離カテゴリ）
    df['distance_cat'] = pd.cut(df['distance'].astype(float), 
                                 bins=[0, 1400, 1800, 2200, 9999],
                                 labels=['sprint', 'mile', 'middle', 'long'])
    
    surface_col = 'surface' if 'surface' in df.columns else 'track_type'
    if surface_col in df.columns:
        df['course_key'] = df['keibajo_code'].astype(str) + '_' + df[surface_col].astype(str) + '_' + df['distance_cat'].astype(str)
    else:
        df['course_key'] = df['keibajo_code'].astype(str) + '_' + df['distance_cat'].astype(str)
    
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    stats = df.groupby(['horse_id', 'course_key']).agg({
        'is_win': 'mean',
        'is_top3': 'mean',
        'rank': ['mean', 'count']
    }).reset_index()
    
    stats.columns = ['horse_id', 'course_key', 'course_win_rate', 'course_top3_rate',
                     'course_avg_rank', 'course_races']
    
    # 少なくとも2戦以上
    stats = stats[stats['course_races'] >= 2]
    
    return stats

def compute_nicks_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Nicks (Sire x BMS) 統計を計算"""
    logger.info("Computing nicks (sire x bms) stats...")
    
    # Required columns
    if 'sire_id' not in df.columns or 'bms_id' not in df.columns:
        logger.warning("sire_id or bms_id missing, skipping nicks stats")
        return pd.DataFrame()
        
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    # Group by Sire + BMS
    stats = df.groupby(['sire_id', 'bms_id']).agg({
        'is_win': ['sum', 'count', 'mean'],
        'is_top3': 'mean',
        'rank': 'mean'
    }).reset_index()
    
    stats.columns = ['sire_id_nicks', 'bms_id_nicks', 'nicks_wins', 'nicks_matches', 
                     'nicks_win_rate', 'nicks_top3_rate', 'nicks_avg_rank']
    
    # Filter for robustness (at least 2 matches)
    stats = stats[stats['nicks_matches'] >= 2]
    
    # Key mapping columns (rename to match expected merge keys or keep distinct)
    # The pipeline usually expects 'sire_id' and 'bms_id' in the main DF, 
    # but for merging, we'll rename to match the main DF keys in production script.
    stats = stats.rename(columns={'sire_id_nicks': 'sire_id', 'bms_id_nicks': 'bms_id'})
    
    return stats

def main():
    logger.info("=" * 60)
    logger.info("Starting Aggregates Cache Update")
    logger.info("=" * 60)
    
    ensure_cache_dir()
    
    # 1. データロード
    loader = JraVanDataLoader()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")
    
    logger.info(f"Loading data from {start_date} to {end_date}...")
    df = loader.load(history_start_date=start_date)
    
    # rank を数値化
    if 'rank' not in df.columns and 'rank_str' in df.columns:
        df['rank'] = pd.to_numeric(df['rank_str'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    
    # 出走前データ（rank が NaN）を除外
    df_finished = df[df['rank'].notnull()].copy()
    
    logger.info(f"Total records: {len(df)}, Finished races: {len(df_finished)}")
    
    # 2. 各統計を計算・保存
    # Jockey Stats
    jockey_stats = compute_jockey_stats(df_finished)
    jockey_stats.to_parquet(os.path.join(CACHE_DIR, "jockey_stats.parquet"))
    logger.info(f"Saved jockey_stats: {len(jockey_stats)} jockeys")
    
    # Trainer Stats
    trainer_stats = compute_trainer_stats(df_finished)
    trainer_stats.to_parquet(os.path.join(CACHE_DIR, "trainer_stats.parquet"))
    logger.info(f"Saved trainer_stats: {len(trainer_stats)} trainers")
    
    # Sire Stats
    sire_stats = compute_sire_stats(df_finished)
    sire_stats.to_parquet(os.path.join(CACHE_DIR, "sire_stats.parquet"))
    logger.info(f"Saved sire_stats: {len(sire_stats)} sires")
    
    # Horse Stats
    horse_stats = compute_horse_stats(df_finished)
    horse_stats.to_parquet(os.path.join(CACHE_DIR, "horse_stats.parquet"))
    logger.info(f"Saved horse_stats: {len(horse_stats)} horses")
    
    # Jockey-Trainer Combos
    jt_combos = compute_jockey_trainer_combos(df_finished)
    jt_combos.to_parquet(os.path.join(CACHE_DIR, "jockey_trainer_combos.parquet"))
    logger.info(f"Saved jockey_trainer_combos: {len(jt_combos)} combinations")
    
    # Course Aptitude
    course_apt = compute_course_aptitude(df_finished)
    course_apt.to_parquet(os.path.join(CACHE_DIR, "course_aptitude.parquet"))
    logger.info(f"Saved course_aptitude: {len(course_apt)} horse-course pairs")

    # Nicks Stats
    nicks_stats = compute_nicks_stats(df_finished)
    nicks_stats.to_parquet(os.path.join(CACHE_DIR, "nicks_stats.parquet"))
    logger.info(f"Saved nicks_stats: {len(nicks_stats)} combinations")
    
    
    # 3. 更新日時を記録
    metadata = {
        "last_updated": datetime.now().isoformat(),
        "data_range": {"start": start_date, "end": end_date},
        "record_counts": {
            "jockey": len(jockey_stats),
            "trainer": len(trainer_stats),
            "sire": len(sire_stats),
            "horse": len(horse_stats),
            "jockey_trainer": len(jt_combos),
            "course_aptitude": len(course_apt),
            "nicks": len(nicks_stats)
        }
    }
    
    with open(os.path.join(CACHE_DIR, "last_updated.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Aggregates Cache Update Complete!")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
