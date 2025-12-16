"""
Phase 6: Build Predictions Table
予測parquetの統一スキーマ作成（正規化・校正列含む）

Usage (in container):
    docker compose exec app python src/phase6/build_predictions_table.py
    docker compose exec app python src/phase6/build_predictions_table.py --input data/derived/preprocessed_with_prob_v12.parquet

Standard Schema:
    - race_id, horse_key (umaban/horse_id), date, year
    - odds, overround
    - p_market (normalized market prob)
    - prob_model_raw (original model output)
    - prob_model_norm (race-normalized)
    - prob_model_calib_temp, prob_model_calib_isotonic, prob_model_calib_beta_full
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.race_filter import filter_jra_only
from utils.calibration import (
    TemperatureScaling, IsotonicCalibrator, FullBetaCalibrator,
    create_market_prob, normalize_prob_per_race
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Standard column names
STANDARD_COLS = {
    'race_id': 'race_id',
    'horse_key': None,  # Will be determined (umaban or horse_id)
    'date': 'date',
    'year': 'year',
    'odds': 'odds',
    'rank': 'rank',
    'p_market': 'p_market',
    'prob_model_raw': 'prob_model_raw',
    'prob_model_norm': 'prob_model_norm',
    'prob_model_calib_temp': 'prob_model_calib_temp',
    'prob_model_calib_isotonic': 'prob_model_calib_isotonic',
    'prob_model_calib_beta_full': 'prob_model_calib_beta_full',
}


def determine_horse_key(df: pd.DataFrame) -> str:
    """horse_key列を決定"""
    if 'umaban' in df.columns:
        return 'umaban'
    elif 'horse_id' in df.columns:
        return 'horse_id'
    elif 'horse_number' in df.columns:
        return 'horse_number'
    else:
        raise ValueError("No horse key column found (umaban/horse_id/horse_number)")


def create_p_market(df: pd.DataFrame) -> pd.DataFrame:
    """市場確率を作成（レース内正規化）"""
    df = df.copy()
    
    if 'odds' not in df.columns or df['odds'].isna().all():
        logger.warning("No valid odds column, p_market will be NaN")
        df['p_market'] = np.nan
        return df
    
    df['p_market_raw'] = 1.0 / df['odds'].replace(0, np.nan)
    df['p_market'] = df.groupby('race_id')['p_market_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.nan
    )
    
    # Calculate overround per race
    df['overround'] = df.groupby('race_id')['p_market_raw'].transform('sum')
    
    # Drop temp column
    df = df.drop(columns=['p_market_raw'], errors='ignore')
    
    n_valid = df['p_market'].notna().sum()
    logger.info(f"Created p_market: {n_valid:,} valid rows")
    
    return df


def create_prob_model_raw(df: pd.DataFrame) -> pd.DataFrame:
    """prob_model_raw列を作成"""
    df = df.copy()
    
    # Try to find source column
    source_cols = ['prob', 'prob_model_raw', 'prediction', 'pred']
    source_col = None
    
    for col in source_cols:
        if col in df.columns and df[col].notna().any():
            source_col = col
            break
    
    if source_col is None:
        logger.warning("No model probability column found")
        df['prob_model_raw'] = np.nan
        return df
    
    if source_col != 'prob_model_raw':
        df['prob_model_raw'] = df[source_col]
        logger.info(f"Renamed {source_col} -> prob_model_raw")
    
    n_valid = df['prob_model_raw'].notna().sum()
    logger.info(f"prob_model_raw: {n_valid:,} valid rows")
    
    return df


def create_prob_model_norm(df: pd.DataFrame) -> pd.DataFrame:
    """正規化確率を作成"""
    df = df.copy()
    
    if 'prob_model_raw' not in df.columns:
        df['prob_model_norm'] = np.nan
        return df
    
    df['prob_model_norm'] = normalize_prob_per_race(
        df['prob_model_raw'].values,
        df['race_id'].values
    )
    
    n_valid = df['prob_model_norm'].notna().sum()
    n_nan = df['prob_model_norm'].isna().sum()
    logger.info(f"Created prob_model_norm: {n_valid:,} valid, {n_nan:,} NaN")
    
    return df


def fit_and_apply_calibrators(
    df: pd.DataFrame,
    train_years: List[int],
    apply_years: List[int]
) -> pd.DataFrame:
    """校正器をfit & apply"""
    df = df.copy()
    
    # Initialize columns
    df['prob_model_calib_temp'] = np.nan
    df['prob_model_calib_isotonic'] = np.nan
    df['prob_model_calib_beta_full'] = np.nan
    
    # Split data
    train_mask = df['year'].isin(train_years)
    train_df = df[train_mask & df['prob_model_norm'].notna() & df['rank'].notna()].copy()
    
    if len(train_df) < 1000:
        logger.warning(f"Not enough training data ({len(train_df)} rows), skipping calibration")
        return df
    
    y_train = (train_df['rank'] == 1).astype(int).values
    p_train = train_df['prob_model_norm'].values
    
    logger.info(f"Fitting calibrators on {len(train_df):,} rows (years {train_years})")
    
    # Temperature
    try:
        temp_cal = TemperatureScaling()
        temp_cal.fit(p_train, y_train)
        
        apply_mask = df['year'].isin(apply_years) & df['prob_model_norm'].notna()
        df.loc[apply_mask, 'prob_model_calib_temp'] = temp_cal.predict(
            df.loc[apply_mask, 'prob_model_norm'].values
        )
        logger.info(f"Temperature calibration applied (T={temp_cal.optimal_T:.4f})")
    except Exception as e:
        logger.error(f"Temperature calibration failed: {e}")
    
    # Isotonic
    try:
        iso_cal = IsotonicCalibrator()
        iso_cal.fit(p_train, y_train)
        
        apply_mask = df['year'].isin(apply_years) & df['prob_model_norm'].notna()
        df.loc[apply_mask, 'prob_model_calib_isotonic'] = iso_cal.predict(
            df.loc[apply_mask, 'prob_model_norm'].values
        )
        logger.info("Isotonic calibration applied")
    except Exception as e:
        logger.error(f"Isotonic calibration failed: {e}")
    
    # Full Beta
    try:
        beta_cal = FullBetaCalibrator()
        beta_cal.fit(p_train, y_train)
        
        apply_mask = df['year'].isin(apply_years) & df['prob_model_norm'].notna()
        df.loc[apply_mask, 'prob_model_calib_beta_full'] = beta_cal.predict(
            df.loc[apply_mask, 'prob_model_norm'].values
        )
        logger.info(f"Full Beta calibration applied (a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f})")
    except Exception as e:
        logger.error(f"Full Beta calibration failed: {e}")
    
    return df


def log_schema(df: pd.DataFrame, title: str = "Output Schema"):
    """スキーマをログ出力"""
    logger.info(f"\n=== {title} ===")
    logger.info(f"Rows: {len(df):,}, Races: {df['race_id'].nunique():,}")
    logger.info("Columns:")
    for col in df.columns:
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        logger.info(f"  {col}: {n_valid:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Build Predictions Table")
    parser.add_argument('--input', type=str, default='data/derived/preprocessed_with_prob_v12.parquet')
    parser.add_argument('--output', type=str, default='data/predictions/calibrated/v12_oof_unified.parquet')
    parser.add_argument('--include_nar', action='store_true', default=False)
    parser.add_argument('--train_years', type=int, nargs='+', default=[2021, 2022, 2023])
    parser.add_argument('--apply_years', type=int, nargs='+', default=[2021, 2022, 2023, 2024])
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    logger.info(f"Input columns: {list(df.columns)}")
    
    if 'year' not in df.columns:
        df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    # JRA-only filter
    if not args.include_nar:
        df = filter_jra_only(df)
    
    logger.info(f"After filter: {len(df):,} rows, {df['race_id'].nunique():,} races")
    
    # Determine horse_key
    horse_key = determine_horse_key(df)
    logger.info(f"Horse key column: {horse_key}")
    
    # Create standard columns
    df = create_p_market(df)
    df = create_prob_model_raw(df)
    df = create_prob_model_norm(df)
    
    # Fit and apply calibrators
    df = fit_and_apply_calibrators(df, args.train_years, args.apply_years)
    
    # Log schema
    log_schema(df, "Final Output Schema")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info(f"Saved to {args.output}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
