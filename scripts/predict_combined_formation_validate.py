"""
Feature validation for predict_combined_formation.py
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

MODEL_PROFILES = {
    "BASE": {
        "v13_model_path": "models/experiments/exp_lambdarank_hard_weighted/model.pkl",
        "v13_feats_path": "models/experiments/exp_lambdarank_hard_weighted/features.csv",
        "v14_model_path": "models/experiments/exp_gap_v14_production/model_v14.pkl",
        "v14_feats_path": "models/experiments/exp_gap_v14_production/features.csv",
        "cache_dir": "data/features_v14/prod_cache",
    },
    "ENHANCED": {
        "v13_model_path": "models/experiments/exp_lambdarank_hard_weighted_enhanced/model.pkl",
        "v13_feats_path": "models/experiments/exp_lambdarank_hard_weighted_enhanced/features.csv",
        "v14_model_path": "models/experiments/exp_gap_v14_production_enhanced/model_v14.pkl",
        "v14_feats_path": "models/experiments/exp_gap_v14_production_enhanced/features.csv",
        "cache_dir": "data/features_v14/prod_cache_enhanced",
    },
}


def load_feature_list(path, logger):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load feature list: {e}")
        return []
    if 'feature' in df.columns:
        return df['feature'].tolist()
    if '0' in df.columns:
        return df['0'].tolist()
    return df.iloc[:, 0].tolist()


def add_v13_odds_features(df, logger):
    if 'odds_10min' in df.columns:
        odds_base = df['odds_10min']
    elif 'odds' in df.columns:
        odds_base = df['odds']
    else:
        logger.warning("V13 odds features: odds column missing. Using neutral defaults.")
        odds_base = pd.Series(np.nan, index=df.index)

    df['odds_calc'] = odds_base.fillna(10.0)
    df['odds_rank'] = df.groupby('race_id')['odds_calc'].rank(ascending=True, method='min')

    if 'relative_horse_elo_z' in df.columns:
        df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False, method='min')
        df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
    else:
        df['odds_rank_vs_elo'] = 0

    df['is_high_odds'] = (df['odds_calc'] >= 10).astype(int)
    df['is_mid_odds'] = ((df['odds_calc'] >= 5) & (df['odds_calc'] < 10)).astype(int)
    if 'odds' in df.columns:
        df['odds'] = df['odds_calc']
    return df


def merge_odds_fluctuation_features(df, logger):
    try:
        from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
        odds_df = compute_odds_fluctuation(df)
        if odds_df.empty:
            logger.warning("Odds fluctuation features empty. Using fallback odds.")
            return df
        odds_cols = [
            'odds_ratio_10min',
            'odds_ratio_60_10',
            'rank_diff_10min',
            'odds_log_ratio_10min',
            'odds_final',
            'odds_10min',
            'odds_60min'
        ]
        odds_df = odds_df.drop(columns=['horse_id'], errors='ignore')
        rename_map = {c: f"{c}_calc" for c in odds_cols if c in odds_df.columns}
        odds_df = odds_df.rename(columns=rename_map)
        df = pd.merge(df, odds_df, on=['race_id', 'horse_number'], how='left')
        for col in odds_cols:
            calc_col = f"{col}_calc"
            if calc_col not in df.columns:
                continue
            if col in df.columns:
                df[col] = df[col].combine_first(df[calc_col])
            else:
                df[col] = df[calc_col]
            df.drop(columns=[calc_col], inplace=True)
        return df
    except Exception as e:
        logger.warning(f"Odds fluctuation skipped: {e}")
        return df


def build_feature_matrix(df, feats, fill_value=np.nan):
    X = df.reindex(columns=feats, fill_value=fill_value)
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X


def summarize_features(X, label, logger):
    if X.empty:
        logger.warning(f"{label}: empty matrix")
        return

    nan_rate = X.isna().mean()
    const_cols = X.nunique(dropna=False) <= 1

    logger.info(f"{label}: rows={len(X):,} cols={X.shape[1]:,}")
    logger.info(f"{label}: nan_cols={int((nan_rate > 0).sum())}, high_nan_cols={int((nan_rate > 0.95).sum())}")
    logger.info(f"{label}: const_cols={int(const_cols.sum())}")

    high_nan = nan_rate[nan_rate > 0.95].sort_values(ascending=False).head(10)
    if not high_nan.empty:
        logger.info(f"{label}: high_nan sample:\n{high_nan}")

    const_sample = const_cols[const_cols].index.tolist()[:10]
    if const_sample:
        logger.info(f"{label}: const sample: {const_sample}")


def summarize_row_uniqueness(X, race_ids, label, logger):
    if X.empty:
        return
    row_hash = pd.util.hash_pandas_object(X, index=False)
    uniq_per_race = pd.DataFrame({'race_id': race_ids, 'hash': row_hash}).groupby('race_id')['hash'].nunique()
    flat = uniq_per_race[uniq_per_race <= 1]
    logger.info(f"{label}: races={uniq_per_race.size}, flat_races={flat.size}")
    if not flat.empty:
        logger.info(f"{label}: flat race sample={flat.index[:5].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYYMMDD')
    parser.add_argument('--race_id', type=str, help='Specific Race ID')
    parser.add_argument('--force', action='store_true', help='Force recompute features')
    parser.add_argument(
        '--model_profile',
        type=str,
        default='BASE',
        choices=['BASE', 'ENHANCED', 'base', 'enhanced'],
        help='Model profile to validate: BASE (current v13/v14) or ENHANCED (new v13/v14).'
    )
    args = parser.parse_args()

    date_str = args.date or (datetime.utcnow() + timedelta(hours=9)).strftime("%Y%m%d")
    target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    profile_key = args.model_profile.upper()
    profile = MODEL_PROFILES.get(profile_key, MODEL_PROFILES['BASE'])
    V13_FEATS_PATH = profile['v13_feats_path']
    V14_FEATS_PATH = profile['v14_feats_path']
    CACHE_DIR = profile['cache_dir']

    logger.info(f"Model profile: {profile_key}")
    logger.info(f"  V13 features: {V13_FEATS_PATH}")
    logger.info(f"  V14 features: {V14_FEATS_PATH}")

    feats_v13 = load_feature_list(V13_FEATS_PATH, logger)
    feats_v14 = load_feature_list(V14_FEATS_PATH, logger)
    if not feats_v13 or not feats_v14:
        logger.error("Feature list load failed.")
        return

    loader = JraVanDataLoader()
    start_history = "2016-01-01"
    df_raw = loader.load(history_start_date=start_history, end_date=target_date, skip_odds=False)
    if df_raw.empty:
        logger.warning("No data found.")
        return

    numeric_cols = ['time', 'last_3f', 'rank', 'weight', 'weight_diff', 'impost', 'honshokin', 'fukashokin']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    if 'time' in df_raw.columns:
        min_times = df_raw.groupby('race_id')['time'].transform('min')
        df_raw['time_diff'] = (df_raw['time'] - min_times).fillna(0)
    else:
        df_raw['time_diff'] = 0

    df_raw['race_id'] = df_raw['race_id'].astype(str)
    df_raw['horse_number'] = pd.to_numeric(df_raw['horse_number'], errors='coerce').fillna(0).astype(int)

    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    blocks = list(pipeline.registry.keys())
    df_features = pipeline.load_features(df_raw, blocks, force=args.force)

    core_cols = ['race_id', 'horse_number', 'date', 'odds', 'odds_10min', 'popularity', 'horse_name', 'start_time_str']
    available_core = [c for c in core_cols if c in df_raw.columns]
    df_merged = pd.merge(
        df_features,
        df_raw[available_core].drop_duplicates(['race_id', 'horse_number']),
        on=['race_id', 'horse_number'],
        how='left'
    )

    if 'date' in df_merged.columns:
        df_merged['date'] = pd.to_datetime(df_merged['date'])
        df_today = df_merged[df_merged['date'] == pd.to_datetime(target_date)].copy()
    else:
        df_raw['date_dt'] = pd.to_datetime(df_raw['date'])
        target_rids = df_raw[df_raw['date_dt'] == pd.to_datetime(target_date)]['race_id'].unique()
        df_today = df_merged[df_merged['race_id'].isin(target_rids)].copy()
        df_today['date'] = pd.to_datetime(target_date)

    if args.race_id:
        df_today = df_today[df_today['race_id'] == args.race_id].copy()

    if df_today.empty:
        logger.warning("Target subset is empty.")
        return

    df_today = merge_odds_fluctuation_features(df_today, logger)

    if 'odds' in df_today.columns:
        df_today['odds'] = pd.to_numeric(df_today['odds'], errors='coerce')
        df_today.loc[df_today['odds'] <= 0, 'odds'] = np.nan

    if 'odds_10min' not in df_today.columns:
        df_today['odds_10min'] = np.nan
    df_today['odds_10min'] = pd.to_numeric(df_today['odds_10min'], errors='coerce')

    invalid_odds_mask = df_today['odds_10min'].isna() | (df_today['odds_10min'] <= 0)
    if invalid_odds_mask.any():
        try:
            year_str = f"'{date_str[:4]}'"
            q_o1 = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = {year_str} AND kaisai_tsukihi = '{date_str[4:]}'"
            df_o1_raw = pd.read_sql(q_o1, loader.engine)
            if not df_o1_raw.empty:
                def build_rid(row):
                    try:
                        return f"{int(float(row['kaisai_nen']))}{int(float(row['keibajo_code'])):02}{int(float(row['kaisai_kai'])):02}{int(float(row['kaisai_nichime'])):02}{int(float(row['race_bango'])):02}"
                    except: return None
                df_o1_raw['race_id'] = df_o1_raw.apply(build_rid, axis=1)
                parsed = []
                for _, row in df_o1_raw.iterrows():
                    s, rid = row['odds_tansho'], row['race_id']
                    if not isinstance(s, str):
                        continue
                    for i in range(0, len(s), 8):
                        chunk = s[i:i+8]
                        if len(chunk) < 8:
                            break
                        try:
                            parsed.append({
                                'race_id': rid,
                                'horse_number': int(chunk[0:2]),
                                'odds_10min': int(chunk[2:6]) / 10.0,
                                'popularity_10min': int(chunk[6:8])
                            })
                        except:
                            continue
                if parsed:
                    df_o1_parsed = pd.DataFrame(parsed).rename(columns={'odds_10min': 'odds_10min_o1'})
                    df_today = pd.merge(df_today, df_o1_parsed, on=['race_id', 'horse_number'], how='left')
                    df_today.loc[invalid_odds_mask, 'odds_10min'] = df_today.loc[invalid_odds_mask, 'odds_10min_o1']
                    df_today = df_today.drop(columns=['odds_10min_o1'], errors='ignore')
        except:
            pass

    invalid_odds_mask = df_today['odds_10min'].isna() | (df_today['odds_10min'] <= 0)
    if invalid_odds_mask.any():
        if 'odds' in df_today.columns:
            df_today.loc[invalid_odds_mask, 'odds_10min'] = df_today.loc[invalid_odds_mask, 'odds']
        else:
            logger.warning("No odds info. Filling odds_10min with 10.0.")
            df_today.loc[invalid_odds_mask, 'odds_10min'] = 10.0

    if 'odds' not in df_today.columns:
        df_today['odds'] = df_today['odds_10min']

    if 'odds_final' not in df_today.columns:
        df_today['odds_final'] = df_today['odds']
    df_today['odds_final'] = df_today['odds_final'].fillna(df_today['odds_10min'])

    if 'odds_ratio_10min' not in df_today.columns or df_today['odds_ratio_10min'].isna().all():
        df_today['odds_ratio_10min'] = df_today['odds_final'] / df_today['odds_10min'].replace(0, np.nan)

    if 'odds_60min' not in df_today.columns:
        df_today['odds_60min'] = df_today['odds_10min']
    if 'odds_ratio_60_10' not in df_today.columns or df_today['odds_ratio_60_10'].isna().all():
        df_today['odds_ratio_60_10'] = df_today['odds_10min'] / df_today['odds_60min'].replace(0, np.nan)

    if 'odds_log_ratio_10min' not in df_today.columns or df_today['odds_log_ratio_10min'].isna().all():
        df_today['odds_log_ratio_10min'] = np.log(df_today['odds_final'] + 1e-9) - np.log(df_today['odds_10min'] + 1e-9)

    df_today['odds_60min'] = df_today['odds_60min'].fillna(df_today['odds_10min'])
    df_today['odds_ratio_10min'] = df_today['odds_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df_today['odds_ratio_60_10'] = df_today['odds_ratio_60_10'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df_today['odds_log_ratio_10min'] = df_today['odds_log_ratio_10min'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df_today['odds_rank_10min'] = df_today.groupby('race_id')['odds_10min'].rank(method='min')
    if 'popularity' not in df_today.columns or df_today['popularity'].isna().all():
        df_today['popularity'] = df_today['odds_rank_10min']
    df_today['rank_diff_10min'] = df_today['popularity'] - df_today['odds_rank_10min']

    if 'field_size' not in df_today.columns:
        df_today['field_size'] = df_today.groupby('race_id')['horse_number'].transform('count')

    df_v13_today = add_v13_odds_features(df_today.copy(), logger)

    missing_v13 = [c for c in feats_v13 if c not in df_v13_today.columns]
    missing_v14 = [c for c in feats_v14 if c not in df_today.columns]
    logger.info(f"V13 missing features: {len(missing_v13)}")
    if missing_v13:
        logger.info(f"V13 missing sample: {missing_v13[:10]}")
    logger.info(f"V14 missing features: {len(missing_v14)}")
    if missing_v14:
        logger.info(f"V14 missing sample: {missing_v14[:10]}")

    X_v13 = build_feature_matrix(df_v13_today, feats_v13, fill_value=np.nan)
    X_v14 = build_feature_matrix(df_today, feats_v14, fill_value=np.nan)

    summarize_features(X_v13, "V13", logger)
    summarize_features(X_v14, "V14", logger)

    summarize_row_uniqueness(X_v13, df_v13_today['race_id'].values, "V13", logger)
    summarize_row_uniqueness(X_v14, df_today['race_id'].values, "V14", logger)


if __name__ == "__main__":
    main()
