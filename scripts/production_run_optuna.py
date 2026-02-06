"""
Production Run with Optuna-Optimized Model
===========================================
Uses: models/experiments/optuna_best_full/model.pkl
Data: data/processed/preprocessed_data_v11.parquet
Strategy: Win-Top1, EV >= 1.5, Prob >= 0.10 (ROI ~107%)

Usage:
  python scripts/production_run_optuna.py
  python scripts/production_run_optuna.py --date 20260117
  python scripts/production_run_optuna.py --race_id 202606010501 --discord
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
CALIBRATOR_PATH = "models/experiments/exp_t2_refined_v3/calibrator.pkl"
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

# EV Strategy (from backtesting: ROI ~107%)
EV_THRESHOLD = 1.5
PROB_THRESHOLD = 0.10

# Columns to exclude
LEAKAGE_COLS = [
    'pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank',
    'last_3f', 'raw_time', 'time_diff', 'margin', 'time',
    'popularity', 'odds', 'relative_popularity_rank',
    'slow_start_recovery', 'track_bias_disadvantage',
    'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate'
]
META_COLS = ['race_id', 'horse_number', 'date', 'rank', 'odds_final',
             'is_win', 'is_top2', 'is_top3', 'year', 'rank_str']
ID_COLS = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']

VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}


def send_discord(webhook_url, content):
    """Send message to Discord"""
    if not webhook_url:
        return
    try:
        MAX_LEN = 1900
        chunks = []
        current = ""
        for line in content.split('\n'):
            if len(current) + len(line) + 1 > MAX_LEN:
                chunks.append(current)
                current = line + "\n"
            else:
                current += line + "\n"
        if current:
            chunks.append(current)
        for chunk in chunks:
            requests.post(webhook_url, json={"content": f"```\n{chunk}\n```"})
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"Discord error: {e}")


def parse_tansho_odds(odds_str):
    """Parse odds_tansho string: 8 chars per horse"""
    result = {}
    if not odds_str or pd.isna(odds_str):
        return result
    s = str(odds_str)
    for i in range(0, len(s), 8):
        block = s[i:i+8]
        if len(block) < 8:
            break
        try:
            h = int(block[0:2])
            o = int(block[2:6]) / 10.0
            if h > 0 and o > 0:
                result[h] = o
        except:
            pass
    return result


def get_realtime_odds(loader, race_id):
    """Get latest odds for a race"""
    query = f"""
    SELECT odds_tansho FROM jvd_o1
    WHERE kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango = '{race_id}'
    ORDER BY happyo_tsukihi_jifun DESC LIMIT 1
    """
    try:
        df = pd.read_sql(query, loader.engine)
        if len(df) > 0:
            return parse_tansho_odds(df.iloc[0]['odds_tansho'])
    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
    return {}


def get_horse_names(loader, race_id):
    """Get horse names for a race"""
    query = f"""
    SELECT umaban, bamei FROM jvd_se
    WHERE kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango = '{race_id}'
    """
    try:
        df = pd.read_sql(query, loader.engine)
        return dict(zip(df['umaban'].astype(int), df['bamei']))
    except:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date (YYYYMMDD)")
    parser.add_argument("--race_id", type=str, help="Single race ID")
    parser.add_argument("--discord", action='store_true')
    parser.add_argument("--webhook_url", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    
    logger.info("=" * 60)
    logger.info(f"🚀 Optuna Model Production Run: {target_date}")
    logger.info(f"   EV >= {EV_THRESHOLD}, Prob >= {PROB_THRESHOLD}")
    logger.info("=" * 60)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)
    expected_features = model.feature_name()
    logger.info(f"Loaded model with {len(expected_features)} features")
    
    # Load calibrator
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        calibrator = joblib.load(CALIBRATOR_PATH)
        logger.info("Loaded calibrator for probability calibration")
    else:
        logger.warning("Calibrator not found, using raw probabilities")
    
    # Load preprocessed data
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data not found: {DATA_PATH}")
        return
    df_all = pd.read_parquet(DATA_PATH)
    df_all['race_id'] = df_all['race_id'].astype(str)
    df_all['date'] = pd.to_datetime(df_all['date'])
    logger.info(f"Loaded data: {len(df_all)} records")
    
    # Get DB connection for odds and horse names
    loader = JraVanDataLoader()
    
    # Get today's race list
    if args.race_id:
        race_ids = [args.race_id]
    else:
        query = f"""
        SELECT DISTINCT kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id
        FROM jvd_ra WHERE kaisai_nen = '{target_date[:4]}' AND kaisai_tsukihi = '{target_date[4:]}'
        ORDER BY race_id
        """
        df_races = pd.read_sql(query, loader.engine)
        race_ids = df_races['race_id'].tolist()
    
    if not race_ids:
        logger.info("No races found.")
        return
    
    logger.info(f"Processing {len(race_ids)} races...")
    
    # Filter data for today's races
    df_today = df_all[df_all['race_id'].isin(race_ids)].copy()
    
    if df_today.empty:
        logger.warning(f"No data in parquet for today's races. Run update_daily_features.py first.")
        return
    
    logger.info(f"Found {len(df_today)} records in parquet")
    
    # Prepare features
    exclude = set(META_COLS + LEAKAGE_COLS + ID_COLS)
    feature_cols = [c for c in df_all.columns if c not in exclude]
    
    X = df_today[feature_cols].copy()
    
    # Convert to numeric
    for c in X.columns:
        if X[c].dtype.name == 'category':
            X[c] = X[c].cat.codes
        elif X[c].dtype == 'object':
            X[c] = pd.Categorical(X[c]).codes
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(-999)
    
    # Align to model features
    for c in expected_features:
        if c not in X.columns:
            X[c] = -999.0
    X = X.reindex(columns=expected_features)
    
    # Convert to numpy for prediction
    X_np = X.values.astype(np.float64)
    
    # Predict
    logger.info("Predicting...")
    raw_preds = model.predict(X_np)
    
    # Store raw score for EV calculation (matches backtest)
    df_today = df_today.copy()
    df_today['raw_score'] = raw_preds
    
    # Convert to probability for display
    # preds = 1 / (1 + np.exp(-raw_preds))  # Sigmoid
    preds = raw_preds # Model output is already probability (LGBM Booster)
    df_today['pred_prob'] = preds
    
    logger.info(f"Raw score range: min={raw_preds.min():.4f}, max={raw_preds.max():.4f}")
    logger.info(f"Prob range: min={preds.min():.4f}, max={preds.max():.4f}")
    
    # Normalize per race (sum to 100%) for ranking/display only
    df_today['pred_prob_norm'] = df_today.groupby('race_id')['pred_prob'].transform(lambda x: x / x.sum())
    
    # Apply calibration if available
    if calibrator is not None:
        df_today['pred_calib'] = calibrator.predict(df_today['pred_prob_norm'].values)
        logger.info(f"Calibrated prob range: min={df_today['pred_calib'].min():.4f}, max={df_today['pred_calib'].max():.4f}")
    else:
        df_today['pred_calib'] = df_today['pred_prob_norm']
    
    # Generate output
    output_lines = []
    output_lines.append(f"🏇 競馬AI予測結果 ({target_date})")
    output_lines.append(f"📊 Optuna最適化モデル | EV閾値: {EV_THRESHOLD}")
    output_lines.append("=" * 50)
    
    bet_count = 0
    
    for race_id in race_ids:
        race_id = str(race_id)
        df_race = df_today[df_today['race_id'] == race_id].copy()
        
        if df_race.empty:
            continue
        
        # Get real-time odds
        odds = get_realtime_odds(loader, race_id)
        names = get_horse_names(loader, race_id)
        
        # Add odds and EV (use raw_score for EV like backtest)
        df_race['horse_number'] = df_race['horse_number'].astype(int)
        df_race['odds'] = df_race['horse_number'].map(odds)
        df_race['ev'] = df_race['pred_prob_norm'] * df_race['odds']  # Use normalized prob for consistency
        df_race['horse_name'] = df_race['horse_number'].map(names)
        
        # Sort by raw score (not normalized)
        df_race = df_race.sort_values('raw_score', ascending=False)
        df_race['pred_rank'] = range(1, len(df_race) + 1)
        
        venue = VENUE_MAP.get(race_id[4:6], '??')
        race_num = int(race_id[-2:])
        
        output_lines.append(f"\n📍 {venue} {race_num}R")
        output_lines.append("-" * 40)
        output_lines.append(f"{'馬番':>3} {'馬名':<10} {'Raw%':>6} {'Cal%':>6} {'ｵｯｽﾞ':>6} {'EV':>5}")
        
        # Show all horses
        for _, row in df_race.iterrows():
            name = str(row['horse_name'])[:10] if pd.notna(row['horse_name']) else '???'
            odds_str = f"{row['odds']:.1f}" if pd.notna(row['odds']) else "---"
            ev_str = f"{row['ev']:.2f}" if pd.notna(row['ev']) else "---"
            # Display both raw and calibrated probability
            calib_str = f"{row['pred_calib']:>5.1%}" if pd.notna(row.get('pred_calib')) else "---"
            output_lines.append(f"{row['horse_number']:>3} {name:<10} {row['pred_prob_norm']:>5.1%} {calib_str:>6} {odds_str:>6} {ev_str:>5}")
        
        # Betting recommendation (use raw_score for prob threshold)
        top1 = df_race.iloc[0]
        if pd.notna(top1['ev']) and top1['raw_score'] >= PROB_THRESHOLD and top1['ev'] >= EV_THRESHOLD:
            output_lines.append(f"\n🎯 推奨: {int(top1['horse_number'])}番 {top1['horse_name']} (EV={top1['ev']:.2f})")
            bet_count += 1
        else:
            output_lines.append("\n⏸️ 条件不満足")
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append(f"📈 推奨ベット: {bet_count}/{len(race_ids)}")
    
    output = "\n".join(output_lines)
    print(output)
    
    # Discord
    if args.discord and args.webhook_url:
        send_discord(args.webhook_url, output)
        logger.info("📤 Sent to Discord")


if __name__ == "__main__":
    main()
