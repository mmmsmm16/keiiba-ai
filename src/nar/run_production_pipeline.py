import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging

# srcディレクトリ（プロジェクトルート）をパスに追加して src.xxx をインポート可能にする
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import lightgbm as lgb
import datetime
try:
    from src.nar.settle_paper_trades import VENUE_MAP
except ImportError:
    VENUE_MAP = {} # Fallback

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
BASE_DATA_PATH = "data/nar/preprocessed_data_south_kanto.pkl" # pkl or parquet? Parquet in task instructions
BASE_DATA_PATH_PARQUET = "data/nar/preprocessed_data_south_kanto.parquet"
MODEL_PATH = "models/production/nar/v2_south_kanto.txt"  # v2 with calibration
CALIBRATOR_PATH = "models/production/nar/v2_calibrator.pkl"
OUTPUT_DIR = "reports/nar/orders"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_markdown_report(df_daily, bets, date_str, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# NAR AI Prediction Report {date_str}\n\n")
        n_races = bets['race_id'].nunique() if not bets.empty else 0
        total_in = bets['amount'].sum() if not bets.empty else 0
        f.write(f"- Target Races: {df_daily['race_id'].nunique()}\n")
        f.write(f"- Bet Races: {n_races}\n")
        f.write(f"- Total Investment: {total_in:,} JPY\n\n")

        races = sorted(df_daily['race_id'].unique().tolist())
        for rid in races:
            rid = str(rid)
            rows = df_daily[df_daily['race_id'] == rid]
            if rows.empty: continue
            r_data = rows.iloc[0]
            
            # Identify columns
            title = r_data.get('race_name') or r_data.get('title') or f"Race {r_data.get('race_number')}"
            venue_code = str(r_data.get('venue', '')).zfill(2)
            venue = VENUE_MAP.get(venue_code, venue_code)
            rnum = r_data.get('race_number', 0)

            f.write(f"## {venue} {rnum}R {title}\n")

            # Top Horses
            f.write("### Top Horses (Score/Prob)\n")
            f.write("| Horse# | Name | Score/Prob |\n")
            f.write("| --- | --- | --- |\n")
            r_horses = rows.sort_values('pred_prob', ascending=False).head(5)
            for _, h in r_horses.iterrows():
                hname = h.get('horse_name', 'Unknown')
                hnum = int(h.get('horse_number', 0))
                prob = h.get('pred_prob', 0.0)
                f.write(f"| {hnum} | {hname} | {prob:.4f} |\n")
            f.write("\n")

            # Bets
            r_bets = bets[bets['race_id'] == rid] if not bets.empty else pd.DataFrame()
            if not r_bets.empty:
                f.write("### Recommended Bets\n")
                f.write("| Type | Combo | Odds | EV | Amount |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                for _, b in r_bets.iterrows():
                    ttype = b['ticket_type']
                    combo = b['combination']
                    odds = b['odds']
                    ev = b['ev']
                    amt = b['amount']
                    f.write(f"| {ttype} | {combo} | {odds} | {ev:.2f} | {amt:,} |\n")
            else:
                 f.write("(No Bets)\n")
            f.write("\n---\n\n")
    logger.info(f"Report saved to {output_path}")

def run_pipeline(date_str, dry_run=False):
    logger.info(f"Starting NAR Production Pipeline for {date_str}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    if os.path.exists(BASE_DATA_PATH_PARQUET):
        logger.info(f"Loading data from {BASE_DATA_PATH_PARQUET}...")
        df = pd.read_parquet(BASE_DATA_PATH_PARQUET)
    else:
        logger.error(f"Data file not found: {BASE_DATA_PATH_PARQUET}")
        return

    # Filter Date
    df['date'] = pd.to_datetime(df['date'])
    daily_df = df[df['date'] == date_str].copy()
    
    if daily_df.empty:
        logger.warning(f"No races found for {date_str}. (Data range: {df['date'].min()} - {df['date'].max()})")
        return
    
    logger.info(f"Target Races: {daily_df['race_id'].nunique()} races, {len(daily_df)} entries")
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
        
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    # Load Calibrator (v2)
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        import joblib
        calibrator = joblib.load(CALIBRATOR_PATH)
        logger.info(f"Loaded calibrator from {CALIBRATOR_PATH}")
    
    # 3. Predict
    # Drop non-features
    drop_cols = [
        'race_id', 'date', 'title', 'horse_id', 'horse_name',
        'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
        'rank', 'target', 'rank_str',
        'time', 'raw_time', 'passing_rank', 'last_3f',
        'odds', 'popularity', 'weight', 'weight_diff_val', 'weight_diff_sign',
        'winning_numbers', 'payout', 'ticket_type',
        'pass_1', 'pass_2', 'pass_3', 'pass_4',
        
        # Leakage/Unused features in Dataset.py
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'track_bias_disadvantage', 'outer_frame_disadv',
        'odds_race_rank', 'popularity_race_rank',
        'odds_deviation', 'popularity_deviation',
        'trend_win_inner_rate', 'trend_win_mid_rate', 'trend_win_outer_rate',
        'trend_win_front_rate', 'trend_win_fav_rate',
        'lag1_odds', 'lag1_popularity',
        'time_index', 'last_3f_index'
    ]
    
    # Select numeric only as per training
    X = daily_df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(exclude=['object'])
    
    logger.info(f"Features: {X.shape[1]}")
    
    # Check features match model
    model_features = model.feature_name()
    missing_cols = set(model_features) - set(X.columns)
    if missing_cols:
        logger.warning(f"Missing features in input: {missing_cols}")
        for c in missing_cols:
            X[c] = 0
            
    # Reorder
    X = X[model_features]
    
    # Force all features to float64 to avoid categorical mismatch errors
    # Only convert numeric columns, drop any remaining object/category columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    for col in X.columns:
        if col not in numeric_cols:
            # Replace non-numeric with 0 or convert category to codes
            if X[col].dtype.name == 'category':
                X[col] = X[col].cat.codes.astype('float64')
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X = X.astype('float64')
    
    preds = model.predict(X)
    daily_df['pred_prob'] = preds
    
    # ==================================================================================
    # 3. Betting Strategy (JRA-aligned: Score -> Prob -> EV -> Optimize)
    # ==================================================================================
    logger.info("Starting Betting Strategy...")
    # Alias for JRA code compatibility
    df_daily = daily_df

    # Load modules dynamically
    from src.probability.ticket_probabilities import compute_ticket_probs
    from src.tickets.generate_candidates import generate_candidates
    from src.backtest.portfolio_optimizer import optimize_bets
    
    # Load Odds from nvd_o1/o2 (proper pre-race odds for all bet types)
    from src.nar.odds_loader import load_odds_from_db
    from src.nar.settle_paper_trades import VENUE_MAP
    race_ids = [str(rid) for rid in df_daily['race_id'].unique().tolist()]
    odds_map = load_odds_from_db(date_str, race_ids=race_ids)
    
    import yaml
    config_path = 'config/nar_policy.yaml'  # NAR専用設定
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        policy = full_config.get('betting', {}).get('parameters', {})
        safety = full_config.get('safety', {})
        # Merge safety settings into policy
        policy['budget'] = safety.get('max_budget_per_race', 3000)
        policy['kelly_fraction'] = safety.get('kelly_fraction', 0.05)
        if 'budget' not in policy: policy['budget'] = 3000
        if 'kelly_fraction' not in policy: policy['kelly_fraction'] = 0.05
        logger.info(f"Loaded NAR policy from {config_path}: {policy}")
    else:
        # Fallback default policy
        policy = {
            'budget': 10000,
            'kelly_fraction': 0.1,
            'min_ev_threshold': 0.8, # Loose threshold for test
            'min_bet_amount': 100,
            'base_stake': 10000 
        }
        logger.warning(f"Config not found at {config_path}. Using default policy: {policy}")
    
    all_orders = []
    
    for rid in race_ids:
        # Filter race data
        r_df = df_daily[df_daily['race_id'] == rid].copy()
        if r_df.empty: continue
        
        # 3a. Compute Probabilities (Softmax/Simulation)
        # compute_ticket_probs expects 'horse_number', 'frame_number', 'pred_prob' (as score)
        # Note: If pred_prob is raw lambda rank score, compute_ticket_probs treats it correctly.
        try:
            prob_dict = compute_ticket_probs(r_df, strength_col='pred_prob', n_samples=5000)
            
            # Apply Calibration (v2) - Correct the softmax probabilities
            if calibrator is not None and 'win' in prob_dict:
                win_probs = prob_dict['win'].values
                calibrated_win_probs = calibrator.predict(win_probs)
                prob_dict['win'] = pd.Series(calibrated_win_probs, index=prob_dict['win'].index)
                
                # Also calibrate place probabilities if available
                if 'place' in prob_dict:
                    place_probs = prob_dict['place'].values
                    calibrated_place_probs = calibrator.predict(place_probs)
                    prob_dict['place'] = pd.Series(calibrated_place_probs, index=prob_dict['place'].index)
                    
        except Exception as e:
            logger.error(f"Prob calc failed for {rid}: {e}")
            continue
            
        # 3b. Generate Candidates
        candidates = generate_candidates(rid, prob_dict)
        if candidates.empty: continue
        
        # 3c. Merge Odds & Compute EV
        # Use odds_map from nvd_o1/o2 (contains WIN, PLACE, UMAREN for all horses/combos)
        r_odds = odds_map.get(str(rid), {})
        
        # Flatten odds to DataFrame
        odds_records = []
        for ttype, combos in r_odds.items():
            for combo, odds_val in combos.items():
                odds_records.append({
                   'ticket_type': ttype.lower(), # generate_candidates uses lowercase 'win', 'place', 'umaren'
                   'combination': str(combo),
                   'odds': odds_val
                })
        
        if not odds_records:
            logger.warning(f"No odds data for {rid}")
            continue
            
        df_odds = pd.DataFrame(odds_records)
        
        # Merge
        # candidate: race_id, ticket_type, combination, p_ticket
        # Inner join will naturally filter out Ticket Types we don't have odds for (Place, Umaren)
        merged = pd.merge(candidates, df_odds, on=['ticket_type', 'combination'], how='inner')
        
        if merged.empty:
            continue
            
        merged['ev'] = merged['p_ticket'] * merged['odds']
        
        # 3d. Optimize
        # Filter for WIN only for now if desired, or allow all
        # merged = merged[merged['ticket_type'] == 'win']
        
        bets = optimize_bets(merged, policy)
        
        if not bets.empty:
            bets['race_id'] = rid
            all_orders.append(bets)

    if not all_orders:
        logger.warning("No orders generated.")
        orders_df = pd.DataFrame(columns=['race_id', 'ticket_type', 'combination', 'amount', 'odds', 'ev', 'p_ticket'])
    else:
        orders_df = pd.concat(all_orders, ignore_index=True)

    # Save Orders csv
    output_file = os.path.join(OUTPUT_DIR, f"{date_str.replace('-','')}_orders.csv")
    orders_df.to_csv(output_file, index=False)
    
    # Save Ledger parquet (for settlement) (Adding extra cols for debug)
    ledger_dir = "reports/nar/ledgers"
    os.makedirs(ledger_dir, exist_ok=True)
    ledger_file = os.path.join(ledger_dir, f"{date_str.replace('-','')}_ledger.parquet")
    orders_df.to_parquet(ledger_file, index=False)
    
    logger.info(f"Generated {len(orders_df)} orders in {output_file}")
    logger.info(f"Generated ledger in {ledger_file}")
    
    # Save Full Predictions (for Settlement Report)
    pred_parquet_dir = "reports/nar/predictions"
    os.makedirs(pred_parquet_dir, exist_ok=True)
    pred_parquet_file = os.path.join(pred_parquet_dir, f"{date_str.replace('-','')}_predictions.parquet")
    
    # Save relevant columns only to save space
    pred_cols = [
        'race_id', 'horse_number', 'horse_name', 'pred_prob', 
        'race_number', 'venue', 'title', 'n_horses'
        # Add others if needed for report like ninki/odds if available in daily_df?
        # daily_df comes from preprocessed, has 'odds', 'ninki'? 
        # Yes, if valid/test mode. In production, unseen.
        # But for backtest/paper trade, they exist.
    ]
    # Filter only existing cols
    save_cols = [c for c in pred_cols if c in df_daily.columns]
    df_pred_save = df_daily[save_cols].copy()
    df_pred_save.to_parquet(pred_parquet_file, index=False)
    logger.info(f"Saved predictions to {pred_parquet_file}")

    print(orders_df.head())

    # Generate Markdown Report (JRA parity)
    report_dir = "reports/nar/predictions"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"{date_str.replace('-','')}_prediction.md")
    generate_markdown_report(df_daily, orders_df, date_str, report_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('date', type=str, help='Target Date (YYYY-MM-DD)')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    run_pipeline(args.date, args.dry_run)
