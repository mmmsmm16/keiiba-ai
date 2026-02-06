
import pandas as pd
import numpy as np
import argparse
import os
import sys
import yaml
import logging
import datetime
import lightgbm as lgb
import joblib

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.backtest.portfolio_optimizer import optimize_bets
    from src.probability.ticket_probabilities import compute_ticket_probs
    from src.tickets.generate_candidates import generate_candidates
    from src.odds.pckeiba_loader import PCKeibaLoader
except ImportError:
    pass # Handle logic later

# Venue Map for Report
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
    # Numeric without zero
    '1': '札幌', '2': '函館', '3': '福島', '4': '新潟', '5': '東京', 
    '6': '中山', '7': '中京', '8': '京都', '9': '阪神',
    # Alphabet Codes (if any)
    'SAP': '札幌', 'HAK': '函館', 'FUK': '福島', 'NII': '新潟', 'TOK': '東京',
    'NAK': '中山', 'CHU': '中京', 'KYO': '京都', 'HAN': '阪神', 'KOK': '小倉'
}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_guardrails(odds_df, date):
    """
    Validate input data before processing.
    """
    # 1. Timestamp Check
    # Ensure no odds from future (should be impossible if loaded from history, but useful for live)
    # Placeholder for logic
    pass

    # 2. Missing Columns
    required = ['race_id', 'odds', 'ticket_type', 'combination']
    missing = [c for c in required if c not in odds_df.columns]
    if missing:
        raise ValueError(f"CRITICAL: Missing columns in odds data: {missing}")

def run_pipeline(date_str, mode='paper', force=False):
    logger.info(f"Starting Production Pipeline for {date_str} (Mode: {mode})")
    
    # Paths
    config_path = 'config/production_policy.yaml'
    config = load_config(config_path)
    
    output_dir = 'outputs/orders'
    ledger_dir = 'outputs/ledgers'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ledger_dir, exist_ok=True)
    
    order_file = f"{output_dir}/{date_str.replace('-','')}_orders.csv"
    ledger_file = f"{ledger_dir}/{date_str.replace('-','')}_ledger.parquet"
    
    # Idempotency Check
    if os.path.exists(order_file) and not force:
        logger.error(f"Order file {order_file} already exists. Use --force to overwrite.")
        sys.exit(1)
        

    
    target_date = pd.Timestamp(date_str)
    year = target_date.year
    
    # 1. Load Data
    logger.info(f"Loading data for {date_str}...")
    
    # Base Features
    # In production/paper, we load from the 'processed' parquet which contains history.
    base_data_path = 'data/processed/preprocessed_data.parquet'
    if not os.path.exists(base_data_path):
        logger.error(f"Base data not found: {base_data_path}")
        sys.exit(1)
        
    df_base = pd.read_parquet(base_data_path)
    df_base['date'] = pd.to_datetime(df_base['date'])
    df_daily = df_base[df_base['date'] == date_str].copy()
    
    if df_daily.empty:
        logger.error(f"No races found for {date_str}")
        sys.exit(0)
        
    logger.info(f"Found {len(df_daily)} entries (horses) for {date_str}")
    
    # Ensure Categorical Types (Same as Training)
    cat_cols = df_daily.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        df_daily[c] = df_daily[c].astype('category')
    
    # Odds Features (T-60, T-30, T-10)
    # We use the existing function but restrict year to current year
    from src.features.odds_movement_features import calculate_odds_movement_features
    
    # This loads the whole year, which is heavy but safe for now.
    # TODO: Optimize for single day
    odds_feat_df = calculate_odds_movement_features(None, start_year=year, end_year=year)
    
    # Merge Odds Features
    df_daily['race_id'] = df_daily['race_id'].astype(str)
    odds_feat_df['race_id'] = odds_feat_df['race_id'].astype(str)
    
    df_daily = pd.merge(df_daily, odds_feat_df, on=['race_id', 'horse_number'], how='left')
    
    # 2. Load Model & Predict
    model_path = config['model']['path']
    feature_list_path = config['model']['feature_config']
    
    if not os.path.exists(feature_list_path):
         # Fallback check
         fallback = model_path.replace('_model.txt', '_feature_list.joblib')
         if os.path.exists(fallback):
             feature_list_path = fallback
         else:
             # Try simpler name
             fallback = "models/production/v13_feature_list.joblib"
             if os.path.exists(fallback):
                 feature_list_path = fallback
         
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run Task 13.1.")
        sys.exit(1)
        
    logger.info(f"Loading Model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    
    logger.info(f"Loading features from {feature_list_path}...")
    if not os.path.exists(feature_list_path):
        logger.error(f"Feature list not found at {feature_list_path}")
        sys.exit(1)
        
    features = joblib.load(feature_list_path)
    
    # Check missing cols
    missing_feats = [c for c in features if c not in df_daily.columns]
    if missing_feats:
        logger.error(f"Missing features in daily data: {missing_feats[:5]}...")
        # Guardrail: Fail if features missing
        sys.exit(1)
        
    X = df_daily[features]
    pred_prob = model.predict(X)
    df_daily['pred_prob'] = pred_prob
    logger.info(f"Predictions generated. Mean: {pred_prob.mean():.4f}, Max: {pred_prob.max():.4f}, Min: {pred_prob.min():.4f}")
    
    
    # Load T-10 Odds for Decision (Payout/Policy)
    logger.info("Loading T-10 Odds for Betting...")
    
    odds_t10_all = pd.DataFrame()
    
    if mode == 'live':
        logger.info("Mode is LIVE: Fetching from PC-Keiba DB...")
        loader = PCKeibaLoader()
        
        # Build Race ID Map from df_daily
        # Key: (Year, Month, Day, Venue, RaceNum) -> Value: RaceID
        race_id_map = {}
        for _, row in df_daily.iterrows():
            try:
                dt = row['date'] # datetime
                y, m, d = dt.year, dt.month, dt.day
                v = str(row['venue']).zfill(2)
                r = int(row['race_number'])
                race_id_map[(y, m, d, v, r)] = str(row['race_id'])
            except Exception as e:
                logger.warning(f"Failed to map race: {row.get('race_id')} - {e}")

        try:
            odds_t10_all = loader.get_latest_odds(date_str, race_id_map=race_id_map)
            logger.info(f"Fetched {len(odds_t10_all)} odds records from DB.")
            if odds_t10_all.empty:
                logger.warning("No odds found in DB! Check pckeiba_loader or DB connection.")
                # Fallback or Exit?
                # In live mode, empty odds means no betting.
                
        except Exception as e:
            logger.error(f"Critical Error fetching live odds: {e}")
            sys.exit(1)
            
    else:
        # Paper Mode: Try Snapshot first, then fallback to DB
        odds_t10_path = f"data/odds_snapshots/{year}/odds_T-10.parquet"
        odds_t10_all = pd.DataFrame()
        races = [str(r) for r in df_daily['race_id'].unique()]  # Define upfront
        
        if os.path.exists(odds_t10_path):
            odds_t10_all = pd.read_parquet(odds_t10_path)
            odds_t10_all['race_id'] = odds_t10_all['race_id'].astype(str)
            
            # Check if races for target date are in snapshot
            odds_for_date = odds_t10_all[odds_t10_all['race_id'].isin(races)]
            
            if odds_for_date.empty:
                logger.warning(f"No odds in snapshot for {date_str}. Trying DB fallback...")
                odds_t10_all = pd.DataFrame()  # Reset to trigger DB fallback
        
        # DB Fallback for paper mode
        if odds_t10_all.empty or len(odds_t10_all[odds_t10_all['race_id'].isin(races)]) == 0:
            logger.info("Fetching odds from DB...")
            try:
                loader = PCKeibaLoader()
                
                # Build Race ID Map from df_daily
                race_id_map = {}
                for _, row in df_daily.iterrows():
                    try:
                        dt = row['date']
                        y, m, d = dt.year, dt.month, dt.day
                        v = str(row['venue']).zfill(2)
                        r = int(row['race_number'])
                        race_id_map[(y, m, d, v, r)] = str(row['race_id'])
                    except:
                        pass
                
                odds_t10_all = loader.get_latest_odds(date_str, race_id_map=race_id_map)
                logger.info(f"Fetched {len(odds_t10_all)} odds records from DB.")
                
            except Exception as e:
                logger.error(f"Failed to fetch odds from DB: {e}")
                sys.exit(1)
    
    if odds_t10_all.empty:
         logger.error("No odds data available.")
         sys.exit(1)

    if not odds_t10_all.empty:
        odds_t10_all['race_id'] = odds_t10_all['race_id'].astype(str)
        odds_t10_all['combination'] = odds_t10_all['combination'].astype(str)
    
    # Validation: Guardrails on Odds
    # Note: check_guardrails needs race_dt? or just current timestamp check?
    # Assuming logic is robust or minimal.
    check_guardrails(odds_t10_all, target_date)
    
    # Filter for today's races (optimization)
    # odds_t10_all usually doesn't have date. Filter by race_id.
    races = df_daily['race_id'].astype(str).unique()
    
    # Ensure race_id type match
    odds_t10_all['race_id'] = odds_t10_all['race_id'].astype(str)
    
    odds_t10_day = odds_t10_all[odds_t10_all['race_id'].isin(races)].copy()
    
    logger.info(f"Races in df_daily: {len(races)}")
    logger.info(f"Odds matched for daily races: {len(odds_t10_day)}")
    if odds_t10_day.empty:
         logger.warning(f"No odds matched! Sample races: {races[:3]}")
         logger.warning(f"Sample odds race_ids: {odds_t10_all['race_id'].unique()[:3]}")

    
    # 3. Probability Engine (Plackett-Luce) & Candidate Gen
    # Need to group by race
    # races already defined
    
    all_bets = []
    
    logger.info("Generating Bets...")
    
    policy = config['betting']['parameters']
    logger.info(f"Betting Policy: {policy}")
    
    for i, rid in enumerate(races):
        rid = str(rid)
        # Filter data for this race
        pdf = df_daily[df_daily['race_id'] == rid].copy()
        
        # Guardrail: Check if race has enough horses
        if len(pdf) < 5:
            # logger.warning(f"Race {rid} has too few horses ({len(pdf)}). Skipping.")
            continue
            
        try:
            # 3a. Ticket Probs
            # Ensure types for prob engine
            pdf['horse_number'] = pd.to_numeric(pdf['horse_number'], errors='coerce').fillna(0).astype(int)
            pdf['frame_number'] = pd.to_numeric(pdf['frame_number'], errors='coerce').fillna(0).astype(int)
            
            # Use 'pred_prob' as strength
            # config parameter n_samples
            n_samples = 5000 
            # Could read from config['betting']['parameters'].get('n_samples', 5000)
            
            probs = compute_ticket_probs(pdf, strength_col='pred_prob', n_samples=n_samples)
            
            # 3b. Candidates
            candidates = generate_candidates(rid, probs)
            if candidates.empty:
                if i < 3: logger.warning(f"Race {rid}: No candidates generated.")
                continue
                
            # 3c. Join Odds (T-10)
            # Filter odds for this race
            rodds = odds_t10_day[odds_t10_day['race_id'] == rid].copy()
            
            # Filter by allowed bet_types
            allowed_types = config['model'].get('bet_types', [])
            
            # --- SYNTHETIC ODDS GENERATION ---
            # Check if we have Win odds to base on
            win_odds = rodds[rodds['ticket_type'] == 'win']
            if not win_odds.empty:
                from src.odds.synthetic_odds import SyntheticOddsGenerator
                gen = SyntheticOddsGenerator(win_odds)
                
                # Types we want to synthesize if missing
                # (User requested Wide, Sanrenpuku, Sanrentan)
                target_synthetic = ['wide', 'sanrenpuku', 'sanrentan']
                
                new_odds_frames = []
                for ttype in target_synthetic:
                    # Only generate if permissible by config AND missing in real data
                    # If allowed_types is restricted, respect it.
                    # But if allowed_types is empty (allow all), then generate.
                    if allowed_types and ttype not in allowed_types:
                        continue
                        
                    # Check if already present
                    if rodds[rodds['ticket_type'] == ttype].empty:
                        syn_df = gen.get_odds(ttype)
                        if not syn_df.empty:
                            syn_df['race_id'] = rid
                            # Add ninki column (dummy or derived?)
                            # Synthetic odds imply ninki order roughly.
                            # We leave ninki as NaN or 9999
                            syn_df['ninki'] = np.nan 
                            new_odds_frames.append(syn_df)
                
                if new_odds_frames:
                    logger.info(f"Race {rid}: Generated synthetic odds types: {[df.iloc[0]['ticket_type'] for df in new_odds_frames]}")
                    rodds = pd.concat([rodds] + new_odds_frames, ignore_index=True)

            if allowed_types:
                rodds = rodds[rodds['ticket_type'].isin(allowed_types)]
            
            if rodds.empty:
                if i < 3: logger.warning(f"Race {rid}: No T-10 odds (after filter). Skipping.")
                continue
                
            merged = pd.merge(
                candidates,
                rodds[['ticket_type', 'combination', 'odds']],
                on=['ticket_type', 'combination'],
                how='inner' # Change to inner to drop candidates without odds (or filtered out)
            )
            
            if merged.empty:
                # if i < 3: logger.warning(f"Race {rid}: Merged bets empty. Cand={len(candidates)}, Odds={len(rodds)}")
                continue

            # Logic: odds are T-10 odds
            merged['ev'] = merged['p_ticket'] * merged['odds']
            merged['odds_t10'] = merged['odds'] # Store explicit T10 column for Audit
            
            # 3d. Optimize
            # Policy from config
            # 'parameters': {kelly_fraction: 0.05, ...}
            df_bets = optimize_bets(merged, policy)
            
            if not df_bets.empty:
                df_bets['race_id'] = rid
                # Add timestamp or date
                df_bets['date'] = str(target_date.date())
                all_bets.append(df_bets)
                
        except Exception as e:
            logger.error(f"Error processing race {rid}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # 4. Concatenate
    if not all_bets:
        logger.warning("No bets generated.")
        sys.exit(0)
        
    final_bets = pd.concat(all_bets, ignore_index=True)
    
    # 5. Safety Cap (Global Budget)
    # config['safety']['max_daily_budget']
    max_budget = config['safety'].get('max_daily_budget', 200000)
    total_amount = final_bets['amount'].sum()
    
    if total_amount > max_budget:
        logger.warning(f"Total budget exceeded ({total_amount} > {max_budget}). Scaling down.")
        scale = max_budget / total_amount
        final_bets['amount'] = (final_bets['amount'] * scale // 100) * 100
        final_bets = final_bets[final_bets['amount'] >= 100]
        
    # Anomaly Detection (Bet Count)
    # e.g. config['safety']['max_bets_per_race'] logic inside optimize_bets or here?
    # Simple warning
    if len(final_bets) > 1000: # Arbitrary high number
        logger.warning(f"High bet count detected: {len(final_bets)}")
        
    # 6. Save Output
    # Orders CSV: Standard Format for Betting Agent
    # race_id, ticket_type, combination, amount
    cols_order = ['race_id', 'ticket_type', 'combination', 'amount', 'odds', 'ev']
    orders_df = final_bets[cols_order].copy()
    
    logger.info(f"Saving {len(orders_df)} orders to {order_file}")
    orders_df.to_csv(order_file, index=False)
    
    # Ledger Parquet: Complete Audit Trail
    # Include output details
    logger.info(f"Saving Ledger to {ledger_file}")
    final_bets['run_id'] = f"{date_str}_{mode}"
    final_bets['created_at'] = pd.Timestamp.now()
    final_bets.to_parquet(ledger_file)
    
    # 7. Save Full Predictions (for Settlement/Analysis)
    pred_file = f"outputs/predictions/{date_str.replace('-','')}_predictions.parquet"
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)
    logger.info(f"Saving Predictions to {pred_file}")
    
    # Select useful columns
    save_cols = [
        'race_id', 'horse_number', 'horse_name', 'pred_prob', 
        'rank', 'odds', 'popularity', 'title', 'venue', 'race_number' # Metadata if available
    ]
    # Filter only existing columns
    save_cols = [c for c in save_cols if c in df_daily.columns]
    df_daily[save_cols].to_parquet(pred_file)
    
    logger.info("Pipeline Completed Successfully.")
    
    # 7. Generate Human Readable Report
    logger.info("Generating Report...")
    report_file = os.path.join("outputs", "reports", f"{date_str.replace('-','')}_prediction.md")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    generate_markdown_report(df_daily, final_bets, date_str, report_file, odds_t10_all)

def generate_markdown_report(df_daily, bets, date_str, output_path, odds_df):
    """
    Generate a markdown report for human consumption.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 競馬AI予測レポート: {date_str}\n\n")
        
        # Summary
        n_races = bets['race_id'].nunique()
        total_in = bets['amount'].sum()
        f.write(f"- 対象レース数: {n_races}\n")
        f.write(f"- 総投資予定額: {total_in:,}円\n\n")
        
        # Group by Race
        races = df_daily['race_id'].unique()
        # Sort by race_id (likely correct order, but can verify)
        # Usually race_id has race_number embedded or sortable
        
        # Map race_id to Title/Number
        # df_daily has 'title', 'race_number', 'venue'
        # Group df_daily by race_id to extract metadata
        
        for rid in sorted(races):
            rid = str(rid)
            r_bets = bets[bets['race_id'] == rid]
            
            # Race Info
            r_data = df_daily[df_daily['race_id'] == rid].iloc[0]
            title = r_data.get('title', 'Unknown Race').strip()
            title = r_data.get('title', 'Unknown Race').strip()
            venue_code = str(r_data.get('venue', '')).zfill(2)
            venue = VENUE_MAP.get(venue_code, venue_code)
            rnum = r_data.get('race_number', 0)
            
            # If no bets, maybe skip or show "No Bet"
            if r_bets.empty:
                # Optional: Show races with no bets? User asked for orders file to be visualized.
                # If orders.csv only has bets, report should probably focus on bets.
                # But showing context is good.
                continue
                
            f.write(f"## {venue} {rnum}R {title}\n")
            
            # Top Horses (by Score/Prob)
            f.write("### 有力馬 (Top 5)\n")
            f.write("| 馬番 | 馬名 | スコア/確率 |\n")
            f.write("| --- | --- | --- |\n")
            
            r_horses = df_daily[df_daily['race_id'] == rid].sort_values('pred_prob', ascending=False).head(5)
            for _, h in r_horses.iterrows():
                hname = h.get('horse_name', 'Unknown')
                hnum = int(h.get('horse_number', 0))
                prob = h.get('pred_prob', 0.0)
                f.write(f"| {hnum} | {hname} | {prob:.4f} |\n")
            
            f.write("\n")
            
            # Bets
            f.write("### 推奨買い目\n")
            f.write("| 券種 | 買い目 | オッズ | 期待値 | 金額 |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            for _, b in r_bets.iterrows():
                ttype = b['ticket_type']
                combo = b['combination']
                odds = b['odds']
                ev = b['ev']
                amt = b['amount']
                f.write(f"| {ttype} | {combo} | {odds} | {ev:.2f} | {amt:,} |\n")
                
            f.write("\n---\n\n")
            
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--mode', type=str, default='paper', choices=['paper', 'live'])
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()
    
    run_pipeline(args.date, args.mode, args.force)
