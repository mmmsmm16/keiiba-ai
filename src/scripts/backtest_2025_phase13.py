"""
2025 Backtest for Phase 13 Production Strategy

This script runs the production pipeline (run_production_pipeline.py logic)
for all 2025 race dates and calculates ROI using settle_paper_trades.py logic.
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
from sqlalchemy import text, create_engine

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venue Map
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}


def get_db_engine():
    """Create database connection."""
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(db_url)


def normalize_combination(combo_str, ticket_type):
    """
    Normalize combination string from DB format (e.g., '0209') to ledger format (e.g., '2-9').
    """
    if not combo_str or pd.isna(combo_str):
        return None
    combo_str = str(combo_str).strip()
    
    if ticket_type in ['win', 'tansho', 'place', 'fukusho']:
        return str(int(combo_str))
    elif ticket_type in ['umaren', 'wide']:
        if len(combo_str) == 4:
            h1, h2 = int(combo_str[:2]), int(combo_str[2:])
            sorted_nums = sorted([h1, h2])
            return f"{sorted_nums[0]}-{sorted_nums[1]}"
        elif '-' in combo_str:
            parts = [int(x) for x in combo_str.split('-')]
            sorted_nums = sorted(parts)
            return f"{sorted_nums[0]}-{sorted_nums[1]}"
    elif ticket_type in ['sanrenpuku']:
        if '-' in combo_str:
            parts = [int(x) for x in combo_str.split('-')]
            sorted_nums = sorted(parts)
            return f"{sorted_nums[0]}-{sorted_nums[1]}-{sorted_nums[2]}"
        elif len(combo_str) == 6: # e.g. 010203?
             # Assuming 2 digits each
             p = [int(combo_str[i:i+2]) for i in range(0, 6, 2)]
             p.sort()
             return f"{p[0]}-{p[1]}-{p[2]}"
    elif ticket_type in ['sanrentan']:
        # Keep order
        if '-' in combo_str:
            parts = [int(x) for x in combo_str.split('-')]
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        elif len(combo_str) == 6:
             p = [int(combo_str[i:i+2]) for i in range(0, 6, 2)]
             return f"{p[0]}-{p[1]}-{p[2]}"
    return combo_str


def load_payout_map_from_db(year):
    """Load all payout data for a year from jvd_hr table."""
    engine = get_db_engine()
    
    query = text(f"""
        SELECT 
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
            haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
            haraimodoshi_fukusho_1a, haraimodoshi_fukusho_1b,
            haraimodoshi_fukusho_2a, haraimodoshi_fukusho_2b,
            haraimodoshi_fukusho_3a, haraimodoshi_fukusho_3b,
            haraimodoshi_umaren_1a, haraimodoshi_umaren_1b, 
            haraimodoshi_umaren_2a, haraimodoshi_umaren_2b, 
            haraimodoshi_umaren_3a, haraimodoshi_umaren_3b,
            haraimodoshi_wide_1a, haraimodoshi_wide_1b,
            haraimodoshi_wide_2a, haraimodoshi_wide_2b,
            haraimodoshi_wide_3a, haraimodoshi_wide_3b,
            haraimodoshi_wide_4a, haraimodoshi_wide_4b,
            haraimodoshi_wide_5a, haraimodoshi_wide_5b,
            haraimodoshi_wide_6a, haraimodoshi_wide_6b,
            haraimodoshi_wide_7a, haraimodoshi_wide_7b,
            haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
            haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
            haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
            haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
            haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
            haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b
        FROM jvd_hr 
        WHERE kaisai_nen = '{year}'
    """)
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} payout records from DB for {year}")
    except Exception as e:
        logger.error(f"Failed to load payout data from DB: {e}")
        return {}
    
    # Build payout map
    payout_map = {}
    
    for _, row in df.iterrows():
        rid = str(row['race_id'])
        if rid not in payout_map:
            payout_map[rid] = {
                'win': {}, 'place': {}, 'umaren': {}, 'wide': {},
                'sanrenpuku': {}, 'sanrentan': {}
            }
        
        def safe_int(val):
            """Safely convert value to int."""
            if val is None or pd.isna(val):
                return None
            if isinstance(val, (int, float)):
                return int(val) if not pd.isna(val) else None
            val_str = str(val).strip()
            if not val_str or not val_str.isdigit():
                return None
            return int(val_str)
        
        # Tansho (win)
        combo = normalize_combination(row.get('haraimodoshi_tansho_1a'), 'win')
        pay = safe_int(row.get('haraimodoshi_tansho_1b'))
        if combo and pay:
            payout_map[rid]['win'][combo] = pay
        
        # Fukusho (place)
        for i in range(1, 4):
            combo = normalize_combination(row.get(f'haraimodoshi_fukusho_{i}a'), 'place')
            pay = safe_int(row.get(f'haraimodoshi_fukusho_{i}b'))
            if combo and pay:
                payout_map[rid]['place'][combo] = pay
        
        # Umaren
        for i in range(1, 4):
            combo = normalize_combination(row.get(f'haraimodoshi_umaren_{i}a'), 'umaren')
            pay = safe_int(row.get(f'haraimodoshi_umaren_{i}b'))
            if combo and pay:
                payout_map[rid]['umaren'][combo] = pay
        
        # Wide
        for i in range(1, 8):
            combo = normalize_combination(row.get(f'haraimodoshi_wide_{i}a'), 'wide')
            pay = safe_int(row.get(f'haraimodoshi_wide_{i}b'))
            if combo and pay:
                payout_map[rid]['wide'][combo] = pay

        # Sanrenpuku
        for i in range(1, 4):
            combo = normalize_combination(row.get(f'haraimodoshi_sanrenpuku_{i}a'), 'sanrenpuku')
            pay = safe_int(row.get(f'haraimodoshi_sanrenpuku_{i}b'))
            if combo and pay:
                payout_map[rid]['sanrenpuku'][combo] = pay

        # Sanrentan
        for i in range(1, 4):
            combo = normalize_combination(row.get(f'haraimodoshi_sanrentan_{i}a'), 'sanrentan')
            pay = safe_int(row.get(f'haraimodoshi_sanrentan_{i}b'))
            if combo and pay:
                payout_map[rid]['sanrentan'][combo] = pay
    
    return payout_map


def run_backtest_2025():
    """Run Phase 13 backtest for 2025."""
    logger.info("=" * 60)
    logger.info("Phase 13 Production Strategy - 2025 Backtest")
    logger.info("=" * 60)
    
    # Load preprocessed data
    data_path = 'data/processed/preprocessed_data_v11.parquet'
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df_2025 = df[df['date'].dt.year == 2025].copy()
    
    if df_2025.empty:
        logger.error("No 2025 data found")
        return
    
    # Get unique dates
    dates = sorted(df_2025['date'].unique())
    logger.info(f"Found {len(dates)} race dates in 2025")
    
    # Load payout map from DB
    payout_map = load_payout_map_from_db(2025)
    if not payout_map:
        logger.error("No payout data available")
        return
    
    logger.info(f"Loaded payout data for {len(payout_map)} races")
    
    # Load T-10 odds
    odds_t10_path = 'data/odds_snapshots/2025/odds_T-10.parquet'
    if not os.path.exists(odds_t10_path):
        logger.error(f"T-10 odds not found: {odds_t10_path}")
        return
    
    odds_t10 = pd.read_parquet(odds_t10_path)
    odds_t10['race_id'] = odds_t10['race_id'].astype(str)
    odds_t10['combination'] = odds_t10['combination'].astype(str)
    
    # Load model
    import lightgbm as lgb
    import joblib
    
    model_path = 'models/production/v13_production_model.txt'
    feature_path = 'models/production/v13_feature_list.joblib'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    model = lgb.Booster(model_file=model_path)
    features = joblib.load(feature_path)
    
    # Import probability and optimization modules
    try:
        from src.backtest.portfolio_optimizer import optimize_bets
        from src.probability.ticket_probabilities import compute_ticket_probs
        from src.tickets.generate_candidates import generate_candidates
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        return
    
    # Load production policy
    import yaml
    with open('config/production_policy.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    policy = config['betting']['parameters']
    allowed_types = config['model'].get('bet_types', [])
    
    # Results tracking
    total_invest = 0
    total_return = 0
    race_count = 0
    bet_count = 0
    hit_count = 0
    
    daily_results = []
    
    # Process each date
    for date in tqdm(dates, desc="Processing dates"):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        day_df = df_2025[df_2025['date'] == date].copy()
        
        if day_df.empty:
            continue
        
        # Ensure categorical types
        cat_cols = day_df.select_dtypes(include=['object', 'category']).columns.tolist()
        for c in cat_cols:
            day_df[c] = day_df[c].astype('category')
        
        # Check missing features
        missing_feats = [c for c in features if c not in day_df.columns]
        if missing_feats:
            for c in missing_feats:
                day_df[c] = 0
        
        # Predict
        X = day_df[features]
        day_df['pred_prob'] = model.predict(X)
        day_df['race_id'] = day_df['race_id'].astype(str)
        
        # Filter T-10 odds for today
        races = day_df['race_id'].unique()
        day_odds = odds_t10[odds_t10['race_id'].isin(races)].copy()
        
        day_invest = 0
        day_return = 0
        
        # Process each race
        for rid in races:
            rid = str(rid)
            pdf = day_df[day_df['race_id'] == rid].copy()
            
            if len(pdf) < 5:
                continue
            
            try:
                # Compute ticket probabilities
                pdf['horse_number'] = pd.to_numeric(pdf['horse_number'], errors='coerce').fillna(0).astype(int)
                pdf['frame_number'] = pd.to_numeric(pdf['frame_number'], errors='coerce').fillna(0).astype(int)
                
                probs = compute_ticket_probs(pdf, strength_col='pred_prob', n_samples=5000)
                
                # Generate candidates
                candidates = generate_candidates(rid, probs)
                if candidates.empty:
                    continue
                
                # Join T-10 odds
                rodds = day_odds[day_odds['race_id'] == rid].copy()
                # --- SYNTHETIC ODDS GENERATION ---
                win_odds = rodds[rodds['ticket_type'] == 'win']
                if not win_odds.empty:
                    from src.odds.synthetic_odds import SyntheticOddsGenerator
                    gen = SyntheticOddsGenerator(win_odds)
                    target_synthetic = ['wide', 'sanrenpuku', 'sanrentan']
                    new_odds_frames = []
                    for ttype in target_synthetic:
                         if allowed_types and ttype not in allowed_types: continue
                         if rodds[rodds['ticket_type'] == ttype].empty:
                             syn_df = gen.get_odds(ttype)
                             if not syn_df.empty:
                                 syn_df['race_id'] = rid
                                 syn_df['ninki'] = np.nan
                                 new_odds_frames.append(syn_df)
                    if new_odds_frames:
                        rodds = pd.concat([rodds] + new_odds_frames, ignore_index=True)

                if allowed_types:
                    rodds = rodds[rodds['ticket_type'].isin(allowed_types)]
                
                if rodds.empty:
                    continue
                
                merged = pd.merge(
                    candidates,
                    rodds[['ticket_type', 'combination', 'odds']],
                    on=['ticket_type', 'combination'],
                    how='inner'
                )

                
                if merged.empty:
                    continue
                
                merged['ev'] = merged['p_ticket'] * merged['odds']
                
                # Optimize bets
                df_bets = optimize_bets(merged, policy)
                
                if df_bets.empty:
                    continue
                
                # Calculate payout
                race_payouts = payout_map.get(rid, {})
                
                # Build rank map
                h2r = dict(zip(pdf['horse_number'], pdf['rank']))
                
                for _, bet in df_bets.iterrows():
                    ttype = bet['ticket_type']
                    combo = str(bet['combination'])
                    amt = bet['amount']
                    
                    day_invest += amt
                    bet_count += 1
                    
                    # Check hit
                    hit = False
                    if ttype == 'win':
                        if h2r.get(int(combo), 99) == 1:
                            hit = True
                    elif ttype == 'place':
                        if h2r.get(int(combo), 99) <= 3:
                            hit = True
                    elif ttype == 'umaren':
                        p = [int(x) for x in combo.split('-')]
                        if h2r.get(p[0], 99) <= 2 and h2r.get(p[1], 99) <= 2:
                            hit = True
                    elif ttype == 'wide':
                        p = [int(x) for x in combo.split('-')]
                        if h2r.get(p[0], 99) <= 3 and h2r.get(p[1], 99) <= 3:
                            hit = True
                    elif ttype == 'sanrenpuku':
                        p = [int(x) for x in combo.split('-')]
                        # All 3 must be in top 3
                        if all(h2r.get(x, 99) <= 3 for x in p):
                            hit = True
                    elif ttype == 'sanrentan':
                        p = [int(x) for x in combo.split('-')]
                        # Exact order: 1st, 2nd, 3rd
                        if h2r.get(p[0], 99) == 1 and h2r.get(p[1], 99) == 2 and h2r.get(p[2], 99) == 3:
                            hit = True
                    
                    if hit:
                        ticket_payouts = race_payouts.get(ttype, {})
                        
                        # Normalize combination
                        if '-' in combo and ttype in ['umaren', 'wide']:
                            parts = [int(x) for x in combo.split('-')]
                            combo_normalized = f"{sorted(parts)[0]}-{sorted(parts)[1]}"
                        else:
                            combo_normalized = combo
                        
                        payout_per_100 = ticket_payouts.get(combo_normalized, 0)
                        if payout_per_100 > 0:
                            payout = int((amt / 100) * payout_per_100)
                            day_return += payout
                            hit_count += 1
                
                race_count += 1
                
            except Exception as e:
                logger.debug(f"Error processing race {rid}: {e}")
                continue
        
        total_invest += day_invest
        total_return += day_return
        
        if day_invest > 0:
            day_roi = day_return / day_invest * 100
            daily_results.append({
                'date': date_str,
                'invest': day_invest,
                'return': day_return,
                'profit': day_return - day_invest,
                'roi': day_roi
            })
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Backtest Results Summary")
    logger.info("=" * 60)
    
    profit = total_return - total_invest
    roi = (total_return / total_invest * 100) if total_invest > 0 else 0
    hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0
    
    logger.info(f"Total Investment: {total_invest:,} 円")
    logger.info(f"Total Return: {total_return:,} 円")
    logger.info(f"Profit: {profit:+,} 円")
    logger.info(f"ROI: {roi:.2f}%")
    logger.info(f"Races: {race_count}")
    logger.info(f"Bets: {bet_count}")
    logger.info(f"Hits: {hit_count}")
    logger.info(f"Hit Rate: {hit_rate:.2f}%")
    
    # Save results
    output_dir = 'reports/phase13'
    os.makedirs(output_dir, exist_ok=True)
    
    df_results = pd.DataFrame(daily_results)
    df_results.to_csv(f'{output_dir}/backtest_2025_daily.csv', index=False)
    
    # Summary report
    with open(f'{output_dir}/backtest_2025_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Phase 13 Production Strategy - 2025 Backtest\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 総合結果\n\n")
        f.write(f"- 総投資額: {total_invest:,} 円\n")
        f.write(f"- 総回収額: {total_return:,} 円\n")
        f.write(f"- 損益: {profit:+,} 円\n")
        f.write(f"- **ROI: {roi:.2f}%**\n")
        f.write(f"- レース数: {race_count}\n")
        f.write(f"- ベット数: {bet_count}\n")
        f.write(f"- 的中数: {hit_count}\n")
        f.write(f"- 的中率: {hit_rate:.2f}%\n\n")
        
        if daily_results:
            f.write("## 日別結果\n\n")
            f.write("| 日付 | 投資 | 回収 | 損益 | ROI |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            for r in daily_results[:30]:  # Show first 30 days
                f.write(f"| {r['date']} | {r['invest']:,} | {r['return']:,} | {r['profit']:+,} | {r['roi']:.1f}% |\n")
            if len(daily_results) > 30:
                f.write(f"\n... and {len(daily_results) - 30} more days\n")
    
    logger.info(f"\nResults saved to {output_dir}/")
    
    return {
        'total_invest': total_invest,
        'total_return': total_return,
        'profit': profit,
        'roi': roi,
        'race_count': race_count,
        'bet_count': bet_count,
        'hit_count': hit_count,
        'hit_rate': hit_rate
    }


if __name__ == '__main__':
    run_backtest_2025()
