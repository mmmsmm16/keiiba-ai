import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
import datetime
from sqlalchemy import text, create_engine

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venue Map for NAR (South Kanto Focus)
VENUE_MAP = {
    '42': '浦和', '43': '船橋', '44': '大井', '45': '川崎',
    '30': '門別', '31': '北海', '32': '水沢', '33': '盛岡', 
    '34': '帯広', '35': '金沢', '36': '笠松', '41': '高知',
    '46': '園田', '47': '姫路', '48': '名古', '50': '佐賀'
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
    if not combo_str or pd.isna(combo_str):
        return None
    combo_str = str(combo_str).strip()
    
    if ticket_type in ['WIN', 'TANSHO', 'PLACE', 'FUKUSHO']:
        return str(int(combo_str))
    elif ticket_type in ['UMAREN', 'WIDE', 'UMATAN']: # UMATAN needs order, but normalize usually sorts for key? Wait, UMATAN is ordered.
        # But payout_map keys might be ordered or not? 
        # Usually standard is: 
        # - Umaren/Wide: Smaller-Larger
        # - Umatan: 1st-2nd
        # Let's assume input and map keys need to align.
        # Check for hyphen
        if '-' in combo_str:
            parts = combo_str.split('-')
            if len(parts) == 2:
                try:
                    h1, h2 = int(parts[0]), int(parts[1])
                    # If ticket type is ordered (Umatan), keep order?
                    # Current load_payout_map uses normalize_combination which sorts for UMAREN/WIDE.
                    if ticket_type in ['UMAREN', 'WIDE']:
                        sorted_nums = sorted([h1, h2])
                        return f"{sorted_nums[0]}-{sorted_nums[1]}"
                    else:
                        return f"{h1}-{h2}"
                except ValueError:
                    return combo_str
        # If no hyphen but length indicates composite (e.g. 0102)
        elif len(combo_str) == 4 and combo_str.isdigit():
             h1, h2 = int(combo_str[:2]), int(combo_str[2:])
             if ticket_type in ['UMAREN', 'WIDE']:
                 sorted_nums = sorted([h1, h2])
                 return f"{sorted_nums[0]}-{sorted_nums[1]}"
             else:
                 return f"{h1}-{h2}"
                 
    return combo_str
    

def load_payout_map_from_db(date_str, race_ids=None):
    """Load payout data from nvd_hr table."""
    engine = get_db_engine()
    year = date_str.split('-')[0]
    
    # NAR Table: nvd_hr or nvd_haraimodoshi
    # Based on check, nvd_hr exists.
    table_name = "nvd_hr"
    
    if race_ids:
        race_ids_str = ",".join([f"'{rid}'" for rid in race_ids])
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
                haraimodoshi_wide_7a, haraimodoshi_wide_7b
            FROM {table_name} 
            WHERE kaisai_nen = '{year}'
            AND CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({race_ids_str})
        """)
    else:
        query = text(f"SELECT * FROM {table_name} WHERE kaisai_nen = '{year}' LIMIT 0") # Fallback safety
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} payout records from DB")
    except Exception as e:
        logger.error(f"Failed to load payout data from DB: {e}")
        return {}
    
    payout_map = {}
    
    def safe_int(val):
        if val is None or pd.isna(val): return None
        try: return int(float(val))
        except: return None

    for _, row in df.iterrows():
        rid = str(row['race_id'])
        if rid not in payout_map:
            payout_map[rid] = {'WIN': {}, 'PLACE': {}, 'UMAREN': {}, 'WIDE': {}}
        
        # Tansho (WIN)
        combo = normalize_combination(row.get('haraimodoshi_tansho_1a'), 'WIN')
        pay = safe_int(row.get('haraimodoshi_tansho_1b'))
        if combo and pay:
            payout_map[rid]['WIN'][combo] = pay # Pipeline uses 'WIN'
        
        # Fukusho (PLACE) - NAR might have fewer places? Usually 3, up to 5 for wide/misc.
        for i in range(1, 6): # Check up to 5 slots just in case
            k_a = f'haraimodoshi_fukusho_{i}a'
            k_b = f'haraimodoshi_fukusho_{i}b'
            if k_a in row and row[k_a]:
                combo = normalize_combination(row.get(k_a), 'PLACE')
                pay = safe_int(row.get(k_b))
                if combo and pay:
                    payout_map[rid]['PLACE'][combo] = pay
        
        # Umaren
        for i in range(1, 4):
            k_a = f'haraimodoshi_umaren_{i}a'
            k_b = f'haraimodoshi_umaren_{i}b'
            if k_a in row and row[k_a]:
                 combo = normalize_combination(row.get(k_a), 'UMAREN')
                 pay = safe_int(row.get(k_b))
                 if combo and pay:
                     payout_map[rid]['UMAREN'][combo] = pay

        # Wide
        for i in range(1, 8):
            k_a = f'haraimodoshi_wide_{i}a'
            k_b = f'haraimodoshi_wide_{i}b'
            if k_a in row and row[k_a]:
                 combo = normalize_combination(row.get(k_a), 'WIDE')
                 pay = safe_int(row.get(k_b))
                 if combo and pay:
                     payout_map[rid]['WIDE'][combo] = pay

        # Add logic for Sanrenpuku/Sanrentan if columns exist individually or handle generic logic?
        # Assuming schema has them. If not visible in previous snippet, skip or infer.
        # Based on schema snippet: wide only went up to 7b. Sanrenpuku not listed but likely there?
        # For now, Umaren and Wide are confirmed visible in the query.
        
    return payout_map

def load_results(date_str):
    path_results = "data/nar/preprocessed_data_south_kanto.parquet"
    if not os.path.exists(path_results):
        logger.error(f"Results file not found: {path_results}")
        return pd.DataFrame()
        
    df = pd.read_parquet(path_results)
    df['date'] = pd.to_datetime(df['date'])
    return df[df['date'] == date_str].copy()

def settle(date_str):
    logger.info(f"Settling trades for {date_str}...")
    date_clean = date_str.replace('-', '')
    ledger_path = f"reports/nar/ledgers/{date_clean}_ledger.parquet"
    
    if not os.path.exists(ledger_path):
        logger.error(f"Ledger not found: {ledger_path}")
        return

    df_bets = pd.read_parquet(ledger_path)
    if df_bets.empty:
        logger.info("No bets to settle.")
        return

    # Load results
    results_df = load_results(date_str)
    if results_df.empty:
        logger.error(f"No results found for {date_str}")
        return
    
    bet_race_ids = [str(rid) for rid in df_bets['race_id'].unique()]
    payout_map = load_payout_map_from_db(date_str, race_ids=bet_race_ids)

    # Load predictions if available
    pred_path = f"reports/nar/predictions/{date_clean}_predictions.parquet"
    if os.path.exists(pred_path):
        predictions = pd.read_parquet(pred_path)
    else:
        predictions = pd.DataFrame()

    # Prepare metadata map from results (Title, Venue, RNum)
    race_meta = {}
    for _, row in results_df.iterrows():
        rid = str(row['race_id'])
        if rid not in race_meta:
            race_meta[rid] = {
                'title': row.get('title'),
                'race_number': row.get('race_number'),
                'venue': row.get('venue')
            }
    
    # Calculate Totals first
    total_invest = df_bets['amount'].sum()
    total_return = 0
    
    # Pre-calculate returns to get total profit for header
    # Also prepare detailed records to avoid double iteration logic complexity or just iterate twice
    # Iterating twice is cheap here.
    
    def calculate_return(row):
        rid = str(row['race_id'])
        ttype = row['ticket_type'].upper()
        combo = str(row['combination'])
        amt = row['amount']
        
        map_combo = normalize_combination(row['combination'], ttype)
        race_pays = payout_map.get(rid, {}).get(ttype, {})
        pay_per_100 = race_pays.get(map_combo, 0)
        
        if pay_per_100 > 0:
            return int((amt / 100) * pay_per_100)
        return 0

    total_return = df_bets.apply(calculate_return, axis=1).sum()
    profit = total_return - total_invest
    roi = (total_return / total_invest * 100) if total_invest > 0 else 0

    output_report = f"reports/nar/daily/{date_clean}_report.md"
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write(f"# NAR Settlement Report: {date_str}\n\n")
        f.write(f"- Invest: {total_invest:,} JPY\n")
        f.write(f"- Return: {total_return:,} JPY\n")
        f.write(f"- Profit: {profit:,} JPY\n")
        color = "red" if roi < 100 else "blue"
        f.write(f"- ROI: **{roi:.2f}%**\n\n")
        f.write("---\n\n")

        # Sort races
        # Try to sort by race_number if available, else race_id
        races = sorted(list(set(df_bets['race_id']) | set(results_df['race_id'])))
        # Filter to races that have bets OR are in predictions? 
        # Usually report only on races we BET on, unless we want full coverage.
        # JRA report usually covers races with bets.
        races = sorted(df_bets['race_id'].unique().tolist())
        
        # Helper to get RNum for sorting
        def get_rnum(rid):
            m = race_meta.get(str(rid))
            if m and m['race_number']: return float(m['race_number'])
            return 99.0
        races.sort(key=get_rnum)

        for rid in races:
            rid = str(rid)
            r_bets = df_bets[df_bets['race_id'] == rid]
            
            # Metadata
            meta = race_meta.get(rid, {})
            title = meta.get('title', 'Unknown')
            rnum = meta.get('race_number', '?')
            venue_code = str(meta.get('venue', ''))
            venue_name = VENUE_MAP.get(venue_code, venue_code)
            
            f.write(f"## {venue_name} {rnum}R {title}\n")
            
            # 1. Prediction Table
            f.write("### AI Prediction (Top 5)\n")
            f.write("| Rank | Horse | Name | Score | Result | Odds |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            if not predictions.empty:
                r_pred = predictions[predictions['race_id'] == rid]
                if not r_pred.empty:
                    r_pred = r_pred.sort_values('pred_prob', ascending=False).head(5)
                    
                    # Get race result for looking up rank/odds
                    r_res = results_df[results_df['race_id'] == rid]
                    h2res = {}
                    for _, row in r_res.iterrows():
                        try:
                            h2res[int(row['horse_number'])] = {'rank': row.get('rank', '-'), 'odds': row.get('odds', '-')}
                        except: pass
                        
                    for i, (idx, p_row) in enumerate(r_pred.iterrows(), 1):
                        hnum = int(p_row['horse_number'])
                        hname = p_row.get('horse_name', '')
                        score = p_row.get('pred_prob', 0)
                        
                        res_info = h2res.get(hnum, {'rank': '-', 'odds': '-'})
                        f.write(f"| {i} | {hnum} | {hname} | {score:.4f} | **{res_info['rank']}** | {res_info['odds']} |\n")
                else:
                    f.write("| - | - | No Prediction Data | - | - | - |\n")
            else:
                f.write("| - | - | Prediction File Missing | - | - | - |\n")
            f.write("\n")

            # 2. Bet Table
            f.write("### Betting Results\n")
            f.write("| Type | Combo | Amount | Result | Return | Profit |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            race_inv = 0
            race_ret = 0
            
            for _, row in r_bets.iterrows():
                ttype = row['ticket_type'].upper()
                combo = str(row['combination'])
                amt = row['amount']
                
                # Logic duplicating hit check (simplified)
                # Map based lookup
                map_combo = normalize_combination(row['combination'], ttype)
                race_pays = payout_map.get(rid, {}).get(ttype, {})
                pay_per_100 = race_pays.get(map_combo, 0)
                
                if pay_per_100 > 0:
                    pay = int((amt / 100) * pay_per_100)
                    res_str = "HIT"
                else:
                    pay = 0
                    res_str = "MISS"
                    
                prof = pay - amt
                f.write(f"| {ttype} | {combo} | {amt:,} | {res_str} | {pay:,} | {prof:,} |\n")
                
                race_inv += amt
                race_ret += pay
                
            f.write(f"\n**Total:** Invest {race_inv:,} / Return {race_ret:,} / Profit {race_ret - race_inv:,}\n\n")
            f.write("---\n")

    logger.info(f"Report saved to {output_report}")
    print(f"Total ROI: {roi:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True)
    settle(parser.parse_args().date)
