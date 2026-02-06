
import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
import yaml
import datetime
from sqlalchemy import text, create_engine

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venue Map for Report (競馬場コード -> 競馬場名)
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
    For umaren/wide: '0209' -> '2-9' (sorted, hyphen-separated)
    For tansho/fukusho: '02' -> '2'
    """
    if not combo_str or pd.isna(combo_str):
        return None
    combo_str = str(combo_str).strip()
    
    if ticket_type in ['win', 'tansho', 'place', 'fukusho']:
        # Single horse: '02' -> '2'
        return str(int(combo_str))
    elif ticket_type in ['umaren', 'wide']:
        # Two horses: '0209' -> '2-9' (order matters for lookup, so sort)
        if len(combo_str) == 4:
            h1, h2 = int(combo_str[:2]), int(combo_str[2:])
            # Sort numerically and join with hyphen
            sorted_nums = sorted([h1, h2])
            return f"{sorted_nums[0]}-{sorted_nums[1]}"
        elif '-' in combo_str:
            # Already formatted
            parts = [int(x) for x in combo_str.split('-')]
            sorted_nums = sorted(parts)
            return f"{sorted_nums[0]}-{sorted_nums[1]}"
    elif ticket_type in ['umatan']:
        # Two horses but order matters: '0209' -> '2-9' (keep order)
        if len(combo_str) == 4:
            h1, h2 = int(combo_str[:2]), int(combo_str[2:])
            return f"{h1}-{h2}"
    elif ticket_type in ['sanrenpuku', 'sanrentan']:
        # Three horses: '020903' -> '2-9-3'
        if len(combo_str) == 6:
            h1, h2, h3 = int(combo_str[:2]), int(combo_str[2:4]), int(combo_str[4:])
            if ticket_type == 'sanrenpuku':
                sorted_nums = sorted([h1, h2, h3])
                return f"{sorted_nums[0]}-{sorted_nums[1]}-{sorted_nums[2]}"
            else:
                return f"{h1}-{h2}-{h3}"
    return combo_str
    

def load_payout_map_from_db(date_str, race_ids=None):
    """Load payout data from jvd_hr table and build payout map."""
    engine = get_db_engine()
    
    # Parse date to get year
    year = date_str.split('-')[0]
    
    # Build query
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
            FROM jvd_hr 
            WHERE kaisai_nen = '{year}'
            AND CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({race_ids_str})
        """)
    else:
        # Date-based filtering not available in jvd_hr directly, so use year
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
            FROM jvd_hr 
            WHERE kaisai_nen = '{year}'
        """)
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} payout records from DB")
    except Exception as e:
        logger.error(f"Failed to load payout data from DB: {e}")
        return {}
    
    # Build payout map: race_id -> ticket_type -> normalized_combination -> payout_per_100yen
    payout_map = {}
    
    def safe_int(val):
        """Safely convert value to int."""
        if val is None or pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return int(val) if not pd.isna(val) else None
        
        # Handle string
        val_str = str(val).strip()
        if not val_str or not val_str.isdigit():
            # Could be whitespace or special chars like '****'
            return None
        return int(val_str)

    for _, row in df.iterrows():
        rid = str(row['race_id'])
        if rid not in payout_map:
            payout_map[rid] = {'win': {}, 'place': {}, 'umaren': {}, 'wide': {}}
        
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
    
    return payout_map


def load_results(year):
    """Load results (rank) from preprocessed data."""
    path_results = f"data/processed/preprocessed_data.parquet"
    results_full = pd.read_parquet(path_results)
    results_year = results_full[results_full['year'] == year].copy()
    return results_year

def settle(date_str):
    logger.info(f"Settling trades for {date_str}...")
    
    date_clean = date_str.replace('-', '')
    ledger_path = f"outputs/ledgers/{date_clean}_ledger.parquet"
    order_path = f"outputs/orders/{date_clean}_orders.csv"
    
    if not os.path.exists(ledger_path):
        logger.error(f"Ledger not found: {ledger_path}")
        return

    df_bets = pd.read_parquet(ledger_path)
    if df_bets.empty:
        logger.info("No bets to settle.")
        return

    target_date = pd.Timestamp(date_str)
    year = target_date.year
    
    # Load results
    results_df = load_results(year)
    if results_df.empty:
        logger.error("Cannot load results.")
        return
    
    # Get unique race IDs from bets
    bet_race_ids = [str(rid) for rid in df_bets['race_id'].unique()]
    
    # Load payout map from DB (only for races with bets)
    payout_map = load_payout_map_from_db(date_str, race_ids=bet_race_ids)
    logger.info(f"Loaded payout data for {len(payout_map)} races")
        
    # Prepare Rank Map
    # race_id -> horse_number -> rank
    results_df['date'] = pd.to_datetime(results_df['date'])
    daily_results = results_df[results_df['date'] == date_str]
    
    rank_map = {}
    for rid, grp in daily_results.groupby('race_id'):
        rank_map[str(rid)] = dict(zip(grp['horse_number'], grp['rank']))
    
    total_invest = df_bets['amount'].sum()
    total_return = 0
    
    # Calculate Payout
    for i, row in df_bets.iterrows():
        rid = str(row['race_id'])
        ttype = row['ticket_type']
        combo = str(row['combination'])
        amt = row['amount']
        
        hit = False
        h2r = rank_map.get(rid, {})
        
        if not h2r:
            # Race result not found
            continue

        # Hit Logic (Duplicate of audit scripts)
        if ttype == 'win':
            if h2r.get(int(combo), 99) == 1: hit = True
        elif ttype == 'place':
             if h2r.get(int(combo), 99) <= 3: hit = True # Simplified (JRA usually 3, sometimes 2)
        elif ttype == 'umaren':
            p = [int(x) for x in combo.split('-')]
            if h2r.get(p[0], 99) <= 2 and h2r.get(p[1], 99) <= 2: hit = True
        elif ttype == 'wide':
            p = [int(x) for x in combo.split('-')]
            r1, r2 = h2r.get(p[0], 99), h2r.get(p[1], 99)
            # Wide hit if both in top 3
            if r1 <= 3 and r2 <= 3: hit = True
        
        # Payout - Use DB payout map (payout per 100 yen)
        pay = 0
        if hit:
            race_payouts = payout_map.get(rid, {})
            ticket_payouts = race_payouts.get(ttype, {})
            
            # Normalize combination for lookup
            combo_normalized = normalize_combination(combo, ttype) if '-' not in combo else combo
            if '-' in combo:
                # Already in hyphen format (e.g., '2-9'), ensure sorted for umaren/wide
                if ttype in ['umaren', 'wide']:
                    parts = [int(x) for x in combo.split('-')]
                    sorted_parts = sorted(parts)
                    combo_normalized = f"{sorted_parts[0]}-{sorted_parts[1]}"
                else:
                    combo_normalized = combo
            
            payout_per_100 = ticket_payouts.get(combo_normalized, 0)
            if payout_per_100 > 0:
                # Pay = (bet amount / 100) * payout_per_100
                pay = int((amt / 100) * payout_per_100)
            
        total_return += pay
        
    profit = total_return - total_invest
    roi = (total_return / total_invest * 100) if total_invest > 0 else 0

    report_file = f"reports/phase13/daily/{date_clean}_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # Prepare Detailed Report Data
    
    # Load Predictions if available
    pred_file = f"outputs/predictions/{date_clean}_predictions.parquet"
    predictions = pd.DataFrame()
    if os.path.exists(pred_file):
        predictions = pd.read_parquet(pred_file)
        
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 日次精算レポート: {date_str}\n\n")
        f.write(f"- 投資総額: {total_invest:,}円\n")
        f.write(f"- 回収総額: {total_return:,}円\n")
        f.write(f"- 損益: {profit:,}円\n")
        color = "red" if roi < 100 else "blue"
        f.write(f"- ROI: **{roi:.2f}%**\n\n")
        
        f.write("---\n\n")
        
        races = df_bets['race_id'].unique()
        
        # Sort races
        rid_to_rnum = {}
        if 'race_number' in results_df.columns:
            for rid, grp in results_df.groupby('race_id'):
                rid_to_rnum[str(rid)] = grp['race_number'].iloc[0]
        races = sorted(races, key=lambda x: rid_to_rnum.get(str(x), str(x)))
        
        for rid in races:
            rid = str(rid)
            r_bets = df_bets[df_bets['race_id'] == rid].copy()
            r_res = results_df[results_df['race_id'] == rid]
            
            # Race Header
            venue_code = ""
            rnum = ""
            title = ""
            if not r_res.empty:
                row0 = r_res.iloc[0]
                venue_code = str(row0.get('venue', '')).strip()
                venue_name = VENUE_MAP.get(venue_code, venue_code)
                rnum = row0.get('race_number', '?')
                title = row0.get('title', 'Unknown').strip()
            else:
                venue_name = venue_code
            
            f.write(f"## {venue_name} {rnum}R {title}\n")
            
            # 1. AI Top Picks vs Result
            f.write("### AI予測 vs 結果 (Top 5)\n")
            f.write("| AI順位 | 馬番 | 馬名 | Score | 着順 | 人気 | 単勝オッズ |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            if not predictions.empty:
                r_pred = predictions[predictions['race_id'] == rid].copy()
                if not r_pred.empty:
                    # Sort by pred_prob desc
                    r_pred = r_pred.sort_values('pred_prob', ascending=False).head(5)
                    
                    # Merge result info just in case
                    # (pred file might optionally contain rank, but results_df is definitive)
                    
                    # Create Rank/Odds map from r_res
                    h2res = {}
                    if not r_res.empty:
                        for _, row in r_res.iterrows():
                            h2res[int(row['horse_number'])] = {
                                'rank': row.get('rank', '-'),
                                'ninki': row.get('ninki', '-'),
                                'odds': row.get('odds', '-')
                            }
                    
                    rank_idx = 1
                    for _, p in r_pred.iterrows():
                        hnum = int(p['horse_number'])
                        hname = p.get('horse_name', 'Unknown')
                        prob = p.get('pred_prob', 0.0)
                        
                        res_info = h2res.get(hnum, {'rank': '-', 'ninki': '-', 'odds': '-'})
                        rank = res_info['rank']
                        ninki = res_info['ninki']
                        odds = res_info['odds']
                        
                        f.write(f"| {rank_idx} | {hnum} | {hname} | {prob:.4f} | **{rank}** | {ninki} | {odds} |\n")
                        rank_idx += 1
                else:
                    f.write("| No Prediction Data | - | - | - | - | - | - |\n")
            else:
                f.write("| No Prediction File | - | - | - | - | - | - |\n")
            
            f.write("\n")
            
            # 2. Result Summary (Top 3) (Optional if above is enough, but good for context)
            f.write("### レース結果 (Top 3)\n")
            f.write("| 着順 | 馬番 | 馬名 | 人気 | 単勝オッズ |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            if not r_res.empty:
                r_res_sorted = r_res.copy()
                r_res_sorted['rank_num'] = pd.to_numeric(r_res['rank'], errors='coerce')
                top3 = r_res_sorted[r_res_sorted['rank_num'] <= 3].sort_values('rank_num')
                
                for _, h in top3.iterrows():
                    rank = int(h['rank_num'])
                    hnum = int(h['horse_number'])
                    hname = h.get('horse_name', 'Unknown')
                    ninki = int(h.get('ninki', 99)) if not pd.isna(h.get('ninki')) else '-'
                    odds = h.get('odds', 0.0)
                    f.write(f"| {rank} | {hnum} | {hname} | {ninki} | {odds} |\n")
            
            f.write("\n")
            
            # 3. Bet Performance
            f.write("### 投票結果\n")
            f.write("| 券種 | 買い目 | 購入額 | 結果 | 払戻 | 収支 |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            
            r_invest = 0
            r_return = 0
            
            for _, b in r_bets.iterrows():
                ttype = b['ticket_type']
                combo = str(b['combination'])
                amt = b['amount']
                
                
                # Logic for hit calculation using Rank Map
                
                # Re-eval hit for display
                h2r = rank_map.get(rid, {})
                hit_disp = False
                if ttype == 'win':
                    if h2r.get(int(combo), 99) == 1: hit_disp = True
                elif ttype == 'place':
                     if h2r.get(int(combo), 99) <= 3: hit_disp = True
                elif ttype == 'umaren':
                    p = [int(x) for x in combo.split('-')]
                    if h2r.get(p[0], 99) <= 2 and h2r.get(p[1], 99) <= 2: hit_disp = True
                elif ttype == 'wakuren':
                    # Wakuren fix
                    if 'frame_number' in r_res.columns:
                        r_res_sorted = r_res.copy()
                        r_res_sorted['rank_num'] = pd.to_numeric(r_res['rank'], errors='coerce')
                        h1 = r_res_sorted[r_res_sorted['rank_num'] == 1]
                        h2 = r_res_sorted[r_res_sorted['rank_num'] == 2]
                        if not h1.empty and not h2.empty:
                            f1 = int(h1.iloc[0]['frame_number'])
                            f2 = int(h2.iloc[0]['frame_number'])
                            p = [int(x) for x in combo.split('-')]
                            if sorted([f1, f2]) == sorted(p): hit_disp = True

                pay_disp = 0
                if hit_disp:
                    race_payouts = payout_map.get(rid, {})
                    ticket_payouts = race_payouts.get(ttype, {})
                    
                    # Normalize combination for lookup
                    if '-' in combo:
                        if ttype in ['umaren', 'wide']:
                            parts = [int(x) for x in combo.split('-')]
                            sorted_parts = sorted(parts)
                            combo_normalized = f"{sorted_parts[0]}-{sorted_parts[1]}"
                        else:
                            combo_normalized = combo
                    else:
                        combo_normalized = combo
                    
                    payout_per_100 = ticket_payouts.get(combo_normalized, 0)
                    if payout_per_100 > 0:
                        pay_disp = int((amt / 100) * payout_per_100)
                
                res_str = "的中" if hit_disp else "ハズレ"
                profit_b = pay_disp - amt
                
                f.write(f"| {ttype} | {combo} | {amt:,} | {res_str} | {pay_disp:,} | {profit_b:,} |\n")
                
                r_invest += amt
                r_return += pay_disp
                
            # Race Total
            r_profit = r_return - r_invest
            f.write(f"\n**Race Total:** 投資 {r_invest:,} / 回収 {r_return:,} / 損益 {r_profit:,}\n\n")
            f.write("---\n")

    logger.info(f"Report saved to {report_file}")
    print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD')
    args = parser.parse_args()
    
    settle(args.date)
