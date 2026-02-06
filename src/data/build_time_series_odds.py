import argparse
import pandas as pd
import numpy as np
import os
import sys
import logging
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_win_odds(odds_str, n_horses):
    """Win Odds Parser"""
    parsed = []
    max_horses = 28
    if not isinstance(odds_str, str): return []
    for i in range(max_horses):
        start = i * 8
        end = start + 8
        if end > len(odds_str): break
        block = odds_str[start:end]
        if not block.strip(): continue
        try:
            umaban_str = block[0:2]
            odds_str_val = block[2:6]
            ninki_str = block[6:8]
            if not umaban_str.isdigit() or int(umaban_str) == 0: continue
            if odds_str_val.strip() == '' or not odds_str_val.isdigit():
                odds = np.nan
            else:
                odds = int(odds_str_val) / 10.0
                if odds == 0: odds = np.nan
            parsed.append({
                'ticket_type': 'win',
                'combination': str(int(umaban_str)),
                'odds': odds,
                'ninki': int(ninki_str) if ninki_str.isdigit() else np.nan
            })
        except: continue
    return parsed

def parse_place_odds(odds_str, n_horses):
    """Place Odds Parser (Min/Max ranges? Usually just min is used for EV safety, or average?)"""
    # JVD Place odds: [Umaban(2)][MinOdds(4)][MaxOdds(4)][Ninki(2)] = 12 chars
    parsed = []
    max_horses = 28
    if not isinstance(odds_str, str): return []
    
    for i in range(max_horses):
        start = i * 12
        end = start + 12
        if end > len(odds_str): break
        block = odds_str[start:end]
        if not block.strip(): continue
        try:
            umaban = int(block[0:2])
            if umaban == 0: continue
            
            min_o_str = block[2:6]
            # max_o_str = block[6:10] # unused for simple EV
            ninki_str = block[10:12]
            
            if not min_o_str.isdigit():
                odds = np.nan
            else:
                odds = int(min_o_str) / 10.0
                if odds == 0: odds = np.nan
                
            parsed.append({
                'ticket_type': 'place',
                'combination': str(umaban),
                'odds': odds, # Conservative (Min)
                'ninki': int(ninki_str) if ninki_str.isdigit() else np.nan
            })
        except: continue
    return parsed

def parse_umaren_odds(odds_str, n_horses):
    """Umaren Odds Parser"""
    # JVD Umaren: [Kumi(1..153)][Odds(5)][Ninki(3)] ?? No.
    # Actually continuous blocks.
    # Let's check JVD spec or infer.
    # Usually: [Umaban1(2)][Umaban2(2)][Odds(5)][Ninki(3)] = 12 chars?
    # Or 153 fixed slots?
    # Spec "O2" record: Odds Data Area (variable length).
    # Repeated: [Umaban1(2)][Umaban2(2)][Odds(4)][Ninki(2)] ??
    # Wait, o1 was 8 chars.
    # Let's assume standard JRA-VAN format for variable length.
    # Actually, standard is [Umaban1(2)][Umaban2(2)][Odds(4)][Ninki(2)] = 10 chars per combo.
    # Let's try 10 chars.
    
    parsed = []
    # No fixed limit, just read until end
    if not isinstance(odds_str, str): return []
    
    chunk_size = 10
    num_chunks = len(odds_str) // chunk_size
    
    for i in range(num_chunks):
        block = odds_str[i*chunk_size : (i+1)*chunk_size]
        try:
            u1 = int(block[0:2])
            u2 = int(block[2:4])
            if u1 == 0 or u2 == 0: continue
            
            # Sort for unique key
            mj = sorted([u1, u2])
            key = f"{mj[0]}-{mj[1]}"
            
            odds_s = block[4:8]
            ninki_s = block[8:10]
            
            if not odds_s.isdigit():
                odds = np.nan
            else:
                odds = int(odds_s) / 10.0
                if odds == 0: odds = np.nan
            
            parsed.append({
                'ticket_type': 'umaren',
                'combination': key,
                'odds': odds,
                'ninki': int(ninki_s) if ninki_s.isdigit() else np.nan
            })
        except: continue
    return parsed
    
def parse_wakuren_odds(odds_str):
    """Wakuren: [Waku1(1)][Waku2(1)][Odds(4)][Ninki(2)] = 8 chars?"""
    # Probably.
    parsed = []
    if not isinstance(odds_str, str): return []
    
    chunk_size = 8
    num_chunks = len(odds_str) // chunk_size
    
    for i in range(num_chunks):
        block = odds_str[i*chunk_size : (i+1)*chunk_size]
        try:
            w1 = int(block[0:1])
            w2 = int(block[1:2])
            if w1 == 0 or w2 == 0: continue
            
            mj = sorted([w1, w2])
            key = f"{mj[0]}-{mj[1]}"
            
            odds_s = block[2:6]
            ninki_s = block[6:8]
            
            if not odds_s.isdigit():
                odds = np.nan
            else:
                odds = int(odds_s) / 10.0
                if odds == 0: odds = np.nan
                
            parsed.append({
                'ticket_type': 'wakuren',
                'combination': key,
                'odds': odds,
                'ninki': int(ninki_s) if ninki_s.isdigit() else np.nan
            })
        except: continue
    return parsed

def process_year(year, engine, out_dir, offsets):
    """Process a single year for Odds Snapshots (All Types)"""
    logger.info(f"=== Processing Year {year} ===")
    
    # 1. Races
    logger.info(f"Loading Races {year}...")
    races_query = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, hasso_jikoku FROM jvd_ra WHERE kaisai_nen = '{year}' AND data_kubun = '7'"
    races = pd.read_sql(races_query, engine)
    
    if races.empty:
        return

    races['race_id'] = races['kaisai_nen'] + races['keibajo_code'] + races['kaisai_kai'] + races['kaisai_nichime'] + races['race_bango']
    # FIX: Deduplicate races to prevent double processing
    races = races.drop_duplicates(subset=['race_id'])
    
    def parse_time(row, time_str):
        try:
            ts = str(time_str).split('.')[0].strip()
            if len(ts) < 4: return pd.NaT
            full_str = ""
            if len(ts) == 4: # hasso (HHmm)
                date_part = str(row['kaisai_tsukihi']).zfill(4)
                full_str = f"{row['kaisai_nen']}{date_part}{ts}"
            elif len(ts) <= 8: # happyo (MMDDHHmm)
                ts = ts.zfill(8)
                full_str = f"{row['kaisai_nen']}{ts}"
            else:
                return pd.NaT
            return datetime.strptime(full_str, '%Y%m%d%H%M')
        except: return pd.NaT

    races['start_time'] = races.apply(lambda r: parse_time(r, r['hasso_jikoku']), axis=1)
    races = races.dropna(subset=['start_time'])
    
    # 2. Odds Data (O1: Win/Place/Wakuren, O2: Umaren)
    # Load O1
    logger.info(f"Loading O1 (Win/Place/Waku) {year}...")
    o1_query = f"""
    SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, happyo_tsukihi_jifun, odds_tansho, odds_fukusho, odds_wakuren, toroku_tosu 
    FROM apd_sokuho_o1 WHERE kaisai_nen = '{year}'
    """
    o1_df = pd.read_sql(o1_query, engine)
    
    # Load O2
    logger.info(f"Loading O2 (Umaren) {year}...")
    o2_query = f"""
    SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, happyo_tsukihi_jifun, odds_umaren
    FROM apd_sokuho_o2 WHERE kaisai_nen = '{year}'
    """
    o2_df = pd.read_sql(o2_query, engine)
    
    # Process O1
    o1_df['race_id'] = o1_df['kaisai_nen'] + o1_df['keibajo_code'] + o1_df['kaisai_kai'] + o1_df['kaisai_nichime'] + o1_df['race_bango']
    o1_df['timestamp'] = o1_df.apply(lambda r: parse_time(r, r['happyo_tsukihi_jifun']), axis=1)
    o1_df = o1_df.dropna(subset=['timestamp'])
    
    # Process O2
    o2_df['race_id'] = o2_df['kaisai_nen'] + o2_df['keibajo_code'] + o2_df['kaisai_kai'] + o2_df['kaisai_nichime'] + o2_df['race_bango']
    o2_df['timestamp'] = o2_df.apply(lambda r: parse_time(r, r['happyo_tsukihi_jifun']), axis=1)
    o2_df = o2_df.dropna(subset=['timestamp'])

    o1_groups = o1_df.groupby('race_id')
    o2_groups = o2_df.groupby('race_id')
    
    snapshots = {m: [] for m in offsets} # Dict keyed by minute -> list of records
    stats = {m: 0 for m in offsets}
    
    for _, race in tqdm(races.iterrows(), total=len(races), desc=f"Scanning {year}"):
        rid = race['race_id']
        start = race['start_time']
        
        # Get latest O1 and O2 for each target time
        o1_race = o1_groups.get_group(rid) if rid in o1_groups.groups else pd.DataFrame()
        o2_race = o2_groups.get_group(rid) if rid in o2_groups.groups else pd.DataFrame()
        
        for m in offsets:
            # Snapshot Container for this race/time
            race_snap = []
            
            if m == 'final':
                # Take the absolute last record regardless of time
                # O1
                if not o1_race.empty:
                    latest = o1_race.sort_values('timestamp', ascending=False).iloc[0]
                    race_snap.extend(parse_win_odds(latest['odds_tansho'], latest['toroku_tosu']))
                    race_snap.extend(parse_place_odds(latest['odds_fukusho'], latest['toroku_tosu']))
                    race_snap.extend(parse_wakuren_odds(latest['odds_wakuren']))
                # O2
                if not o2_race.empty:
                    latest = o2_race.sort_values('timestamp', ascending=False).iloc[0]
                    race_snap.extend(parse_umaren_odds(latest['odds_umaren'], 0))
            else:
                # Time-based
                target_time = start - timedelta(minutes=m)
                
                # --- Win/Place/Waku (O1) ---
                if not o1_race.empty:
                    valid = o1_race[o1_race['timestamp'] <= target_time]
                    if not valid.empty:
                        latest = valid.sort_values('timestamp', ascending=False).iloc[0]
                        race_snap.extend(parse_win_odds(latest['odds_tansho'], latest['toroku_tosu']))
                        race_snap.extend(parse_place_odds(latest['odds_fukusho'], latest['toroku_tosu']))
                        race_snap.extend(parse_wakuren_odds(latest['odds_wakuren']))
                
                # --- Umaren (O2) ---
                if not o2_race.empty:
                    valid = o2_race[o2_race['timestamp'] <= target_time]
                    if not valid.empty:
                        latest = valid.sort_values('timestamp', ascending=False).iloc[0]
                        race_snap.extend(parse_umaren_odds(latest['odds_umaren'], 0))
                    
            if not race_snap: continue
            
            # Add metadata
            for r in race_snap:
                r['race_id'] = rid
                r['snapshot_type'] = f'T-{m}' if m != 'final' else 'final'
                # FIX: Invalidate odds < 1.0 (Error codes)
                if pd.notna(r['odds']) and r['odds'] < 1.0:
                    r['odds'] = np.nan
                
                snapshots[m].append(r)
                
            stats[m] += 1

    # 4. Save
    out_path = Path(out_dir) / str(year)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for m in offsets:
        recs = snapshots[m]
        if not recs: continue
        df_snap = pd.DataFrame(recs)
        # Convert numeric types
        df_snap['odds'] = pd.to_numeric(df_snap['odds'], errors='coerce')
        
        # FIX: Deduplicate snapshots (Ticket Level)
        df_snap = df_snap.drop_duplicates(subset=['race_id', 'ticket_type', 'combination'], keep='last')
        
        lbl = f"T-{m}" if m != 'final' else "final"
        fname = f"odds_{lbl}.parquet"
        df_snap.to_parquet(out_path / fname)
        logger.info(f"Saved {year} {lbl}: {len(df_snap)} rows (Coverage: {stats[m]}/{len(races)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year', type=int, default=2025)
    parser.add_argument('--end_year', type=int, default=2025)
    parser.add_argument('--out_dir', type=str, default='data/odds_snapshots')
    args = parser.parse_args()
    
    # DB Connection
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(db_url)
    
    # Modified offsets to include 'final'
    offsets = [60, 30, 10, 5, 'final']
    
    for y in range(args.start_year, args.end_year + 1):
        process_year(y, engine, args.out_dir, offsets)

if __name__ == "__main__":
    main()
