import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_odds_string(odds_str, n_horses):
    """Parse JRA odds string (8 chars per horse)"""
    # 01003002 -> Uma=01, Odds=3.0, Ninki=02
    parsed = []
    # apd_sokuho_o1 is fixed length 224 (28 horses * 8 chars)
    # But we only care up to n_horses or non-empty blocks
    
    # Sometimes n_horses might be missing, so we iterate until we hit empty/invalid
    max_horses = 28
    
    for i in range(max_horses):
        start = i * 8
        end = start + 8
        if end > len(odds_str):
            break
            
        block = odds_str[start:end]
        if not block.strip():
            continue
            
        try:
            umaban_str = block[0:2]
            odds_str_val = block[2:6]
            ninki_str = block[6:8]
            
            # Check if valid number
            if not umaban_str.isdigit():
                continue
                
            umaban = int(umaban_str)
            if umaban == 0: # Invalid padding
                continue
                
            # Parse odds
            if odds_str_val.strip() == '' or not odds_str_val.isdigit():
                odds = np.nan
            else:
                odds = int(odds_str_val) / 10.0
                if odds == 0: # 0 means invalid/cancel sometimes?
                    odds = np.nan
                    
            parsed.append({
                'horse_number': umaban,
                'odds_snapshot': odds,
                'ninki_snapshot': int(ninki_str) if ninki_str.isdigit() else np.nan
            })
            
        except Exception as e:
            # logger.warning(f"Error parsing block '{block}': {e}")
            continue
            
    return parsed

def main():
    parser = argparse.ArgumentParser(description='Build Time-Series Odds Snapshot')
    parser.add_argument('--year', type=int, default=2025, help='Target year')
    parser.add_argument('--snapshot_label', type=str, default='T-10m', help='Snapshot label (e.g. T-10m, T-30m)')
    parser.add_argument('--offset_minutes', type=int, default=10, help='Minutes before start time')
    parser.add_argument('--out_dir', type=str, default='data/odds_snapshots', help='Output directory')
    args = parser.parse_args()
    
    # DB Connection
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    logger.info(f"Connecting to DB: {host}:{port}/{dbname}")
    engine = create_engine(db_url)
    
    # 1. Get Race Start Times (JRA Only for now)
    logger.info(f"Fetching race info for {args.year}...")
    race_query = f"""
    SELECT 
        kaisai_nen,
        kaisai_tsukihi,
        keibajo_code,
        kaisai_kai,
        kaisai_nichime,
        race_bango,
        hasso_jikoku -- HHmm string
    FROM jvd_ra
    WHERE kaisai_nen = '{args.year}' 
      AND data_kubun = '7' -- Race info
    """
    try:
        races_df = pd.read_sql(race_query, engine)
        logger.info(f"Found {len(races_df)} races.")
    except Exception as e:
        logger.error(f"Error fetching races: {e}")
        return

    # Create race_id and proper timestamp
    # race_id standard: YYYY + Venue + Kai + Nichi + RR
    
    races_df['date_str'] = str(args.year) + races_df['kaisai_tsukihi']
    races_df['race_id'] = (
        races_df['kaisai_nen'] + 
        races_df['keibajo_code'] + 
        races_df['kaisai_kai'] + 
        races_df['kaisai_nichime'] + 
        races_df['race_bango']
    )
    
    # Parse start time
    def parse_start_time(row):
        # YYYY MMDD HHmm
        dt_str = f"{row['kaisai_nen']}{row['kaisai_tsukihi']}{row['hasso_jikoku']}"
        try:
            return datetime.strptime(dt_str, '%Y%m%d%H%M')
        except:
            return pd.NaT

    races_df['start_time'] = races_df.apply(parse_start_time, axis=1)
    
    # Filter valid start times
    races_df = races_df.dropna(subset=['start_time'])
    
    # Calculate Target Snapshot Time
    offset = timedelta(minutes=args.offset_minutes)
    races_df['target_time'] = races_df['start_time'] - offset
    
    # 2. Fetch Time-Series Odds
    logger.info("Fetching apd_sokuho_o1 data...")
    # Fetch all records for the year (might be large, but filtered by year)
    # To optimize, we might fetch only necessary columns
    odds_query = f"""
    SELECT 
        kaisai_nen,
        kaisai_tsukihi,
        keibajo_code,
        kaisai_kai,
        kaisai_nichime,
        race_bango,
        happyo_tsukihi_jifun, -- MMDDHHmm (No year!)
        odds_tansho,
        toroku_tosu
    FROM apd_sokuho_o1
    WHERE kaisai_nen = '{args.year}'
    """
    sokuho_df = pd.read_sql(odds_query, engine)
    logger.info(f"Loaded {len(sokuho_df)} sokuho records.")
    
    # Parse timestamp in sokuho
    # happyo_tsukihi_jifun is MMDDHHmm. Need to add year.
    def parse_sokuho_time(row):
        # YYYY MMDDHHmm
        dt_str = f"{row['kaisai_nen']}{row['happyo_tsukihi_jifun']}"
        try:
            return datetime.strptime(dt_str, '%Y%m%d%H%M')
        except:
            return pd.NaT
            
    sokuho_df['timestamp'] = sokuho_df.apply(parse_sokuho_time, axis=1)
    sokuho_df = sokuho_df.dropna(subset=['timestamp'])
    
    # Create matching key
    sokuho_df['race_key'] = (
        sokuho_df['kaisai_nen'] + 
        sokuho_df['keibajo_code'] + 
        sokuho_df['kaisai_kai'] + 
        sokuho_df['kaisai_nichime'] + 
        sokuho_df['race_bango']
    )
    races_df['race_key'] = races_df['race_id'] # Same key structure
    
    # 3. Find latest record before target time
    logger.info("Matching snapshots...")
    
    snapshot_records = []
    
    # Group sokuho by race
    sokuho_groups = sokuho_df.groupby('race_key')
    
    hits = 0
    total = len(races_df)
    
    for _, race in tqdm(races_df.iterrows(), total=total):
        race_key = race['race_key']
        target = race['target_time']
        
        if race_key not in sokuho_groups.groups:
            continue
            
        # Get odds records for this race
        race_odds = sokuho_groups.get_group(race_key)
        
        # Filter records before target time
        valid_odds = race_odds[race_odds['timestamp'] <= target]
        
        if valid_odds.empty:
            continue
            
        # Get the latest one
        latest = valid_odds.sort_values('timestamp', ascending=False).iloc[0]
        
        # Parse odds
        horses = parse_odds_string(latest['odds_tansho'], latest['toroku_tosu'])
        
        for h in horses:
            snapshot_records.append({
                'race_id': race['race_id'],
                'horse_number': h['horse_number'], # matches horse_number in prediction
                'odds_snapshot': h['odds_snapshot'],
                'snapshot_timestamp': latest['timestamp'],
                'time_to_start': (latest['timestamp'] - race['start_time']).total_seconds() / 60.0 # minutes
            })
            
        hits += 1
        
    logger.info(f"Snapshot coverage: {hits}/{total} races ({hits/total:.1%})")
    
    # 4. Save
    if not snapshot_records:
        logger.warning("No snapshots generated.")
        return

    out_df = pd.DataFrame(snapshot_records)
    
    # Ensure standard types
    out_df['horse_number'] = out_df['horse_number'].astype(int)
    out_df['odds_snapshot'] = out_df['odds_snapshot'].astype(float)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{args.year}_win_{args.snapshot_label}_jra_only.parquet"
    out_path = out_dir / filename
    
    out_df.to_parquet(out_path)
    logger.info(f"Saved snapshot data to {out_path}")
    
    # Save coverage report
    report_path = out_dir / f"coverage_{args.year}_{args.snapshot_label}.md"
    with open(report_path, 'w') as f:
        f.write(f"# Odds Snapshot Coverage Report ({args.snapshot_label})\n\n")
        f.write(f"- Year: {args.year}\n")
        f.write(f"- Label: {args.snapshot_label} (T-{args.offset_minutes}m)\n")
        f.write(f"- Total Races: {total}\n")
        f.write(f"- Covered Races: {hits}\n")
        f.write(f"- Coverage: {hits/total:.1%}\n\n")
        f.write("## Sample Data\n\n")
        f.write(out_df.head(10).to_markdown(index=False))

if __name__ == '__main__':
    main()
