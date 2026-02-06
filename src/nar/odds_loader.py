"""
NAR Odds Loader from nvd_o1 and nvd_o2 tables.

Data format (based on JV-Link specification analysis):
- nvd_o1.odds_tansho (Win): HorseNumber(2) + Odds(4, x10) + Flag(2) = 8 chars per horse
- nvd_o1.odds_fukusho (Place): HorseNumber(2) + OddsMin(4, x10) + OddsMax(4, x10) + Flag(2) = 12 chars per horse
- nvd_o2.odds_umaren (Quinella): Horse1(2) + Horse2(2) + Odds(4, x10) + Flag(2) = 10 chars per combination
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(db_url)

def parse_tansho_odds(odds_str: str) -> dict:
    """
    Parse Win (Tansho) odds string.
    Format: 8 chars per horse: HorseNumber(2) + Odds(4) + Flag(2)
    Odds value is divided by 10 (e.g., 0182 -> 18.2)
    
    Returns: {horse_number_str: odds_float}
    """
    result = {}
    if not odds_str:
        return result
    
    chunk_size = 8
    for i in range(0, len(odds_str), chunk_size):
        chunk = odds_str[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        try:
            horse_no = chunk[0:2]
            odds_raw = int(chunk[2:6])
            # flag = chunk[6:8]  # Not used currently
            
            if odds_raw > 0:
                odds = odds_raw / 10.0
                result[horse_no.lstrip('0') or '0'] = odds
        except (ValueError, IndexError):
            continue
    
    return result

def parse_fukusho_odds(odds_str: str) -> dict:
    """
    Parse Place (Fukusho) odds string.
    Format: 12 chars per horse: HorseNumber(2) + OddsMin(4) + OddsMax(4) + Flag(2)
    Odds values are divided by 10.
    
    Returns: {horse_number_str: (odds_min, odds_max)}
    For simplicity, we return the average: {horse_number_str: odds_avg}
    """
    result = {}
    if not odds_str:
        return result
    
    chunk_size = 12
    for i in range(0, len(odds_str), chunk_size):
        chunk = odds_str[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        try:
            horse_no = chunk[0:2]
            odds_min_raw = int(chunk[2:6])
            odds_max_raw = int(chunk[6:10])
            # flag = chunk[10:12]
            
            if odds_min_raw > 0 and odds_max_raw > 0:
                odds_min = odds_min_raw / 10.0
                odds_max = odds_max_raw / 10.0
                # Use average for EV calculation
                odds_avg = (odds_min + odds_max) / 2.0
                result[horse_no.lstrip('0') or '0'] = odds_avg
        except (ValueError, IndexError):
            continue
    
    return result

def parse_umaren_odds(odds_str: str) -> dict:
    """
    Parse Quinella (Umaren) odds string.
    Format: 10 chars per combination: Horse1(2) + Horse2(2) + Odds(4) + Flag(2)
    Odds value is divided by 10.
    
    Returns: {"H1-H2": odds_float} where H1 < H2 (sorted)
    """
    result = {}
    if not odds_str:
        return result
    
    chunk_size = 10
    for i in range(0, len(odds_str), chunk_size):
        chunk = odds_str[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        try:
            horse1_raw = chunk[0:2]
            horse2_raw = chunk[2:4]
            odds_raw = int(chunk[4:8])
            # flag = chunk[8:10]
            
            if odds_raw > 0:
                h1 = horse1_raw.lstrip('0') or '0'
                h2 = horse2_raw.lstrip('0') or '0'
                # Ensure sorted key
                combo = f"{min(int(h1), int(h2))}-{max(int(h1), int(h2))}"
                odds = odds_raw / 10.0
                result[combo] = odds
        except (ValueError, IndexError):
            continue
    
    return result

def load_odds_from_db(date_str: str, race_ids: list = None) -> dict:
    """
    Load full odds data for a given date from nvd_o1 and nvd_o2.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD' format
        race_ids: Optional list of specific race IDs to load
        
    Returns:
        {race_id: {'WIN': {combo: odds}, 'PLACE': {combo: odds}, 'UMAREN': {combo: odds}}}
    """
    engine = get_db_engine()
    
    # Parse date
    year = date_str[:4]
    month = date_str[5:7]
    day = date_str[8:10]
    mmdd = month + day
    
    result = defaultdict(lambda: defaultdict(dict))
    
    with engine.connect() as conn:
        # Query O1 (Win, Place, Wakuren)
        q1 = text(f"""
            SELECT 
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
                odds_tansho, odds_fukusho, odds_wakuren
            FROM nvd_o1
            WHERE kaisai_nen = :year AND kaisai_tsukihi = :mmdd
        """)
        df1 = pd.read_sql(q1, conn, params={'year': year, 'mmdd': mmdd})
        
        for _, row in df1.iterrows():
            rid = row['race_id']
            if race_ids and rid not in race_ids:
                continue
            
            # Parse Win
            win_odds = parse_tansho_odds(row['odds_tansho'])
            for combo, odds in win_odds.items():
                result[rid]['WIN'][combo] = odds
            
            # Parse Place
            place_odds = parse_fukusho_odds(row['odds_fukusho'])
            for combo, odds in place_odds.items():
                result[rid]['PLACE'][combo] = odds
        
        # Query O2 (Umaren)
        q2 = text(f"""
            SELECT 
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
                odds_umaren
            FROM nvd_o2
            WHERE kaisai_nen = :year AND kaisai_tsukihi = :mmdd
        """)
        df2 = pd.read_sql(q2, conn, params={'year': year, 'mmdd': mmdd})
        
        for _, row in df2.iterrows():
            rid = row['race_id']
            if race_ids and rid not in race_ids:
                continue
            
            umaren_odds = parse_umaren_odds(row['odds_umaren'])
            for combo, odds in umaren_odds.items():
                result[rid]['UMAREN'][combo] = odds
    
    logger.info(f"Loaded odds for {len(result)} races from nvd_o1/o2")
    return dict(result)


if __name__ == "__main__":
    # Test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample date
    test_date = "2025-01-01"
    odds_map = load_odds_from_db(test_date)
    
    print(f"Loaded {len(odds_map)} races for {test_date}")
    for rid, types in list(odds_map.items())[:2]:
        print(f"\nRace: {rid}")
        for ttype, combos in types.items():
            print(f"  {ttype}: {len(combos)} entries, sample: {list(combos.items())[:3]}")
