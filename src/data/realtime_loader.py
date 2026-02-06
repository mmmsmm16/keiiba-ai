import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class RealTimeDataLoader:
    """
    Real-time odds loader from 'apd_sokuho_*' tables.
    """
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'user')
        password = os.environ.get('POSTGRES_PASSWORD', 'password')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(connection_str)
        
        # Table mapping
        self.table_map = {
            'win': 'apd_sokuho_o1', # Win (Tansho)
            'place': 'apd_sokuho_o1', # Place (Fukusho)
            'umaren': 'apd_sokuho_o2', # Umaren
            'wide': 'apd_sokuho_o3',
            'umatan': 'apd_sokuho_o4',
            'trio': 'apd_sokuho_o5', # Sanrenpuku
            'trifecta': 'apd_sokuho_o6' # Sanrentan
        }

    def get_latest_odds(self, race_id_list: List[str], bet_type: str = 'win') -> Dict[str, Dict[str, float]]:
        """
        Fetch latest odds for a list of race_ids.
        
        Args:
            race_id_list: List of race_ids (YYYYMMDDJJRR format expected, or similar)
            bet_type: 'win', 'place', 'umaren', etc.
            
        Returns:
            Dict: {race_id: {selection_key: odds_value}}
                  selection_key is '01' for win, '01-02' for umaren.
        """
        if len(race_id_list) == 0:
            return {}
            
        table_name = self.table_map.get(bet_type)
        if not table_name:
            logger.error(f"Unsupported bet type: {bet_type}")
            return {}

        results = {}
        
        # Group race_ids by Kaijo/Year to optimize if needed, but for now simple query loop or IN clause
        # The key in apd_sokuho is kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango.
        # Check standard race_id format: YYYY(4) + Venue(2) + Kai(2) + Nichi(2) + RR(2) usually = 12 chars?
        # But JRA-VAN standard race_id is often constructed as above.
        # User's race_id is 2025121409050405 ? (16 chars?)
        # Let's assume input is standard 16-digit or compatible.
        
        for race_id in race_id_list:
            odds_dict = self._fetch_single_race_odds(race_id, table_name, bet_type)
            if odds_dict:
                results[race_id] = odds_dict
                
        return results

    def _fetch_single_race_odds(self, race_id: str, table_name: str, bet_type: str) -> Optional[Dict[str, float]]:
        # Parse race_id
        # Expecting: YYYY(0:4) Venue(6:8) - Need to map carefully.
        # Standard Loader uses: YYYY + Venue + Kai + Nichi + RR
        # 16 Digits: YYYY(4) + Venue(2) + Kai(2) + Nichi(2) + RR(2) ? No thats 12.
        # Actually loader.py says:
        # CONCAT(r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango)
        # = 4 + 2 + 2 + 2 + 2 = 12 digits.
        # But wait, date is separate?
        # loader.py: race_id = YYYY + Venue + Kai + Nichi + RR.
        # apd_sokuho columns: kaisai_nen(4), kaisai_tsukihi(4), keibajo_code(2), kaisai_kai(2), kaisai_nichime(2), race_bango(2)
        # We need to map 12-digit race_id to these.
        
        if len(race_id) != 12:
            logger.warning(f"Invalid race_id length: {race_id} (Expected 12)")
            return None
            
        y = race_id[0:4]
        v = race_id[4:6]
        k = race_id[6:8]
        n = race_id[8:10]
        r = race_id[10:12]
        
        # Query
        query = text(f"""
            SELECT * FROM {table_name}
            WHERE kaisai_nen = :y
              AND keibajo_code = :v
              AND kaisai_kai = :k
              AND kaisai_nichime = :n
              AND race_bango = :r
            ORDER BY data_sakusei_nengappi DESC, happyo_tsukihi_jifun DESC
            LIMIT 1
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'y': y, 'v': v, 'k': k, 'n': n, 'r': r})
                
            if df.empty:
                return {}
                
            row = df.iloc[0]
            
            if bet_type == 'win':
                return self._parse_win_odds(row['odds_tansho'])
            elif bet_type == 'umaren':
                return self._parse_umaren_odds(row['odds_umaren'])
            # Add others later
            
        except Exception as e:
            logger.error(f"Error fetching odds for {race_id}: {e}")
            return None
            
        return {}

    def _parse_win_odds(self, raw_str: str) -> Dict[str, float]:
        # H(2) + O(4) + P(2) = 8
        chunk_size = 8
        odds_map = {}
        for i in range(0, len(raw_str), chunk_size):
            chunk = raw_str[i:i+chunk_size]
            if len(chunk) < chunk_size: break
            
            h_num = chunk[0:2].lstrip('0')
            o_str = chunk[2:6].lstrip('0')
            
            if not h_num or not o_str: continue
            
            try:
                # Odds format: "0191" -> 19.1
                odds = float(o_str) / 10.0
                odds_map[h_num] = odds
            except:
                continue
        return odds_map

    def _parse_umaren_odds(self, raw_str: str) -> Dict[str, float]:
        # H1(2) + H2(2) + O(6) + P(3) = 13 (Verified)
        chunk_size = 13
        odds_map = {}
        for i in range(0, len(raw_str), chunk_size):
            chunk = raw_str[i:i+chunk_size]
            if len(chunk) < chunk_size: break
            
            h1 = chunk[0:2].lstrip('0')
            h2 = chunk[2:4].lstrip('0')
            o_str = chunk[4:10].lstrip('0')
            
            if not h1 or not h2 or not o_str: continue
            
            try:
                # Odds format: "000281" -> 28.1 (Usually 10x implied for odds?)
                # Win was 4 digits for XX.X (e.g. 191 -> 19.1). (3 ints + 1 dec?) -> No default is just raw.
                # Win "0191" -> 19.1. So divider 10.
                # Umaren 6 digits "000281" -> 28.1. Divider 10.
                odds = float(o_str) / 10.0
                key = f"{int(h1):02d}-{int(h2):02d}" # Standardize key
                odds_map[key] = odds
            except:
                continue
        return odds_map

if __name__ == "__main__":
    # Test logic
    loader = RealTimeDataLoader()
    # Sample Test
    pass
