import pandas as pd
import sqlalchemy
import logging
from typing import Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class OddsFetcher:
    def __init__(self, db_url: str):
        self.engine = sqlalchemy.create_engine(db_url)
        
    def fetch_odds(self, race_id: str, max_retries: int = 3) -> Dict:
        """
        Fetch Win (Tansho) and Umaren odds for a given race_id.
        Returns:
            {
                'tansho': {horse_num: odds, ...},
                'umaren': {(h1, h2): odds, ...},
                'timestamp': str
            }
        """
        # Parse race_id elements from 16-digit string or whatever format
        # Usually race_id in DB: kaisai_nen, keibajo, kai, nichime, race_no
        # My system uses 202401010101 (12 digits) or JRA-VAN format?
        # The project uses PC-KEIBA schema.
        # My race_id usually: YYYYMMDDJJRR (12 chars)?
        # Actually loader.py constructs it: YYYY + Place + Kai + Day + Race
        # Let's assume input race_id matches the DB concatenation logic.
        
        # However, apd_sokuho tables have separate columns.
        # kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango
        
        # Need to parse race_id back to components if not provided.
        # Or just query by constructing the ID on the fly in SQL?
        # Most reliable: query WHERE CONCAT(...) = race_id
        
        retries = 0
        while retries < max_retries:
            try:
                with self.engine.connect() as conn:
                     # 1. Win Odds (o1)
                    query_o1 = sqlalchemy.text("""
                        SELECT * FROM apd_sokuho_o1 
                        WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) = :rid
                        ORDER BY happyo_tsukihi_jifun DESC LIMIT 1
                    """)
                    row_o1 = conn.execute(query_o1, {"rid": race_id}).fetchone()
                    
                    # 2. Umaren Odds (o2)
                    query_o2 = sqlalchemy.text("""
                        SELECT * FROM apd_sokuho_o2
                        WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) = :rid
                        ORDER BY happyo_tsukihi_jifun DESC LIMIT 1
                    """)
                    row_o2 = conn.execute(query_o2, {"rid": race_id}).fetchone()
                    
                    if not row_o1 and not row_o2:
                        # Odds not found
                        logger.warning(f"No odds found for {race_id} (Attempt {retries+1})")
                        time.sleep(1)
                        retries += 1
                        continue
                        
                    res = {
                        'tansho': {},
                        'umaren': {},
                        'timestamp': None
                    }
                    
                    # Parse O1 (Win)
                    if row_o1:
                        # columns are accessible by name in sqlalchemy row?
                        # RowMapping keys?
                        # In text query, names should be preserved.
                        # odds_tansho, odds_fukusho are large strings.
                        # Parse logic needed (from loader.py)
                        
                        # Get column index or mapping. 
                        # Use _mapping
                        cols = row_o1._mapping
                        ts = cols.get('happyo_tsukihi_jifun')
                        res['timestamp'] = ts
                        
                        o_str = cols.get('odds_tansho')
                        if o_str:
                             res['tansho'] = self._parse_tansho(o_str)
                             
                    # Parse O2 (Umaren)
                    if row_o2:
                        cols = row_o2._mapping
                        o_str = cols.get('odds_umaren')
                        if o_str:
                            res['umaren'] = self._parse_umaren(o_str)
                            
                    return res
                    
            except Exception as e:
                logger.error(f"Error fetching odds: {e}")
                retries += 1
                time.sleep(2)
                
        return {} # Failed

    def _parse_tansho(self, s: str) -> Dict[str, float]:
        # Chunk 5 chars: [Horse 2][Odds 3] (Scaled x10)?
        # Check src/preprocessing/loader.py for specific format
        # Actually loader.py says:
        # odds_tansho: 14 chars header? No.
        # "12 chars per horse" in Place? Win is different.
        # Let's re-verify inspect_odds_format.py results if possible.
        # Standard JRA-VAN Data Lab spec:
        # Tansho: Horse(2) + Odds(4) + Pop(2) = 8 chars?
        # Loader logic:
        #   h_num = int(block[0:2])
        #   odds = int(block[2:6]) / 10.0
        #   pop = int(block[6:8])
        # Assume 8 chars per horse.
        res = {}
        chunk = 8
        for i in range(0, len(s), chunk):
            block = s[i:i+chunk]
            if len(block) < chunk: break
            try:
                h = block[0:2] # Keep as str "01"? or int?
                # User config maps horse_number (int) to payout keys (str "01").
                # Let's use int for consistent logic with predictions, then format to str when needed.
                h_int = int(h)
                o_val = int(block[2:6]) / 10.0
                if o_val > 0:
                    res[h_int] = o_val
            except: pass
        return res

    def _parse_umaren(self, s: str) -> Dict[Tuple[int, int], float]:
        # Standard: Horse1(2) + Horse2(2) + Odds(6) + Pop(3) = 13 chars
        res = {}
        chunk = 13
        for i in range(0, len(s), chunk):
            block = s[i:i+chunk]
            if len(block) < chunk: break
            try:
                h1 = int(block[0:2])
                h2 = int(block[2:4])
                o_val = int(block[4:10]) / 10.0
                if o_val > 0:
                    key = tuple(sorted((h1, h2)))
                    res[key] = o_val
            except: pass
        return res
