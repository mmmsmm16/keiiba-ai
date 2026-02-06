import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

class PCKeibaLoader:
    def __init__(self):
        # Load connection details from env or default
        self.user = os.environ.get('POSTGRES_USER', 'user')
        self.password = os.environ.get('POSTGRES_PASSWORD', 'password')
        self.host = os.environ.get('POSTGRES_HOST', 'db')
        self.port = os.environ.get('POSTGRES_PORT', '5432')
        self.dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        
        self.connection_str = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        self.engine = create_engine(self.connection_str)

    def get_latest_odds(self, target_date_str, race_id_map=None):
        """
        Fetch odd for a specific date (usually today).
        
        Args:
            target_date_str (str): "YYYY-MM-DD"
            race_id_map (dict, optional): Map from (year, month, day, venue, race_num) to race_id.
                                          Used if kaisai_nichisu is missing in DB.
            
        Returns:
            pd.DataFrame: Pipeline compatible odds dataframe
                columns: [race_id, ticket_type, combination, odds]
        """
        # Convert YYYY-MM-DD to kaisai_nen, kaisai_tsukihi (MMDD)
        dt = pd.Timestamp(target_date_str)
        k_nen = str(dt.year)
        k_tsukihi = dt.strftime('%m%d')
        
        self.race_id_map = race_id_map # Store for use in _make_race_id
        
        return self._process_all_odds(k_nen, k_tsukihi)

    def _process_all_odds(self, k_nen, k_tsukihi):
        all_dfs = []
        
        # O1: Tansho, Fukusho, Wakuren
        query = f"""
            SELECT * FROM apd_sokuho_o1 
            WHERE kaisai_nen = '{k_nen}' AND kaisai_tsukihi = '{k_tsukihi}'
        """
        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                # Iterate rows (races)
                for _, row in df.iterrows():
                    rid = self._make_race_id(row)
                    if not rid: continue # Skip if ID resolution failed
                    
                    # Tansho
                    t_str = row.get('odds_tansho', '')
                    if t_str and len(t_str) > 0:
                        all_dfs.append(self._parse_single_horse_odds(rid, 'win', t_str))
                        
                    # Fukusho
                    f_str = row.get('odds_fukusho', '')
                    if f_str and len(f_str) > 0:
                         all_dfs.append(self._parse_single_horse_odds(rid, 'place', f_str))
                         
                    # Wakuren
                    w_str = row.get('odds_wakuren', '')
                    if w_str and len(w_str) > 0:
                        all_dfs.append(self._parse_wakuren_odds(rid, w_str))
        except Exception as e:
            logger.error(f"Error processing O1: {e}")

        # O2: Umaren
        # query = f"""
        #     SELECT * FROM apd_sokuho_o2
        #     WHERE kaisai_nen = '{k_nen}' AND kaisai_tsukihi = '{k_tsukihi}'
        # """
        # try:
        #     df = pd.read_sql(query, self.engine)
        #     if not df.empty:
        #         for _, row in df.iterrows():
        #             rid = self._make_race_id(row)
        #             if not rid: continue

        #             s = row.get('odds_umaren', '')
        #             if s:
        #                 all_dfs.append(self._parse_combination_odds(rid, 'umaren', s, 15)) # 15 chars inferred?
        # except Exception as e:
        #     logger.error(f"Error processing O2: {e}")
            
        # O3: Wide
        # query = f"""
        #     SELECT * FROM apd_sokuho_o3
        #     WHERE kaisai_nen = '{k_nen}' AND kaisai_tsukihi = '{k_tsukihi}'
        # """
        # try:
        #     df = pd.read_sql(query, self.engine)
        #     if not df.empty:
        #         for _, row in df.iterrows():
        #             rid = self._make_race_id(row)
        #             if not rid: continue

        #             s = row.get('odds_wide', '')
        #             if s:
        #                 all_dfs.append(self._parse_combination_odds(rid, 'wide', s, 15)) 
        # except Exception as e:
        #     logger.error(f"Error processing O3: {e}")

        # O6: Sanrentan (Optional, heavy)
        # Maybe skip if not used? But user might want it.
        # But fixed width for 3ren is huge.
        
        if not all_dfs:
            return pd.DataFrame()
            
        return pd.concat(all_dfs, ignore_index=True)

    def _make_race_id(self, row):
        # Format: YYYY + PlaceCode(2) + Kai(2) + Day(2) + R(2)
        # If 'kaisai_nichisu' is missing, try to resolve via race_id_map
        y = str(row['kaisai_nen'])
        p = str(row['keibajo_code']).zfill(2)
        k = str(row['kaisai_kai']).zfill(2)
        r = str(row['race_bango']).zfill(2)
        
        # Check map first
        if self.race_id_map:
            # Map key: (int(Year), int(Month), int(Day), str(VenueCode), int(RaceNum))
            # Need to parse tsukihi
            md = str(row['kaisai_tsukihi']).zfill(4)
            m = int(md[:2])
            d = int(md[2:])
            
            key = (int(y), m, d, p, int(r))
            if key in self.race_id_map:
                return self.race_id_map[key]
        
        # Fallback (Requires kaisai_nichisu)
        d_raw = row.get('kaisai_nichisu')
        if d_raw:
            d_val = str(d_raw).zfill(2)
            return f"{y}{p}{k}{d_val}{r}"
            
        return None # Cannot construct ID


    def _parse_single_horse_odds(self, rid, ttype, raw_str):
        # Format depends on type
        # Win (Tansho): Horse(2) + Odds(4) + Pop(2) = 8 bytes
        # Place (Fukusho): Horse(2) + MinOdds(4) + MaxOdds(4) + Pop(2) = 12 bytes
        
        chunk_size = 8
        if ttype == 'place':
            chunk_size = 12
            
        records = []
        for i in range(0, len(raw_str), chunk_size):
            chunk = raw_str[i:i+chunk_size]
            if len(chunk) < chunk_size: break
            
            h = chunk[0:2].strip()
            
            if ttype == 'place':
                # Place: Horse(2) + Min(4) + Max(4) + Pop(2)
                o_str = chunk[2:6].strip()
            else:
                # Win: Horse(2) + Odds(4) + Pop(2)
                o_str = chunk[2:6].strip()
            
            if not h.isdigit(): continue
            try:
                if not o_str.isdigit(): continue
                val = int(o_str)
                if val == 0: continue 
                
                odds = val / 10.0
                records.append({
                    'race_id': rid,
                    'ticket_type': ttype,
                    'combination': str(int(h)),
                    'odds': odds
                })
            except:
                pass
        return pd.DataFrame(records)

    def _parse_wakuren_odds(self, rid, raw_str):
        # Format: W1(1) W2(1) Odds(6)
        # Note: Wakuren might use 8 chars too?
        # Standard JV structure for Wakuren is "11001234..." (1-1 -> 123.4)
        chunk_size = 8 
        records = []
        for i in range(0, len(raw_str), chunk_size):
            chunk = raw_str[i:i+chunk_size]
            if len(chunk) < chunk_size: break
            
            # W1, W2 are 1 char each?
            # "12001234" -> 1-2, 123.4
            
            f1 = chunk[0:1]
            f2 = chunk[1:2]
            o_str = chunk[2:8]
            
            if f1.isdigit() and f2.isdigit():
                try:
                    if not o_str.isdigit(): continue
                    val = int(o_str)
                    if val == 0: continue
                    
                    odds = val / 10.0
                    combo = f"{int(f1)}-{int(f2)}"
                    records.append({
                        'race_id': rid,
                        'ticket_type': 'wakuren',
                        'combination': combo,
                        'odds': odds
                    })
                except:
                    pass
        return pd.DataFrame(records)

    def _parse_combination_odds(self, rid, ttype, raw_str, chunk_size=15):
        # Generic parser for Umaren/Wide etc.
        # Based on Umaren inspection: "010200000005501" -> 15 chars.
        # H1(2) H2(2) Odds(6) Ninki(5??) -> 2+2+6+5=15?
        
        records = []
        for i in range(0, len(raw_str), chunk_size):
            chunk = raw_str[i:i+chunk_size]
            if len(chunk) < chunk_size: break
            h1 = chunk[0:2]
            h2 = chunk[2:4]
            o_str = chunk[4:10] # 6 chars odds
            
            if h1.isdigit() and h2.isdigit():
                 try:
                    if not o_str.isdigit(): continue
                    val = int(o_str)
                    if val == 0: continue
                    
                    odds = val / 10.0
                    combo = f"{int(h1)}-{int(h2)}"
                    
                    records.append({
                        'race_id': rid,
                        'ticket_type': ttype,
                        'combination': combo,
                        'odds': odds
                    })
                 except:
                     pass
        return pd.DataFrame(records)
