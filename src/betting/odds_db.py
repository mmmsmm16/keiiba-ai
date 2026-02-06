
import pandas as pd
from sqlalchemy import create_engine
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

from src.betting.odds import OddsProvider

logger = logging.getLogger(__name__)

class JraDbOddsProvider(OddsProvider):
    """
    Fetches real odds from PC-KEIBA DB (apd_sokuho tables).
    """
    def __init__(self, engine=None):
        if engine is None:
            user = os.environ.get('POSTGRES_USER', 'postgres')
            password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
            host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
            port = os.environ.get('POSTGRES_PORT', '5433')
            dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
            connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            self.engine = create_engine(connection_str)
        else:
            self.engine = engine
            
        self.table_map = {
            'win': 'apd_sokuho_o1',
            'place': 'apd_sokuho_o1',
            'wakuren': 'apd_sokuho_o1',
            'umaren': 'apd_sokuho_o2',
            'wide': 'apd_sokuho_o3',
            'umatan': 'apd_sokuho_o4',
            'sanrenpuku': 'apd_sokuho_o5',
            'sanrentan': 'apd_sokuho_o6'
        }
        
        # Cache for Race + TicketType -> DataFrame
        self.cache = {}

    def get_odds(self, race_id: str, ticket_type: str, asof: datetime = None) -> pd.DataFrame:
        """
        Get odds. If asof is provided, finding efficient latest record before asof is hard 
        without full table scan or optimized index. 
        For backtest, we might assume 'latest available' or specific logic.
        """
        # Parsing race_id to DB keys
        # 202401010101 -> kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango
        if len(race_id) < 12: return pd.DataFrame()
        
        y = race_id[0:4]
        p = race_id[4:6]
        k = race_id[6:8]
        d = race_id[8:10]
        r = race_id[10:12]
        
        table = self.table_map.get(ticket_type)
        if not table: return pd.DataFrame()
        
        # Query DB
        # We assume we want the *latest* odds record if asof is None.
        # If asof is set, we want max(timestamp) <= asof.
        # The sokuho tables have `happyo_tsukihi_jifun` (MMDDHHmm). No year in column!
        # We must assume `kaisai_nen` is the year.
        
        # Optimization: Fetch one record order by time desc
        
        cols = "*" # Optimize later
        
        query = f"""
        SELECT * FROM {table}
        WHERE kaisai_nen = '{y}' 
          AND keibajo_code = '{p}'
          AND kaisai_kai = '{k}'
          AND kaisai_nichime = '{d}'
          AND race_bango = '{r}'
        ORDER BY happyo_tsukihi_jifun DESC
        LIMIT 20
        """
        
        try:
            df = pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return pd.DataFrame()
            
        if df.empty:
            return pd.DataFrame()
            
        # Filter by asof
        if asof:
            # Construct timestamp
            # happyo_tsukihi_jifun is MMDDHHMM
            def parse_time(row):
                t_str = row['happyo_tsukihi_jifun']
                if not t_str or len(t_str) < 8: return pd.NaT
                full_str = f"{row['kaisai_nen']}{t_str}"
                try:
                    return datetime.strptime(full_str, "%Y%m%d%H%M")
                except:
                    return pd.NaT

            df['timestamp'] = df.apply(parse_time, axis=1)
            df = df.dropna(subset=['timestamp'])
            
            # Keep only <= asof
            df = df[df['timestamp'] <= asof]
            if df.empty:
                return pd.DataFrame(columns=['combination', 'odds'])
            
            # Take latest
            target_row = df.iloc[0] # Already sorted DESC
        else:
            # Take latest available
            target_row = df.iloc[0]
            
        # Parse Odds String
        return self._parse_sokuho_row(target_row, ticket_type)

    def _parse_sokuho_row(self, row, ticket_type) -> pd.DataFrame:
        # Re-using logic from pckeiba_loader/build_odds_snapshot ideally.
        # But inline here for independence.
        
        # Win: odds_tansho (8 chars/horse)
        # Place: odds_fukusho (12 chars/horse)
        # Umaren: odds_umaren (15 chars)
        # Wide: odds_wide (15 chars)
        # Umatan: odds_umatan (14 chars: H1(2) H2(2) Odds(6) Ninki(4)?) -> Need verify
        # Sanrenpuku: odds_sanrenpuku (17 chars: H1(2) H2(2) H3(2) Odds(6) Ninki(5)?)
        # Sanrentan: odds_sanrentan (17 chars?)
        
        records = []
        raw = ""
        chunk_size = 0
        
        if ticket_type == 'win':
            raw = row.get('odds_tansho', '')
            chunk_size = 8 # H(2) O(4) P(2)
        elif ticket_type == 'place':
            raw = row.get('odds_fukusho', '')
            chunk_size = 12
        elif ticket_type == 'umaren':
            raw = row.get('odds_umaren', '')
            chunk_size = 15 # H(2) H(2) O(6) P(5)
        elif ticket_type == 'wide':
            raw = row.get('odds_wide', '')
            chunk_size = 15 
        elif ticket_type == 'umatan':
            raw = row.get('odds_umatan', '')
            chunk_size = 14 # H(2) H(2) O(6) P(4)? 
        elif ticket_type == 'sanrenpuku':
            raw = row.get('odds_sanrenpuku', '')
            chunk_size = 17 # H(2) H(2) H(2) O(6) P(5)?
        elif ticket_type == 'sanrentan':
            raw = row.get('odds_sanrentan', '')
            chunk_size = 17 # Check spec. Usually huge string.
            
        if not raw: return pd.DataFrame()
        
        # Parsing Loop
        length = len(raw)
        i = 0
        while i + chunk_size <= length:
            chunk = raw[i:i+chunk_size]
            i += chunk_size
            
            try:
                combo = ""
                odds_val = 0.0
                
                if ticket_type == 'win':
                    h = chunk[0:2].strip()
                    if not h.isdigit(): continue
                    o_str = chunk[2:6]
                    if not o_str.isdigit(): continue
                    odds_val = int(o_str) / 10.0
                    combo = str(int(h))
                    
                elif ticket_type == 'place':
                     # Place has min/max. Ticket usually uses Min or "Odds".
                     # H(2) Min(4) Max(4) P(2)
                     h = chunk[0:2].strip()
                     if not h.isdigit(): continue
                     o_str = chunk[2:6]
                     if not o_str.isdigit(): continue
                     odds_val = int(o_str) / 10.0 # Min odds
                     combo = str(int(h))
                     
                elif ticket_type == 'umaren' or ticket_type == 'wide':
                    h1 = chunk[0:2]
                    h2 = chunk[2:4]
                    if not h1.isdigit() or not h2.isdigit(): continue
                    o_str = chunk[4:10]
                    if not o_str.isdigit(): continue
                    odds_val = int(o_str) / 10.0
                    combo = f"{int(h1)}-{int(h2)}"
                    
                elif ticket_type == 'umatan':
                    h1 = chunk[0:2]
                    h2 = chunk[2:4]
                    if not h1.isdigit() or not h2.isdigit(): continue
                    o_str = chunk[4:10]
                    if not o_str.isdigit(): continue
                    odds_val = int(o_str) / 10.0
                    combo = f"{int(h1)}-{int(h2)}"
                    
                elif ticket_type == 'sanrenpuku':
                    h1 = chunk[0:2]
                    h2 = chunk[2:4]
                    h3 = chunk[4:6]
                    if not h1.isdigit(): continue
                    o_str = chunk[6:12]
                    if not o_str.isdigit(): continue
                    odds_val = int(o_str) / 10.0
                    combo = f"{int(h1)}-{int(h2)}-{int(h3)}"

                elif ticket_type == 'sanrentan':
                    # Structure guess: H1(2) H2(2) H3(2) Odds(6) Pop(5) = 17 ?
                    h1 = chunk[0:2]
                    h2 = chunk[2:4]
                    h3 = chunk[4:6]
                    if not h1.isdigit(): continue
                    o_str = chunk[6:12]
                    if not o_str.isdigit(): continue
                    odds_val = int(o_str) / 10.0
                    combo = f"{int(h1)}-{int(h2)}-{int(h3)}"
                
                if odds_val > 0:
                    records.append({
                         'combination': combo,
                         'odds': odds_val,
                         'ticket_type': ticket_type
                    })
            except:
                continue
                
        return pd.DataFrame(records)
