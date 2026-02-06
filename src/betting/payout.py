
from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, List, Optional
from sqlalchemy import create_engine
import os

logger = logging.getLogger(__name__)

class PayoutProvider(ABC):
    @abstractmethod
    def get_payouts(self, race_ids: List[str]) -> pd.DataFrame:
        """
        Get payouts for multiple races.
        
        Args:
            race_ids: List of race IDs.
            
        Returns:
            pd.DataFrame: Normalized payout table.
            Columns: ['race_id', 'bet_type', 'selections', 'payout', 'popularity']
            'selections' should be a standardized string (e.g., "1-2" sorted or ordered as per outcome).
            'payout' is per 100 yen.
        """
        pass

class JraDbPayoutProvider(PayoutProvider):
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
            
        # Mapping for JVD columns
        self.type_map = {
            'tansho': 'win',
            'fukusho': 'place',
            'wakuren': 'wakuren',
            'umaren': 'umaren',
            'wide': 'wide',
            'umatan': 'umatan',
            'sanrenpuku': 'sanrenpuku',
            'sanrentan': 'sanrentan'
        }

    def get_payouts(self, race_ids: List[str]) -> pd.DataFrame:
        if not race_ids:
            return pd.DataFrame()
        
        # Build query
        # Since race_id in JVD is split (Year/Venue/etc), we might need to join or assume standardization.
        # Assuming our 'race_id' is the standard 16-digit or so ID used in this project.
        # "202401010101" -> Year(4)+Venue(2)+Kai(2)+Day(2)+Race(2)
        
        # We can construct a WHERE clause.
        # It's more efficient to fetch by year if many, but let's do IN clause if strictly list.
        # Payout table `jvd_hr` uses split keys.
        
        # Construct KeyTuples
        # race_id validation
        # We need a robust way to map race_id -> keys.
        
        # Alternative: query by year/month from the race_ids to narrow down?
        # Let's assume passed race_ids are 2025+ and query efficiently?
        # Actually, `jvd_hr` doesn't have `race_id` column. We must construct keys.
        
        # Generate filters
        conditions = []
        for rid in race_ids:
            if len(rid) >= 12:
                # 2024 01 01 01 01
                y = rid[0:4]
                p = rid[4:6]
                k = rid[6:8]
                d = rid[8:10]
                r = rid[10:12]
                conditions.append(f"('{y}', '{p}', '{k}', '{d}', '{r}')")
        
        if not conditions:
            return pd.DataFrame()
            
        keys_str = ",".join(conditions)
        # Limit batch size if too huge, but usually fine for daily/batch.
        
        query = f"""
        SELECT *
        FROM jvd_hr
        WHERE (kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({keys_str})
        """
        
        try:
            df = pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Error fetching payouts: {e}")
            return pd.DataFrame()
        
        results = []
        for _, row in df.iterrows():
            rid = f"{row['kaisai_nen']}{row['keibajo_code']}{row['kaisai_kai']}{row['kaisai_nichime']}{row['race_bango']}"
            
            for jvd_type, std_type in self.type_map.items():
                # Loop through potential multiple payouts (1-3 for most, 1-5 for Place/Wide?)
                # JVD usually has 1a, 1b through 3a, 3b etc.
                # Max 5 for Place? Max 7 for Wide (Tie)?
                # Standard Loop 1 to 10 safe.
                
                for i in range(1, 11):
                    col_a = f"haraimodoshi_{jvd_type}_{i}a"
                    col_b = f"haraimodoshi_{jvd_type}_{i}b"
                    col_c = f"haraimodoshi_{jvd_type}_{i}c"
                    
                    if col_a not in df.columns or col_b not in df.columns:
                        break
                        
                    val_a = row.get(col_a)
                    val_b = row.get(col_b)
                    
                    if pd.isna(val_a) or val_a is None or str(val_a).strip() == "":
                        continue
                        
                    val_a = str(val_a).strip()
                    # val_b is padded string "000000180"
                    
                    try:
                        payout = int(val_b)
                        if payout == 0: continue
                        
                        # Normalize selections
                        # Simple format adjustment if needed
                        # Win/Place: "05" -> "5"
                        # Umaren: "0506" -> "5-6"
                        # Sanrentan: "010203" -> "1-2-3"
                        # Logic depends on type
                        
                        combo_str = self._normalize_combo(std_type, val_a)
                        
                        results.append({
                            'race_id': rid,
                            'bet_type': std_type,
                            'selections': combo_str,
                            'payout': payout,
                            # 'popularity': ...
                        })
                    except:
                        continue
                        
        return pd.DataFrame(results)

    def _normalize_combo(self, bet_type, val_a):
        # Remove leading zeros?
        # Basic logic: chunk by 2 chars?
        # Win(2), Place(2), Wakuren(1+1?), Umaren(2+2), Wide(2+2), Umatan(2+2)
        # Sanrenpuku(2+2+2), Sanrentan(2+2+2)
        
        # JRA VAN Data spec:
        # Wakuren: 1 char + 1 char? e.g. "12" -> Frame 1 - Frame 2?
        # Yes, Wakuren is 2 chars total typically in this column, but let's verify.
        # Actually in PC-KEIBA DB, columns are generous char/varchar.
        # Given "06" for Win, it seems 2-char horse numbers.
        
        if bet_type == 'wakuren':
            # "12" -> 1-2. "0102"? No, typically just "12".
            # If 2 chars:
            if len(val_a) == 2:
                return f"{int(val_a[0])}-{int(val_a[1])}"
            # If 4 chars "0102"
            if len(val_a) == 4:
                return f"{int(val_a[:2])}-{int(val_a[2:])}"
            return val_a
            
        # Others likely 2-char blocks
        parts = []
        for i in range(0, len(val_a), 2):
            part = val_a[i:i+2]
            if len(part) == 2:
                try:
                    parts.append(str(int(part)))
                except:
                    parts.append(part)
        
        # Special Sort for Unordered bets??
        # Usually Payout Table returns "Low-High" for combinations like Umaren.
        # But tickets might be "High-Low" if we are not careful.
        # However, Ticket class does not enforce sorting.
        # We should enforce sorting here and in Ticket to ensure match.
        
        if bet_type in ['umaren', 'wide', 'sanrenpuku', 'wakuren']:
             # Sort integers
             try:
                 parts_int = sorted([int(p) for p in parts])
                 return "-".join(map(str, parts_int))
             except:
                 pass
                 
        return "-".join(parts)
