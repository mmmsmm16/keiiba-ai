"""
Phase 7: Payout Data Loader
jvd_hrテーブルから払戻データをロード

Usage:
    from utils.payout_loader import PayoutLoader
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([2021, 2022, 2023, 2024])
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_engine():
    """PostgreSQL接続エンジンを取得"""
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


class PayoutLoader:
    """払戻データローダー"""
    
    def __init__(self):
        self.engine = get_db_engine()
    
    def load_payout_dataframe(self, years: List[int]) -> pd.DataFrame:
        """
        jvd_hrテーブルから払戻データを取得
        
        Returns:
            DataFrame with race_id and haraimodoshi columns
        """
        logger.info(f"払戻データ(jvd_hr)をロード中... Years={years}")
        years_str = ",".join([f"'{y}'" for y in years])
        
        query = text(f"""
            SELECT 
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
                haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
                haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
                haraimodoshi_umaren_2a, haraimodoshi_umaren_2b,
                haraimodoshi_umaren_3a, haraimodoshi_umaren_3b,
                haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
                haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
                haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
                haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
                haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
                haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
                haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b,
                haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b,
                haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
            FROM jvd_hr
            WHERE kaisai_nen IN ({years_str})
        """)
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"払戻データロード完了: {len(df)} 件")
            return df
        except Exception as e:
            logger.error(f"払戻データロードエラー: {e}")
            return pd.DataFrame()
    
    def build_payout_map(self, df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        払戻データをマップ形式に変換
        
        Returns:
            Dict[race_id] -> Dict[ticket_type] -> Dict[combination] -> payout
            
        Example:
            {'202401010101': {
                'tansho': {'01': 320},
                'umaren': {'0102': 1520},
                'sanrenpuku': {'010203': 4560},
                'sanrentan': {'010203': 12340}
            }}
        """
        payout_map = {}
        
        for _, row in df.iterrows():
            rid = row['race_id']
            
            if rid not in payout_map:
                payout_map[rid] = {
                    'tansho': {},
                    'umaren': {},
                    'sanrenpuku': {},
                    'sanrentan': {}
                }
            
            # Parse helper
            def parse_pay(prefix: str, count: int) -> Dict[str, int]:
                result = {}
                for k in range(1, count + 1):
                    comb = row.get(f'{prefix}_{k}a')
                    pay = row.get(f'{prefix}_{k}b')
                    if comb and pay and str(comb).strip():
                        try:
                            result[str(comb).strip()] = int(float(str(pay).strip()))
                        except:
                            pass
                return result
            
            # Parse each ticket type
            payout_map[rid]['tansho'].update(parse_pay('haraimodoshi_tansho', 1))
            payout_map[rid]['umaren'].update(parse_pay('haraimodoshi_umaren', 3))
            payout_map[rid]['sanrenpuku'].update(parse_pay('haraimodoshi_sanrenpuku', 3))
            payout_map[rid]['sanrentan'].update(parse_pay('haraimodoshi_sanrentan', 6))
        
        return payout_map
    
    def load_payout_map(self, years: List[int]) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        払戻マップを直接ロード（便利関数）
        """
        df = self.load_payout_dataframe(years)
        if df.empty:
            return {}
        return self.build_payout_map(df)


def format_combination(horses: List[int], ordered: bool = False) -> str:
    """
    馬番のリストを組み合わせ文字列に変換
    
    Args:
        horses: [1, 3, 5] など
        ordered: Trueなら順あり（三連単）、Falseなら順不同（馬連/三連複）
    
    Returns:
        "010305" など
    """
    if ordered:
        formatted = horses
    else:
        formatted = sorted(horses)
    
    return "".join([f"{h:02}" for h in formatted])


def check_hit(
    bet_horses: List[int],
    payout_map: Dict,
    race_id: str,
    ticket_type: str,
    ordered: bool = False
) -> int:
    """
    的中判定と払戻金額を返す
    
    Returns:
        払戻金額（100円当たり）。不的中なら0
    """
    if race_id not in payout_map:
        return 0
    
    if ticket_type not in payout_map[race_id]:
        return 0
    
    comb_str = format_combination(bet_horses, ordered)
    return payout_map[race_id][ticket_type].get(comb_str, 0)


if __name__ == "__main__":
    # Test
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([2024])
    
    logger.info(f"Loaded payout for {len(payout_map)} races")
    
    # Sample
    for race_id, payouts in list(payout_map.items())[:3]:
        logger.info(f"Race {race_id}:")
        for ticket_type, combos in payouts.items():
            if combos:
                logger.info(f"  {ticket_type}: {list(combos.items())[:3]}")
