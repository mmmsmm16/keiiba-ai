"""
Logistics Features (Batch 2)
============================
Generates features related to transport and location mismatch:
- is_transported: Trainer affiliation vs Race venue region
- is_away_jockey: Jockey affiliation vs Race venue region
"""
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

# Venue to Region Mapping
VENUE_REGION_MAP = {
    '01': 'N', '02': 'N',  # Sapporo, Hakodate
    '03': 'E', '04': 'E', '05': 'E', '06': 'E',  # Fukushima, Niigata, Tokyo, Nakayama
    '07': 'W', '08': 'W', '09': 'W', '10': 'W',  # Chukyo, Kyoto, Hanshin, Kokura
}

# Affiliation Code Mapping (tozai_shozoku_code: 1=Miho/E, 2=Ritto/W)
AFFILIATION_REGION_MAP = {1: 'E', 2: 'W', 3: 'L', 4: 'O'}


class LogisticsFeatureGenerator:
    """
    Generates logistics features by loading jockey/trainer affiliations directly.
    """
    
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'postgres')
        password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    
    def _load_affiliations(self):
        """Load jockey and trainer affiliations from master tables."""
        jockey_q = "SELECT kishu_code as jockey_code, tozai_shozoku_code as jockey_belong FROM jvd_ks"
        trainer_q = "SELECT chokyoshi_code as trainer_code, tozai_shozoku_code as trainer_belong FROM jvd_ch"
        try:
            jockey_df = pd.read_sql(text(jockey_q), self.engine)
            trainer_df = pd.read_sql(text(trainer_q), self.engine)
            return jockey_df, trainer_df
        except Exception as e:
            logger.error(f"Error loading affiliations: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("LogisticsFeatureGenerator: Transforming...")
        df = df.copy()
        
        jockey_df, trainer_df = self._load_affiliations()
        
        # Derive venue_region
        if 'venue' in df.columns:
            df['venue_str'] = df['venue'].astype(str).str.zfill(2)
            df['venue_region'] = df['venue_str'].map(VENUE_REGION_MAP).fillna('U')
        else:
            df['venue_region'] = 'U'
        
        # Merge jockey affiliation
        if not jockey_df.empty and 'jockey_id' in df.columns:
            df['jockey_id_str'] = df['jockey_id'].astype(str).str.zfill(5)
            jockey_df['jockey_code'] = jockey_df['jockey_code'].astype(str).str.zfill(5)
            jockey_df['jockey_belong'] = pd.to_numeric(jockey_df['jockey_belong'], errors='coerce').fillna(0).astype(int)
            df = df.merge(jockey_df[['jockey_code', 'jockey_belong']], left_on='jockey_id_str', right_on='jockey_code', how='left')
            df['jockey_region'] = df['jockey_belong'].map(AFFILIATION_REGION_MAP).fillna('U')
            df.drop(columns=['jockey_id_str', 'jockey_code', 'jockey_belong'], inplace=True, errors='ignore')
        else:
            df['jockey_region'] = 'U'
        
        # Merge trainer affiliation
        if not trainer_df.empty and 'trainer_id' in df.columns:
            df['trainer_id_str'] = df['trainer_id'].astype(str).str.zfill(5)
            trainer_df['trainer_code'] = trainer_df['trainer_code'].astype(str).str.zfill(5)
            trainer_df['trainer_belong'] = pd.to_numeric(trainer_df['trainer_belong'], errors='coerce').fillna(0).astype(int)
            df = df.merge(trainer_df[['trainer_code', 'trainer_belong']], left_on='trainer_id_str', right_on='trainer_code', how='left')
            df['trainer_region'] = df['trainer_belong'].map(AFFILIATION_REGION_MAP).fillna('U')
            df.drop(columns=['trainer_id_str', 'trainer_code', 'trainer_belong'], inplace=True, errors='ignore')
        else:
            df['trainer_region'] = 'U'
        
        # Calculate flags
        df['is_transported'] = 0
        df.loc[(df['trainer_region'] == 'E') & (df['venue_region'] == 'W'), 'is_transported'] = 1
        df.loc[(df['trainer_region'] == 'W') & (df['venue_region'] == 'E'), 'is_transported'] = 1
        df.loc[(df['trainer_region'].isin(['E', 'W'])) & (df['venue_region'] == 'N'), 'is_transported'] = 1
        
        df['is_away_jockey'] = 0
        df.loc[(df['jockey_region'] == 'E') & (df['venue_region'] == 'W'), 'is_away_jockey'] = 1
        df.loc[(df['jockey_region'] == 'W') & (df['venue_region'] == 'E'), 'is_away_jockey'] = 1
        df.loc[(df['jockey_region'].isin(['E', 'W'])) & (df['venue_region'] == 'N'), 'is_away_jockey'] = 1
        
        output_cols = ['race_id', 'horse_number', 'horse_id', 'is_transported', 'is_away_jockey']
        return df[[c for c in output_cols if c in df.columns]]

