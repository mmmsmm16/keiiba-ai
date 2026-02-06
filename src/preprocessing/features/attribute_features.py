"""
Attribute Features (Batch 2)
============================
Generates Jockey and Trainer attribute features:
- Age (at race time)
- Career years
- Affiliation (East/West)
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import logging

logger = logging.getLogger(__name__)

class AttributeFeatureGenerator:
    """
    Generates attribute features for Jockeys and Trainers by loading
    master data (jvd_ks, jvd_ch) and merging onto the main DataFrame.
    """
    
    def __init__(self):
        # DB Connection
        user = os.environ.get('POSTGRES_USER', 'postgres')
        password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
        
        self._jockey_master = None
        self._trainer_master = None
    
    def _load_jockey_master(self):
        """Load jvd_ks (Jockey Master)"""
        if self._jockey_master is not None:
            return self._jockey_master
        
        logger.info("Loading jvd_ks (Jockey Master)...")
        query = """
            SELECT 
                kishu_code as jockey_code,
                seinengappi as jockey_birthdate,
                menkyo_kofu_nengappi as jockey_license_date,
                tozai_shozoku_code as jockey_belong_code
            FROM jvd_ks
            WHERE massho_kubun = '0' OR massho_kubun IS NULL
        """
        try:
            df = pd.read_sql(text(query), self.engine)
            logger.info(f"  Loaded {len(df)} jockeys.")
            self._jockey_master = df
            return df
        except Exception as e:
            logger.error(f"Failed to load jvd_ks: {e}")
            return pd.DataFrame()
    
    def _load_trainer_master(self):
        """Load jvd_ch (Trainer Master)"""
        if self._trainer_master is not None:
            return self._trainer_master
        
        logger.info("Loading jvd_ch (Trainer Master)...")
        query = """
            SELECT 
                chokyoshi_code as trainer_code,
                seinengappi as trainer_birthdate,
                menkyo_kofu_nengappi as trainer_license_date,
                tozai_shozoku_code as trainer_belong_code
            FROM jvd_ch
            WHERE massho_kubun = '0' OR massho_kubun IS NULL
        """
        try:
            df = pd.read_sql(text(query), self.engine)
            logger.info(f"  Loaded {len(df)} trainers.")
            self._trainer_master = df
            return df
        except Exception as e:
            logger.error(f"Failed to load jvd_ch: {e}")
            return pd.DataFrame()
    
    def _calc_age(self, race_date: pd.Series, birthdate_str: pd.Series) -> pd.Series:
        """Calculate age at race time"""
        birthdate = pd.to_datetime(birthdate_str, format='%Y%m%d', errors='coerce')
        age_days = (race_date - birthdate).dt.days
        return (age_days / 365.25).fillna(-1).astype(float)
    
    def _calc_career_years(self, race_date: pd.Series, license_date_str: pd.Series) -> pd.Series:
        """Calculate career years at race time"""
        license_date = pd.to_datetime(license_date_str, format='%Y%m%d', errors='coerce')
        career_days = (race_date - license_date).dt.days
        return (career_days / 365.25).fillna(-1).astype(float)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Jockey/Trainer attributes onto the main DataFrame.
        Expects: jockey_id, trainer_id, date columns.
        """
        logger.info("AttributeFeatureGenerator: Transforming...")
        
        # Ensure date column
        if 'date' not in df.columns:
            logger.warning("'date' column not found. Skipping AttributeFeatureGenerator.")
            return df[['race_id', 'horse_number', 'horse_id']].copy()
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Jockey attributes
        jockey_df = self._load_jockey_master()
        if not jockey_df.empty and 'jockey_id' in df.columns:
            df['jockey_id_str'] = df['jockey_id'].astype(str).str.zfill(5)
            jockey_df['jockey_code'] = jockey_df['jockey_code'].astype(str).str.zfill(5)
            df = df.merge(jockey_df, left_on='jockey_id_str', right_on='jockey_code', how='left')
            
            df['jockey_age'] = self._calc_age(df['date'], df['jockey_birthdate'])
            df['jockey_career_years'] = self._calc_career_years(df['date'], df['jockey_license_date'])
            # jockey_belong_code: 1=美浦(East), 2=栗東(West), 3=地方, 4=海外
            df['jockey_belong_code'] = pd.to_numeric(df['jockey_belong_code'], errors='coerce').fillna(0).astype(int)
            
            df.drop(columns=['jockey_id_str', 'jockey_code', 'jockey_birthdate', 'jockey_license_date'], inplace=True, errors='ignore')
        else:
            df['jockey_age'] = -1
            df['jockey_career_years'] = -1
            df['jockey_belong_code'] = 0
        
        # Trainer attributes
        trainer_df = self._load_trainer_master()
        if not trainer_df.empty and 'trainer_id' in df.columns:
            df['trainer_id_str'] = df['trainer_id'].astype(str).str.zfill(5)
            trainer_df['trainer_code'] = trainer_df['trainer_code'].astype(str).str.zfill(5)
            df = df.merge(trainer_df, left_on='trainer_id_str', right_on='trainer_code', how='left')
            
            df['trainer_age'] = self._calc_age(df['date'], df['trainer_birthdate'])
            df['trainer_career_years'] = self._calc_career_years(df['date'], df['trainer_license_date'])
            df['trainer_belong_code'] = pd.to_numeric(df['trainer_belong_code'], errors='coerce').fillna(0).astype(int)
            
            df.drop(columns=['trainer_id_str', 'trainer_code', 'trainer_birthdate', 'trainer_license_date'], inplace=True, errors='ignore')
        else:
            df['trainer_age'] = -1
            df['trainer_career_years'] = -1
            df['trainer_belong_code'] = 0

        # Output columns
        output_cols = [
            'race_id', 'horse_number', 'horse_id',
            'jockey_age', 'jockey_career_years', 'jockey_belong_code',
            'trainer_age', 'trainer_career_years', 'trainer_belong_code'
        ]
        
        return df[[c for c in output_cols if c in df.columns]]
