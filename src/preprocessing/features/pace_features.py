"""
Pace Features (Batch 3)
=======================
Generates features related to race pace and horse's pace experience:
- pace_diff: Front 3F - Last 3F
- pace_type: Categorical (Slow/Medium/Fast)
- Historical pace experience aggregations
"""
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)


class PaceFeatureGenerator:
    """
    Generates pace features by loading race-level pace data (zenhan_3f, kohan_3f)
    and computing horse-level historical pace experience.
    """
    
    def __init__(self):
        user = os.environ.get('POSTGRES_USER', 'postgres')
        password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
        host = os.environ.get('POSTGRES_HOST', 'db')
        port = os.environ.get('POSTGRES_PORT', '5432')
        dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
        self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
        self._pace_data = None
    
    def _load_pace_data(self):
        """Load pace data from jvd_ra."""
        if self._pace_data is not None:
            return self._pace_data
        
        logger.info("Loading pace data from jvd_ra...")
        query = """
            SELECT 
                CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
                       LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
                zenhan_3f,
                kohan_3f
            FROM jvd_ra
            WHERE zenhan_3f IS NOT NULL AND zenhan_3f != '000'
        """
        try:
            df = pd.read_sql(text(query), self.engine)
            # Convert to numeric (1/10 seconds -> seconds)
            df['zenhan_3f_sec'] = pd.to_numeric(df['zenhan_3f'], errors='coerce') / 10.0
            df['kohan_3f_sec'] = pd.to_numeric(df['kohan_3f'], errors='coerce') / 10.0
            df['pace_diff'] = df['zenhan_3f_sec'] - df['kohan_3f_sec']
            
            # Pace type
            df['pace_type'] = 1  # Medium
            df.loc[df['pace_diff'] > 2, 'pace_type'] = 2  # Fast
            df.loc[df['pace_diff'] < -2, 'pace_type'] = 0  # Slow
            
            logger.info(f"  Loaded {len(df)} races with pace data.")
            self._pace_data = df[['race_id', 'pace_diff', 'pace_type']].drop_duplicates()
            return self._pace_data
        except Exception as e:
            logger.error(f"Failed to load pace data: {e}")
            return pd.DataFrame()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pace features.
        Expects: race_id, horse_id, date columns.
        """
        logger.info("PaceFeatureGenerator: Transforming...")
        df = df.copy()
        
        pace_df = self._load_pace_data()
        
        if pace_df.empty:
            df['pace_diff'] = 0.0
            df['pace_type'] = 1
            df['horse_fast_pace_count'] = 0
            df['horse_slow_pace_count'] = 0
            df['horse_avg_pace_diff'] = 0.0
            df['horse_pace_versatility'] = 0.0
            return df[['race_id', 'horse_number', 'horse_id', 
                       'pace_diff', 'pace_type', 
                       'horse_fast_pace_count', 'horse_slow_pace_count',
                       'horse_avg_pace_diff', 'horse_pace_versatility']]
        
        # Merge current race pace
        df['race_id'] = df['race_id'].astype(str)
        pace_df['race_id'] = pace_df['race_id'].astype(str)
        df = df.merge(pace_df, on='race_id', how='left')
        df['pace_diff'] = df['pace_diff'].fillna(0.0)
        df['pace_type'] = df['pace_type'].fillna(1).astype(int)
        
        # Historical pace experience
        # Need to compute for each horse based on PAST races only
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['horse_id', 'date']).reset_index(drop=True)
        
        # Expanding window aggregation by horse_id (only past races)
        def compute_pace_history(group):
            group = group.sort_values('date')
            # Shift to only include past races
            past_pace = group['pace_diff'].shift(1)
            past_pace_type = group['pace_type'].shift(1)
            
            # Expanding mean/std (past only)
            group['horse_avg_pace_diff'] = past_pace.expanding(min_periods=1).mean().fillna(0.0)
            group['horse_pace_versatility'] = past_pace.expanding(min_periods=2).std().fillna(0.0)
            
            # Cumulative counts
            group['horse_fast_pace_count'] = (past_pace_type == 2).expanding().sum().fillna(0).astype(int)
            group['horse_slow_pace_count'] = (past_pace_type == 0).expanding().sum().fillna(0).astype(int)
            
            return group
        
        logger.info("  Computing historical pace experience per horse...")
        df = df.groupby('horse_id', group_keys=False).apply(compute_pace_history)
        
        # Output columns
        output_cols = [
            'race_id', 'horse_number', 'horse_id',
            'pace_diff', 'pace_type',
            'horse_fast_pace_count', 'horse_slow_pace_count',
            'horse_avg_pace_diff', 'horse_pace_versatility'
        ]
        
        return df[[c for c in output_cols if c in df.columns]]
