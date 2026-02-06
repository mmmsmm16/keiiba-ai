"""
Corner Position Features (Batch 4)
==================================
Generates features related to corner position changes:
- corner_position_change: (4th corner pos - 1st corner pos) / field_size
- makuri_score: Degree of advancing through the field
- Historical patterns of corner behavior
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CornerPositionFeatureGenerator:
    """
    Generates corner position change features from pass_1~pass_4 columns.
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate corner position features.
        Expects: pass_1, pass_2, pass_3, pass_4, field_size columns.
        """
        logger.info("CornerPositionFeatureGenerator: Transforming...")
        df = df.copy()
        
        # Ensure pass columns are numeric
        for c in ['pass_1', 'pass_2', 'pass_3', 'pass_4']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
            else:
                df[c] = 0
        
        # Field size (fallback to max(pass_4) per race if not available)
        if 'field_size' not in df.columns:
            df['field_size'] = df.groupby('race_id')['pass_4'].transform('max').clip(lower=1)
        df['field_size'] = df['field_size'].clip(lower=1)
        
        # Corner position change (normalized by field size)
        # Positive = moved forward (makuri), Negative = dropped back
        df['corner_position_change'] = 0.0
        has_corners = (df['pass_1'] > 0) & (df['pass_4'] > 0)
        df.loc[has_corners, 'corner_position_change'] = (
            (df.loc[has_corners, 'pass_1'] - df.loc[has_corners, 'pass_4']) / 
            df.loc[has_corners, 'field_size']
        )
        
        # Makuri score (how many positions advanced from 1st to 4th corner)
        df['makuri_positions'] = 0
        df.loc[has_corners, 'makuri_positions'] = (
            df.loc[has_corners, 'pass_1'] - df.loc[has_corners, 'pass_4']
        ).clip(lower=0)  # Only positive advancement
        
        # Late charge (advancement from 3rd to 4th corner)
        has_34 = (df['pass_3'] > 0) & (df['pass_4'] > 0)
        df['late_charge'] = 0
        df.loc[has_34, 'late_charge'] = (
            df.loc[has_34, 'pass_3'] - df.loc[has_34, 'pass_4']
        ).clip(lower=0)
        
        # Historical patterns per horse (using expanding window)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['horse_id', 'date']).reset_index(drop=True)
        
        def compute_corner_history(group):
            group = group.sort_values('date')
            # Shift to only include past races
            past_change = group['corner_position_change'].shift(1)
            past_makuri = group['makuri_positions'].shift(1)
            
            # Expanding aggregations
            group['horse_avg_corner_change'] = past_change.expanding(min_periods=1).mean().fillna(0.0)
            group['horse_makuri_rate'] = (past_makuri > 0).expanding().mean().fillna(0.0)
            group['horse_total_makuri'] = past_makuri.expanding().sum().fillna(0).astype(int)
            
            return group
        
        logger.info("  Computing historical corner patterns per horse...")
        df = df.groupby('horse_id', group_keys=False).apply(compute_corner_history)
        
        # Output columns
        output_cols = [
            'race_id', 'horse_number', 'horse_id',
            'corner_position_change', 'makuri_positions', 'late_charge',
            'horse_avg_corner_change', 'horse_makuri_rate', 'horse_total_makuri'
        ]
        
        return df[[c for c in output_cols if c in df.columns]]
