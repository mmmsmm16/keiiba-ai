"""
Incremental Category Aggregator

Manages stateful accumulation of category statistics (Jockey, Sire, etc.)
to allow efficient walk-forward preprocessing without re-reading full history.

[v12 P1] Added Bayesian smoothing for rate calculations.
"""
import pandas as pd
import numpy as np
import logging
import os
import joblib

logger = logging.getLogger(__name__)

# P1: ベイズ平滑化パラメータ (category_aggregators.py と同じ値)
PRIOR_SAMPLES = 30.0  # 強めに設定（専門家のアドバイスに基づき正則化を強化）
GLOBAL_WIN_RATE = 0.07
GLOBAL_TOP3_RATE = 0.21

class IncrementalCategoryAggregator:
    def __init__(self):
        # State: dict of {category_col: DataFrame(index=category_val, columns=[count, wins, top3])}
        self.states = {}
        
    def fit(self, df: pd.DataFrame):
        """
        Initial training on historical data.
        Calculates cumulative stats up to the end of df.
        This df is assumed to be the 'past' data.
        """
        logger.info("Initializing incremental state from historical data...")
        self.states = {}
        
        # Define targets (Same as CategoryAggregator)
        # 1. Basic
        targets = ['jockey_id', 'trainer_id', 'sire_id', 'class_level']
        
        # 2. Contextual
        # Context keys needs special handling (combine cols)
        # For simplicity, we create temporary combined columns or use MultiIndex
        # MultiIndex state is cleaner.
        
        # Helper to update state
        def update_state(d, keys, prefix):
            # Aggregation by keys
            # wins/top3 logic
            # Use temporary dataframe for aggregation to avoid polluting original d
            # Copying 800k rows is cheap enough (100MB) relative to safety
            temp = d[keys].copy()
            temp['is_win'] = (d['rank'] == 1).astype(int)
            temp['is_top3'] = (d['rank'] <= 3).astype(int)
            
            # Group by keys
            g = temp.groupby(keys)[['is_win', 'is_top3']].agg(['sum', 'count'])
            # g columns: (is_win, sum), (is_win, count), (is_top3, sum), ...
            # We only need count once.
            
            stats = pd.DataFrame(index=g.index)
            stats['count'] = g[('is_win', 'count')]
            stats['wins'] = g[('is_win', 'sum')]
            stats['top3'] = g[('is_top3', 'sum')]
            
            self.states[prefix] = stats
            
            del temp
            
        # Ensure distance_cat exists
        if 'distance' in df.columns and 'distance_cat' not in df.columns:
             df['distance_cat'] = pd.cut(
                df['distance'], 
                bins=[0, 1399, 1899, 2399, 9999], 
                labels=['Sprint', 'Mile', 'Intermediate', 'Long']
            )

        # 1. Basic Stats
        for col in targets:
            if col in df.columns:
                update_state(df, [col], col)
                
        # 2. Context Stats (Interactions)
        if 'venue' in df.columns:
            if 'jockey_id' in df.columns: update_state(df, ['jockey_id', 'venue'], 'jockey_venue')
            if 'trainer_id' in df.columns: update_state(df, ['trainer_id', 'venue'], 'trainer_venue')
            if 'sire_id' in df.columns: update_state(df, ['sire_id', 'venue'], 'sire_venue')
            
        if 'sire_id' in df.columns and 'distance_cat' in df.columns:
            update_state(df, ['sire_id', 'distance_cat'], 'sire_dist')
            
        if 'jockey_id' in df.columns and 'trainer_id' in df.columns:
            update_state(df, ['jockey_id', 'trainer_id'], 'jockey_trainer')
        
        logger.info(f"Initialized states for {list(self.states.keys())}")

    def transform_update(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply current stats to df (transform), THEN update state with df contents.
        Crucial: Features for df must use State BEFORE df (anti-leakage).
        """
        logger.info("Incremental transform & update...")
        
        # Ensure distance_cat
        if 'distance' in df.columns and 'distance_cat' not in df.columns:
             df['distance_cat'] = pd.cut(
                df['distance'], 
                bins=[0, 1399, 1899, 2399, 9999], 
                labels=['Sprint', 'Mile', 'Intermediate', 'Long']
            )

        # Mapping of {state_key: (join_cols, feature_prefix)}
        # Naming alignment with CategoryAggregator
        mappings = {
            'jockey_id': (['jockey_id'], 'jockey_id'),
            'trainer_id': (['trainer_id'], 'trainer_id'),
            'sire_id': (['sire_id'], 'sire_id'),
            'class_level': (['class_level'], 'class_level'),
            'jockey_venue': (['jockey_id', 'venue'], 'jockey_course'),
            'trainer_venue': (['trainer_id', 'venue'], 'trainer_course'),
            'sire_venue': (['sire_id', 'venue'], 'sire_course'),
            'sire_dist': (['sire_id', 'distance_cat'], 'sire_dist'),
            'jockey_trainer': (['jockey_id', 'trainer_id'], 'jockey_trainer')
        }

        for key, (join_cols, prefix) in mappings.items():
            if key not in self.states:
                continue
            
            state_df = self.states[key]
            if not all(k in df.columns for k in join_cols):
                continue
                
            # Merge state
            merged = df[join_cols].merge(state_df, left_on=join_cols, right_index=True, how='left')
            merged[['count', 'wins', 'top3']] = merged[['count', 'wins', 'top3']].fillna(0)
            
            # Apply Bayesian Smoothing
            df[f'{prefix}_n_races'] = merged['count']
            df[f'{prefix}_win_rate'] = (merged['wins'] + PRIOR_SAMPLES * GLOBAL_WIN_RATE) / (merged['count'] + PRIOR_SAMPLES)
            df[f'{prefix}_top3_rate'] = (merged['top3'] + PRIOR_SAMPLES * GLOBAL_TOP3_RATE) / (merged['count'] + PRIOR_SAMPLES)
            
            # Now UPDATE state with current df stats
            self._update_single_state(df, key, join_cols)

            
        return df

    def _update_single_state(self, df, key, join_cols):
        # Calculate current batch stats
        d = df.copy()
        d['is_win'] = (d['rank'] == 1).astype(int)
        d['is_top3'] = (d['rank'] <= 3).astype(int)
        
        g = d.groupby(join_cols)[['is_win', 'is_top3']].agg(['sum', 'count'])
        
        current_stats = pd.DataFrame(index=g.index)
        current_stats['count'] = g[('is_win', 'count')]
        current_stats['wins'] = g[('is_win', 'sum')]
        current_stats['top3'] = g[('is_top3', 'sum')]
        
        # Merge with existing state (add)
        if key in self.states:
            old_state = self.states[key]
            # Align indices
            # Combine indices
            all_indices = old_state.index.union(current_stats.index)
            old_state = old_state.reindex(all_indices).fillna(0)
            current_stats = current_stats.reindex(all_indices).fillna(0)
            
            self.states[key] = old_state + current_stats
        else:
            self.states[key] = current_stats

    def save_state(self, path):
        joblib.dump(self.states, path)
        
    def load_state(self, path):
        self.states = joblib.load(path)
