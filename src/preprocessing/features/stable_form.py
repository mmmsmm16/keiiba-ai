import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.9] 厩舎/騎手の直近フォーム (Stable Form)
    - 30日/60日間の勝率・複勝率
    - shift(1)して過去データのみを使用するロジックは jockey_stats と同じだが、期間が短い
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    out_cols = []
    
    # Use standard rolling logic for Trainer and Jockey
    # Targets: trainer_id, jockey_id
    
    # Prepare sorting by date
    df_sorted = df[['race_id', 'horse_number', 'horse_id', 'date', 'jockey_id', 'trainer_id', 'rank']].copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    
    # [Fix] Standardize IDs as strings and fill NaNs
    for id_col in ['jockey_id', 'trainer_id']:
        df_sorted[id_col] = df_sorted[id_col].fillna('99999').astype(str)

    df_sorted = df_sorted.sort_values('date')
    
    # [Fix] Exclude rank=0 (DNF/取消) from win/top3 statistics
    # rank=0 means horse didn't finish properly
    valid_rank = df_sorted['rank'] > 0
    df_sorted['is_win'] = ((df_sorted['rank'] == 1) & valid_rank).astype(int)
    df_sorted['is_top3'] = ((df_sorted['rank'] <= 3) & valid_rank).astype(int)
    
    for entity in ['trainer_id', 'jockey_id']:
        if entity not in df_sorted.columns: continue
        
        # Sort by Entity then Date
        df_entity = df_sorted.sort_values([entity, 'date'])
        
        # Group
        grp = df_entity.set_index('date').groupby(entity)
        
        for days in [30, 60]:
            window = f"{days}D"
            # Rolling closed='left' excludes today
            r = grp.rolling(window, closed='left')
            
            # Count, Sum Win, Sum Top3
            # NOTE: Groupby.rolling returns MultiIndex (Entity, Date). 
            # We need to map back to original index or merge ON (Entity, Date).
            # Merging on Entity+Date is tricky if multiple races same day.
            # But Rolling value is identical for same day same entity? Yes (closed=left).
            
            # Use transform-like approach or calc and merge.
            # Calc and merge is safer.
            
            stats = r[['is_win', 'is_top3', 'rank']].agg({
                'is_win': ['count', 'sum'],
                'is_top3': 'sum'
            })
            # Flatten columns: is_win_count, is_win_sum, is_top3_sum
            stats.columns = ['_'.join(c) for c in stats.columns]
            
            # Reset index to get Entity, Date columns
            stats = stats.reset_index()
            # stats has [entity, date, values...]
            # Duplicates on (entity, date) if multiple races same day? 
            # Yes, rolling result is repeated for each row in group? 
            # No, rolling on groupby yields result per input row.
            # So stats has same length as df_entity? Yes.
            # We can assign directly if index matches?
            # Rolling drops index? No.
            # Let's verify indices.
            
            # Actually, `grp.rolling` preserves the original index in recent pandas versions?
            # If so, simple assignment works.
            # But `set_index('date')` changed the index.
            
            # Strategy:
            # 1. Sort df_entity (already done)
            # 2. Use `on='date'` in rolling? `groupby(entity).rolling(on='date')`
            # This keeps other columns?
            # Let's try `df_entity.groupby(entity).rolling(window, on='date', closed='left')`
            # This returns MultiIndex (Entity, Index) ?
            
            # Safe way:
            r_obj = df_entity.set_index('date').groupby(entity).rolling(window, closed='left')
            
            # Extract values
            cnt = r_obj['is_win'].count().values # count of races
            win = r_obj['is_win'].sum().values
            top3 = r_obj['is_top3'].sum().values
            
            pfx = entity.replace('_id', '')
            
            # Assign to df_entity columns (in-place)
            # Order matches df_entity (sorted)
            df_entity[f'{pfx}_n_{days}d'] = cnt
            df_entity[f'{pfx}_win_{days}d'] = win
            df_entity[f'{pfx}_top3_{days}d'] = top3
            
            # Fix: values might be numpy array if aligned
            # Use Series to ensure index alignment or handle numpy array
            rate_win = win / np.maximum(cnt, 1)
            rate_top3 = top3 / np.maximum(cnt, 1)
            
            # Check if it is series (it should be if win/cnt are series)
            # But if some operation made it numpy, converting back or using np.nan_to_num
            if isinstance(rate_win, pd.Series):
                df_entity[f'{pfx}_win_rate_{days}d'] = rate_win.fillna(0)
                df_entity[f'{pfx}_top3_rate_{days}d'] = rate_top3.fillna(0)
            else:
                df_entity[f'{pfx}_win_rate_{days}d'] = np.nan_to_num(rate_win, nan=0.0)
                df_entity[f'{pfx}_top3_rate_{days}d'] = np.nan_to_num(rate_top3, nan=0.0)
            
            out_cols.extend([f'{pfx}_win_rate_{days}d', f'{pfx}_top3_rate_{days}d'])

        # Map back to df_sorted (merge on index or keys)
        # df_entity has new cols.
        # Minimal merge
        merge_cols = ['race_id', 'horse_number'] + [c for c in df_entity.columns if 'rate_' in c]
        # Drop duplicates if any (shouldn't be)
        # df_entity contains all rows? Yes.
        df_sorted = pd.merge(df_sorted, df_entity[merge_cols], on=['race_id', 'horse_number'], how='left', suffixes=('', '_new'))
        
        # Cleanup suffixes if needed
        # (Assuming no collision for first entity)

    return df_sorted[cols + out_cols].copy()
