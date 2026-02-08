
import pandas as pd
import numpy as np
class TrackAptitudeFeatureGenerator:
    """
    馬場状態適性（Going Aptitude）特徴量を生成するジェネレータ
    馬ごとの「良」「稍重」「重」「不良」成績を集計する。
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Require: horse_id, going_code (int), rank (numeric), date (for expanding window)
        
        # 1. Sort by date to ensure expanding window is correct
        df_sorted = df.sort_values('date').copy()
        df_sorted['going_code'] = pd.to_numeric(df_sorted.get('going_code', 0), errors='coerce').fillna(0).astype(int)
        
        # 2. Pivot / Grouping
        # We need historical stats for the *current* race's going_code.
        # But going_code varies by race.
        # Strategy:
        # a. Calculate cumulative stats for EACH going_code for each horse.
        #    Columns: horse_going_1_runs, horse_going_1_wins, ... (1=Ryo, 2=Yaya, 3=Omo, 4=Furyo)
        # b. Mapping: Select the column matching the current race's going_code.
        
        # Map rank to win/top3 flags
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
        df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)
        
        # One-Hot encoding for calculation
        # going_code: 1~4
        for code in [1, 2, 3, 4]:
            df_sorted[f'is_going_{code}'] = (df_sorted['going_code'] == code).astype(int)
            df_sorted[f'win_going_{code}'] = df_sorted[f'is_going_{code}'] * df_sorted['is_win']
            df_sorted[f'top3_going_{code}'] = df_sorted[f'is_going_{code}'] * df_sorted['is_top3']

        # GroupBy Horse and Shift (Past records only)
        # rolling / expanding sum
        # Ideally using incremental aggregator logic for speed, but here using pandas groupby for simplicity in batch 1.
        # Note: Large dataset might be slow.
        
        grp = df_sorted.groupby('horse_id')
        
        start_cols = []
        for code in [1, 2, 3, 4]:
            df_sorted[f'cum_runs_{code}'] = grp[f'is_going_{code}'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
            df_sorted[f'cum_wins_{code}'] = grp[f'win_going_{code}'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
            df_sorted[f'cum_top3_{code}'] = grp[f'top3_going_{code}'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
            start_cols.extend([f'cum_runs_{code}', f'cum_wins_{code}', f'cum_top3_{code}'])

        # 3. Retrieve feature for Current Race
        # Identify current going_code for each row
        # Vectorized lookup using numpy
        
        # Create result arrays
        n_runs = np.zeros(len(df_sorted))
        n_wins = np.zeros(len(df_sorted))
        n_top3 = np.zeros(len(df_sorted))
        
        # Fill based on going_code
        # This loop is small (4 iterations)
        current_codes = df_sorted['going_code'].values
        
        for code in [1, 2, 3, 4]:
            mask = (current_codes == code)
            n_runs[mask] = df_sorted.loc[mask, f'cum_runs_{code}']
            n_wins[mask] = df_sorted.loc[mask, f'cum_wins_{code}']
            n_top3[mask] = df_sorted.loc[mask, f'cum_top3_{code}']

        # 4. Calculate Rates
        # Avoid division by zero
        # Default average: If 0 runs, maybe use overall average or 0? 0 is safe for tree models.
        
        eps = 1e-6
        # To avoid confusion with "0 wins / 0 runs" vs "0 wins / 10 runs", 
        # usually we add prior or just leave as 0. 
        # Here: 0 if no runs.
        
        horse_going_win_rate = np.where(n_runs > 0, n_wins / (n_runs + eps), 0.0)
        horse_going_top3_rate = np.where(n_runs > 0, n_top3 / (n_runs + eps), 0.0)
        
        df_sorted['horse_going_count'] = n_runs
        df_sorted['horse_going_win_rate'] = horse_going_win_rate
        df_sorted['horse_going_top3_rate'] = horse_going_top3_rate
        
        # 5. Mudder Flag (Special logic for heavy tracks)
        # If current track IS heavy/bad (3 or 4), AND horse has good record on 3/4.
        # Or just "proven mudder" as a static attribute regardless of current track?
        # User request: "Horse Track Aptitude". 
        # Let's provide "proven_mudder_rate" (Win rate on 3+4) as a separate feature, 
        # which is useful if current track is Ryo but might turn bad, or just general toughness.
        
        df_sorted['mud_runs'] = df_sorted['cum_runs_3'] + df_sorted['cum_runs_4']
        df_sorted['mud_wins'] = df_sorted['cum_wins_3'] + df_sorted['cum_wins_4']
        df_sorted['mud_top3'] = df_sorted['cum_top3_3'] + df_sorted['cum_top3_4']
        
        df_sorted['is_proven_mudder'] = np.where(
            (df_sorted['mud_runs'] >= 2) & ((df_sorted['mud_top3'] / (df_sorted['mud_runs'] + eps)) >= 0.33), 
            1, 0
        )

        # Going shift adaptability
        df_sorted['prev_going_code'] = grp['going_code'].shift(1).fillna(0).astype(int)
        df_sorted['going_shift'] = (df_sorted['going_code'] - df_sorted['prev_going_code']).astype(int)
        df_sorted['is_shift_up'] = (df_sorted['going_shift'] >= 1).astype(int)
        df_sorted['is_shift_down'] = (df_sorted['going_shift'] <= -1).astype(int)
        df_sorted['is_top3_shift_up_obs'] = np.where(df_sorted['is_shift_up'] == 1, df_sorted['is_top3'], np.nan)
        df_sorted['is_top3_shift_down_obs'] = np.where(df_sorted['is_shift_down'] == 1, df_sorted['is_top3'], np.nan)
        df_sorted['going_shift_up_top3_rate'] = grp['is_top3_shift_up_obs'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.0)
        df_sorted['going_shift_down_top3_rate'] = grp['is_top3_shift_down_obs'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.0)
        
        out_cols = [
            'race_id', 'horse_number', 'horse_id',
            'horse_going_count', 'horse_going_win_rate', 'horse_going_top3_rate',
            'is_proven_mudder', 'going_shift', 'going_shift_up_top3_rate', 'going_shift_down_top3_rate'
        ]
        
        # Align index back to original if needed, but Pipeline usually expects dataframe with matching index or sort handling.
        # FeatureGenerator usually returns dataframe matching input length/index.
        # Sort back to index
        df_sorted = df_sorted.loc[df.index]
        
        return df_sorted[out_cols]
