
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_runstyle_fit(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] RunStyle Fit (脚質・展開適合性)
    - 脚質 (Running Style) と コース条件・枠番・展開予測 の相互作用を特徴量化する。
    - 単体の能力値ではなく、「条件が向くか」をスコア化する。
    
    Assumption:
    - 'running_style' column exists (1:Nige, 2:Senko, 3:Sashi, 4:Oikomi).
    - If not, utilize 'first_corner_rank' from previous race (shift) to proxy.
    - Here we assume we have a 'prev_running_style' or similar available.
    - Actually, raw data usually has 'running_style' for the *current* race results (Leakage!).
    - WE MUST USE PREVIOUS RUNNING STYLE.
    
    Logic:
    1. Determine 'RunStyle' for this race:
       - Use 'prev_running_style' (from previous race).
       - Or Aggregated running style from history (Most Common).
       - For simplicity and safety, we calculate "Primary RunStyle" from past 1-3 races.
    
    Features:
    - fit_nige_short: Nige x Short Straight
    - fit_sashi_long: Sashi x Long Straight
    - fit_inner_nige: Inner Frame (1-3) x Nige
    - fit_outer_sashi: Outer Frame (6-8) x Sashi (Less blockage?) - optional
    - pace_predicted: Based on member composition (Number of Nige horses)
    - fit_pace: Nige x Slow Pace Prediction, Sashi x Fast Pace Prediction
    """
    logger.info("ブロック計算中: compute_runstyle_fit")
    
    # Required Cols
    # We need history of running_style. 
    # 'running_style' in raw df is the result of THIS race. -> LEAKAGE.
    # We must calculate strictly from past.
    
    # Check if we have history stats block merged? No, we perform calculation here.
    # We need 'running_style' and date/horse_id.
    
    req_cols = ['horse_id', 'date', 'race_id', 'venue', 'horse_number']
    
    # 0. Infer Running Style if missing
    if 'running_style' not in df.columns:
        if 'pass_1' in df.columns:
            logger.info("Column 'running_style' missing. Inferring from 'pass_1'.")
            def infer_style(p):
                # [Fix] '00' means no corner data - treat as unknown, not position 0
                if pd.isna(p) or p == '' or p == '00' or p == 0:
                    return np.nan
                try:
                    p_val = float(p)
                    if p_val <= 1.0: return 1 # Nige (Lead)
                    if p_val <= 4.0: return 2 # Senko
                    if p_val <= 10.0: return 3 # Sashi
                    return 4 # Oikomi
                except:
                    return np.nan # Unknown
            df['running_style'] = df['pass_1'].apply(infer_style)
            # Fill NaN with 2 (Senko) - default for unknown
            df['running_style'] = df['running_style'].fillna(2).astype(int)
        else:
            # Cannot calculate
            logger.warning("Column 'running_style' and 'pass_1' not found. Using dummy.")
            return df[['race_id', 'horse_number', 'horse_id']].copy()

    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
    # 1. Determine "Predicted RunStyle" (Mode of all past available races)
    # 1:Nige, 2:Senko, 3:Sashi, 4:Oikomi
    # Use expanding mode to avoid leakage and handle missing data
    def get_past_mode(x):
        # x is a series of runstyles
        # We need to return a series where each entry is the mode of all elements ABOVE it
        res = []
        history = []
        for val in x:
            if history:
                # Mode of history
                m = pd.Series(history).mode()
                res.append(m[0] if not m.empty else 2)
            else:
                res.append(2)
            # Add current to history for next
            if not pd.isna(val):
                history.append(val)
        return pd.Series(res, index=x.index)

    df_sorted['pred_runstyle'] = df_sorted.groupby('horse_id')['running_style'].transform(get_past_mode)
    df_sorted['pred_runstyle'] = df_sorted['pred_runstyle'].fillna(2).astype(int)
    
    # 2. Course Attributes (Reuse logic)
    # [Fix] Use numeric keibajo codes because loader loads 'keibajo_code AS venue'
    # '04': Niigata, '05': Tokyo, '07': Chukyo, '08': Kyoto, '09': Hanshin
    major_venues = ['04', '05', '07', '08', '09']
    
    def is_long_straight(v):
        v_str = str(v).zfill(2)
        return 1 if v_str in major_venues else 0
        
    df_sorted['is_long'] = df_sorted['venue'].apply(is_long_straight)
    df_sorted['is_short'] = 1 - df_sorted['is_long']
    
    # 3. Frame Attributes
    # horse_number is not bracket number (wakuban). 
    # Usually Inner: 1-4, Outer: 10+.
    # Let's use horse_number directly as approximation if frame not avail.
    # If 'frame_number' (wakuban) is available?
    if 'frame_number' in df_sorted.columns:
        df_sorted['is_inner'] = (df_sorted['frame_number'] <= 3).astype(int)
        df_sorted['is_outer'] = (df_sorted['frame_number'] >= 6).astype(int)
    else:
        df_sorted['is_inner'] = (df_sorted['horse_number'] <= 4).astype(int)
        df_sorted['is_outer'] = (df_sorted['horse_number'] >= 10).astype(int)
        
    # 4. Generate Fits
    
    # 4.1 Nige x Short / Inner
    # Nige = 1
    is_nige = (df_sorted['pred_runstyle'] == 1).astype(int)
    is_sashi = (df_sorted['pred_runstyle'] >= 3).astype(int)
    
    df_sorted['fit_nige_short'] = is_nige * df_sorted['is_short']
    df_sorted['fit_inner_nige'] = is_nige * df_sorted['is_inner']
    
    # 4.2 Sashi x Long
    df_sorted['fit_sashi_long'] = is_sashi * df_sorted['is_long']
    
    # 5. Pace Prediction (Member Interaction)
    # Calculate how many "Nige" horses are in the race.
    # Shifted runstyle is already calculated as 'pred_runstyle'.
    # We can group by race_id.
    
    # Count Nige per race
    # transform('sum') of is_nige
    df_sorted['n_nige_in_race'] = df_sorted.groupby('race_id')['pred_runstyle'].transform(lambda x: (x == 1).sum())
    
    # Bias: The horse itself is included.
    # "Other Nige" = Total Nige - (1 if Self is Nige else 0)
    df_sorted['n_other_nige'] = df_sorted['n_nige_in_race'] - is_nige
    
    # Predicted Pace
    # Many Nige (e.g. >= 2 others) -> High Pace likely -> Good for Sashi, Bad for Nige
    # No Nige (e.g. 0 others) -> Slow Pace likely -> Good for Nige/Senko
    
    df_sorted['pred_pace_high'] = (df_sorted['n_other_nige'] >= 2).astype(int)
    df_sorted['pred_pace_slow'] = (df_sorted['n_other_nige'] == 0).astype(int)
    
    # 5.1 Pace Fits
    # Nige x Slow (Easy Lead)
    df_sorted['fit_nige_slow'] = is_nige * df_sorted['pred_pace_slow']
    
    # Sashi x High (Pace Collapses)
    df_sorted['fit_sashi_high'] = is_sashi * df_sorted['pred_pace_high']
    
    # 6. Return
    feats = [
        'pred_runstyle', # Useful feature itself
        'fit_nige_short', 'fit_inner_nige',
        'fit_sashi_long',
        'n_nige_in_race', 'n_other_nige',
        'fit_nige_slow', 'fit_sashi_high'
    ]
    keys = ['race_id', 'horse_number', 'horse_id']
    
    return df_sorted[keys + feats].copy()
