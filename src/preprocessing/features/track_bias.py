
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_track_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Track Bias (Pace/Inside-Outside) and Adversity Score (Performance against bias).
    
    Args:
        df: DataFrame containing race results (race_id, horse_id, rank, pass_rank, waku_no, date).
            Must contain historical data to calculate past bias.
            
    Returns:
        pd.DataFrame: Features [race_id, horse_number, bias_adversity_score_mean_5]
    """
    # Work on copy first
    df_work = df.copy()
    
    # Alias columns if using standard loader format
    if 'pass_rank' not in df_work.columns and 'passing_rank' in df_work.columns:
        df_work['pass_rank'] = df_work['passing_rank']
    if 'waku_no' not in df_work.columns and 'frame_number' in df_work.columns:
        df_work['waku_no'] = df_work['frame_number']
        
    # Check required columns
    required_cols = ['race_id', 'horse_id', 'horse_number', 'date', 'rank', 'pass_rank', 'waku_no']
    missing = [c for c in required_cols if c not in df_work.columns]
    if missing:
        logger.warning(f"Missing track_bias required cols: {missing}. Available: {df_work.columns.tolist()[:15]}")
        return pd.DataFrame()

    # Pre-process pass_rank (take first corner position if strictly needed, or mean of corners)
    # Assuming 'pass_rank' is available (JRA-VAN typically has '1-1-1-1' format str).
    # If it is string, we need to parse. If it's already int (preprocessed), good.
    # Check dtype.
    if df_work['pass_rank'].dtype == object:
        # Simple parse: take the first number (Start Position) or Mean?
        # Usually '1-1-1-1' -> Escape. '10-10-10-10' -> Chase.
        # Let's take the first value as "Positioning Strategy"
        def parse_first_pass(s):
            try:
                if pd.isna(s): return np.nan
                s_str = str(s).replace('.0', '') # standard loader might leave .0
                parts = s_str.split('-')
                if not parts: return np.nan
                return float(parts[0])
            except:
                return np.nan
        df_work['pos_style'] = df_work['pass_rank'].apply(parse_first_pass)
    else:
        df_work['pos_style'] = df_work['pass_rank']

    # 1. Calculate Race Bias (The "Flow" of the race)
    # Bias is defined by the average position of the TOP 5 horses.
    # If Top 5 were all Front (pos < 5), it's a Front Bias race.
    
    # Filter Top 5 (Using rank <= 5)
    top5 = df_work[df_work['rank'].isin([1, 2, 3, 4, 5])].copy()
    
    if top5.empty:
        return pd.DataFrame()

    race_bias = top5.groupby('race_id').agg({
        'pos_style': 'mean', # Low = Front Bias, High = Rear Bias
        'waku_no': 'mean'    # Low = Inner Bias, High = Outer Bias
    }).rename(columns={'pos_style': 'race_pos_bias', 'waku_no': 'race_waku_bias'})
    
    # Merge Bias back to all horses (to evaluate their performance against it)
    df_work = pd.merge(df_work, race_bias, on='race_id', how='left')
    
    # 2. Calculate Adversity Score per race
    # Logic: Performance * Mismatch
    # Performance metric: Reciprocal of rank (1/1=1.0, 1/2=0.5, ... 1/10=0.1)
    # But we want to highlight "Good Performance in Bad Match".
    # Mismatch: abs(MyPos - RacePosBias)
    # If I was Front(1) and RaceBias was Rear(10) -> Mismatch 9.
    # If I won (Rank 1) -> Score = 1.0 * 9 = 9.0 (Huge Hero)
    # If I was Front(1) and RaceBias was Front(1) -> Mismatch 0.
    # If I won (Rank 1) -> Score = 1.0 * 0 = 0.0 (Easy Game) -> Wait, 0 is too low?
    # Maybe (1 + Mismatch)?
    
    df_work['perf_score'] = 1.0 / (df_work['rank'] + 1.0) # 0.5 max for win? No, Rank1->0.5. Let's use 3.0/(Rank+1)?
    # Or just standard 1/Rank. Rank 1->1.0. 
    # Let's fix 1/Rank.
    # Handle Rank 0 or NaN? Rank in JRA is 1-18. 
    # Use clip to be safe.
    df_work['perf_score'] = 1.0 / df_work['rank'].clip(lower=1)
    
    # Mismatch
    df_work['pos_mismatch'] = (df_work['pos_style'] - df_work['race_pos_bias']).abs()
    df_work['waku_mismatch'] = (df_work['waku_no'] - df_work['race_waku_bias']).abs()
    
    # Adversity Score
    # We combine both mismatches? Or focus on Position (Pace) as user requested "Ma-ba state/Zen-kori".
    # User mentioned "Front remaining ma-ba but came from behind".
    # Case: RaceBias=Front(2.0). I was Rear(10.0). Mismatch=8.0. I got Rank 5. 
    # Score = (1/5) * (1 + 8) = 0.2 * 9 = 1.8.
    # Case: RaceBias=Front(2.0). I was Front(2.0). Mismatch=0.0. I got Rank 5.
    # Score = 0.2 * 1 = 0.2.
    # Seems reasonable.
    
    df_work['adversity_score'] = df_work['perf_score'] * (1.0 + df_work['pos_mismatch'] + 0.5 * df_work['waku_mismatch'])
    
    # 3. Aggregate Past Scores (Lag Features)
    # We need to sort by date and calc rolling mean for each horse.
    df_work = df_work.sort_values('date')
    
    # Group by horse_id
    # We want "Average Adversity Score over last 5 races"
    # shift(1) is crucial to avoid leakage (current race result shouldn't be used for current prediction)
    
    def calc_rolling(series):
        return series.shift(1).rolling(window=5, min_periods=1).mean()
        
    df_work['bias_adversity_score_mean_5'] = df_work.groupby('horse_id')['adversity_score'].transform(calc_rolling)
    
    # Fill NA with 0 (New horses have no adversity history)
    df_work['bias_adversity_score_mean_5'] = df_work['bias_adversity_score_mean_5'].fillna(0)
    
    return df_work[['race_id', 'horse_number', 'horse_id', 'bias_adversity_score_mean_5']]
