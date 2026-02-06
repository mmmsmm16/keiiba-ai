import pandas as pd
import numpy as np
from scipy.special import logit
from .plackett_luce import sample_rankings, estimate_p_win, estimate_p_place, estimate_p_umaren, estimate_p_wakuren

def compute_ticket_probs(race_df, strength_col='pred_prob', n_samples=20000, seed=42):
    """
    Compute ticket probabilities for a single race.
    
    Args:
        race_df (pd.DataFrame): Must contain 'horse_number', 'frame_number', and 'strength_col'.
                                If 'strength_col' is probability (0-1), it will be logit-transformed.
                                If it is already logit/score, specify is_logit=True? 
                                Actually, strict PL assumes exp(score) proportional to prob.
                                So if we have p_win, log(p_win) is the score (roughly).
                                Because P(i) = exp(s_i) / sum(exp(s_j)).
                                If s_i = log(p_i), then exp(s_i)=p_i. sum(p_i)=1. Correct.
        strength_col (str): Column name for win probability or score.
        n_samples (int): MC samples.
        seed (int): Seed.
        
    Returns:
        dict: {
            'win': pd.Series(index=horse_number, data=prob),
            'place': pd.Series(index=horse_number, data=prob),
            'umaren': pd.DataFrame(index=horse1, columns=horse2, data=prob),
            'wakuren': pd.DataFrame(index=frame1, columns=frame2, data=prob)
        }
    """
    # Validation
    required = ['horse_number', 'frame_number']
    if not all(c in race_df.columns for c in required):
        raise ValueError(f"Missing columns: {required}")
        
    df = race_df.copy().sort_values('horse_number')
    horses = df['horse_number'].values
    frames = df['frame_number'].values
    
    # Prepare Strengths
    # If input is probability, convert to logit (essentially log(p) if normalized, but logit(p) maps 0-1 to -inf/inf)
    # Using log(p) is safer if sum(p)=1.
    # If using model output `pred_prob` (which might not sum to 1 perfectly due to calibration independent), 
    # we normalize first?
    vals = df[strength_col].values.astype(float)
    
    # If vals are probabilities (0-1), normalize and log.
    if vals.min() >= 0 and vals.max() <= 1.0:
        vals = vals / (vals.sum() + 1e-9)
        strengths = np.log(vals + 1e-10)
    else:
        # Assume already log scale (logits)
        strengths = vals
        
    # Sample
    # rankings is (n_samples, n_horses) of INDICES (0..N-1) relative to df rows
    rankings = sample_rankings(strengths, n_samples=n_samples, seed=seed)
    n_horses = len(horses)
    
    # Win
    p_win_arr = estimate_p_win(rankings, n_horses)
    s_win = pd.Series(p_win_arr, index=horses, name='p_win')
    
    # Place (Top 3 usually. JRA rule: 7 or fewer horses -> Top 2. 8+ -> Top 3)
    n_starters = len(horses)
    n_places = 2 if n_starters <= 7 else 3
    p_place_arr = estimate_p_place(rankings, n_horses, n_places=n_places)
    s_place = pd.Series(p_place_arr, index=horses, name='p_place')
    
    # Umaren
    p_umaren_mat = estimate_p_umaren(rankings, n_horses)
    df_umaren = pd.DataFrame(p_umaren_mat, index=horses, columns=horses)
    
    # Wakuren
    # horse_to_frame map: array where index is row index, value is frame
    # frames is aligned with df rows (because we didn't reorder df other than sort by horse_num)
    p_wakuren_mat = estimate_p_wakuren(rankings, frames)
    # create index 1..8
    # output matrix size is 9x9 (0-8), we want 1-8
    valid_frames = list(range(1, 9))
    df_wakuren = pd.DataFrame(p_wakuren_mat[1:9, 1:9], index=valid_frames, columns=valid_frames)
    
    return {
        'win': s_win,
        'place': s_place,
        'umaren': df_umaren,
        'wakuren': df_wakuren
    }
