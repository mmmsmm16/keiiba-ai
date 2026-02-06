
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_jockey_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Jockey Strategy (乗り替わり戦略)
    - 騎手の変更パターンから陣営の意図（勝負気配、試走、減量狙いなど）を読み解く。
    
    Features:
    - jockey_win_rate: 当該騎手の（その時点までの）通算勝率。
    - jockey_rank_diff: 今回騎手と前回騎手の勝率差。(今回 - 前回)。プラスなら「鞍上強化」。
    - is_jockey_change: 乗り替わり発生フラグ。
    - is_jockey_return: 前回は違ったが、前々回以前に乗っていた騎手に戻った（手戻り）。
    - is_top_jockey_switch: 勝率15%以上のトップジョッキーへの乗り替わり。
    """
    logger.info("ブロック計算中: compute_jockey_strategy")
    
    keys = ['race_id', 'horse_number', 'horse_id', 'date']
    
    # Needs jockey_id, is_win (to calc rate), rank?
    if 'jockey_id' not in df.columns:
        return df[keys].copy() # Should handle if keys missing? Assuming basic keys exist
        
    # Sort for expanding calcs
    # 1. Calc Global Jockey Win Rate (Expanding)
    # We Sort by date globally to simulate "Reading Table"
    df_sorted = df.sort_values(['date', 'race_id']).copy()
    
    # Target encoding for Jockey Limit Leakage
    # We calculate expanding mean of 'is_win' grouped by jockey_id.
    # Shift(1) is essential.
    
    if 'rank' in df_sorted.columns:
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(float)
    else:
        # Cannot calc rate if target missing. fallback
        df_sorted['is_win'] = np.nan
        
    # Jockey Rate
    # expanding mean
    df_sorted['jockey_win_rate'] = df_sorted.groupby('jockey_id')['is_win'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df_sorted['jockey_win_rate'] = df_sorted['jockey_win_rate'].fillna(0.05) # Impute avg
    
    # Now sort by Horse to check rotation
    df_horse = df_sorted.sort_values(['horse_id', 'date']).copy()
    
    # Prev Jockey info
    df_horse['prev_jockey_id'] = df_horse.groupby('horse_id')['jockey_id'].shift(1)
    df_horse['prev_jockey_win_rate'] = df_horse.groupby('horse_id')['jockey_win_rate'].shift(1)
    
    # Features
    df_horse['is_jockey_change'] = (df_horse['jockey_id'] != df_horse['prev_jockey_id']) & df_horse['prev_jockey_id'].notnull()
    df_horse['is_jockey_change'] = df_horse['is_jockey_change'].astype(int)
    
    # Rank Diff (Upgrade/Downgrade)
    # If no prev (first run), diff is 0
    df_horse['jockey_rate_diff'] = df_horse['jockey_win_rate'] - df_horse['prev_jockey_win_rate']
    df_horse['jockey_rate_diff'] = df_horse['jockey_rate_diff'].fillna(0)
    
    # Is Return
    # Check lag2
    df_horse['lag2_jockey_id'] = df_horse.groupby('horse_id')['jockey_id'].shift(2)
    # Return = (Change happened) AND (Current == Lag2)
    df_horse['is_jockey_return'] = (df_horse['is_jockey_change'] == 1) & (df_horse['jockey_id'] == df_horse['lag2_jockey_id'])
    df_horse['is_jockey_return'] = df_horse['is_jockey_return'].astype(int)
    
    # Top Jockey Switch
    # Definition of Top: Rate > 0.15 (15%)
    TOP_THRESHOLD = 0.15
    df_horse['is_top_jockey'] = (df_horse['jockey_win_rate'] >= TOP_THRESHOLD).astype(int)
    df_horse['is_top_jockey_switch'] = (df_horse['is_jockey_change'] == 1) & (df_horse['is_top_jockey'] == 1)
    df_horse['is_top_jockey_switch'] = df_horse['is_top_jockey_switch'].astype(int)
    
    # Return cols
    feats = [
        'jockey_win_rate',
        'jockey_rate_diff',
        'is_jockey_change',
        'is_jockey_return',
        'is_top_jockey_switch'
    ]
    
    return df_horse[keys + feats].copy()
