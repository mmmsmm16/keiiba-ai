
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_jockey_trainer_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Jockey Trainer Compatibility (黄金コンビ・相性)
    - 既存の `jockey_trainer_stats` (単純な累積勝率) とは異なり、
      「騎手本来の実力に対して、その調教師と組んだ時にどれだけパフォーマンスが上がるか？」
      という【相性 (Compatibility/Synergy)】を特徴量化する。
    
    Features:
    - jt_win_diff: (Combo Win Rate) - (Jockey Global Win Rate)
    - jt_top3_diff: (Combo Top3 Rate) - (Jockey Global Top3 Rate)
    - is_first_combo: 初結成フラグ
    - jt_log_count: 結成回数の対数 (信頼度)
    
    Logic:
    - Use expanding window for leakage prevention.
    - Global Rate is also expanding (average up to that point).
    """
    logger.info("ブロック計算中: compute_jockey_trainer_compatibility")
    
    req_cols = ['race_id', 'horse_number', 'horse_id', 'date', 'jockey_id', 'trainer_id', 'is_win', 'is_top3']
    # Check targets (might be missing in raw load if not careful, but pipeline ensures targets usually)
    # Actually raw df might not have is_win/is_top3 if not added by pipeline common logic.
    
    df_sorted = df.sort_values(['date', 'race_id']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    # Ensure targets
    if 'is_win' not in df_sorted.columns:
        if 'rank' in df_sorted.columns:
            df_sorted['rank_num'] = pd.to_numeric(df_sorted['rank'], errors='coerce')
            df_sorted['is_win'] = (df_sorted['rank_num'] == 1).astype(int)
            df_sorted['is_top3'] = (df_sorted['rank_num'] <= 3).astype(int)
        else:
            return df[['race_id', 'horse_number', 'horse_id']].copy()

    # 1. Calc Combo Stats (Expanded)
    # Group: [Jockey, Trainer]
    def calc_stat(d, keys, tgt):
        return d.groupby(keys)[tgt].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
    
    df_sorted['combo_win'] = calc_stat(df_sorted, ['jockey_id', 'trainer_id'], 'is_win')
    df_sorted['combo_top3'] = calc_stat(df_sorted, ['jockey_id', 'trainer_id'], 'is_top3')
    
    # Run Count
    df_sorted['combo_count'] = df_sorted.groupby(['jockey_id', 'trainer_id']).cumcount()
    df_sorted['jt_log_count'] = np.log1p(df_sorted['combo_count'])
    
    df_sorted['is_first_combo'] = (df_sorted['combo_count'] == 0).astype(int)
    
    # 2. Calc Jockey Global Stats (Expanded)
    # Group: [Jockey]
    # Note: Using transform on large df with high cardinality 'jockey_id' can be slow but okay for 1M rows.
    df_sorted['jockey_win'] = calc_stat(df_sorted, ['jockey_id'], 'is_win')
    df_sorted['jockey_top3'] = calc_stat(df_sorted, ['jockey_id'], 'is_top3')
    
    # 3. Compatibility Diff
    # If combo count is low, diff is unreliable.
    # We leave it as raw diff, model can handle it via count feature.
    # Or we can shrink towards 0? Raw diff is fine.
    
    df_sorted['jt_win_diff'] = df_sorted['combo_win'] - df_sorted['jockey_win']
    df_sorted['jt_top3_diff'] = df_sorted['combo_top3'] - df_sorted['jockey_top3']
    
    # Return
    feats = [
        'jt_win_diff', 'jt_top3_diff',
        'is_first_combo', 'jt_log_count',
        # Optional: Include raw combo stats if we think existing block is insufficient?
        # Let's keep minimal set to test "Compatibility" hypothesis.
        # Maybe Trainer Global stats too? 
        # Trainer might filter Jockeys.
    ]
    keys = ['race_id', 'horse_number', 'horse_id']
    
    return df_sorted[keys + feats].copy()
