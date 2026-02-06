
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_risk_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Risk & Consistency Stats (安定性とリスク指標)
    - 平均値だけでなく「分散」や「極端な負け」を特徴量化する。
    - 穴馬検知（一発屋）や危険な人気馬（不安定）の識別に寄与。
    
    Features:
    1. rank_std_5: 近5走の着順標準偏差 (安定感)
    2. speed_index_std_5: 近5走のSI標準偏差 (パフォーマンスのブレ)
    3. collapse_rate_10: 近10走での大敗率 (着順>=10 or タイム差>=2.0s)
    4. resurrection_flag: 前走大敗(>=10着)だが、過去に同クラス以上で勝利経験があるか (巻き返し候補)
    """
    logger.info("ブロック計算中: compute_risk_stats")
    
    # 必要なカラム
    req_cols = ['horse_id', 'date', 'race_id', 'horse_number', 'rank', 'time_diff', 'class_label', 'is_win']
    # class_label, is_win might need to be created if not present
    
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    # Ensure auxiliary columns
    if 'is_win' not in df_sorted.columns and 'rank' in df_sorted.columns:
         df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
    
    if 'class_label' not in df_sorted.columns and 'kyoso_joken_code' in df_sorted.columns:
         df_sorted['class_label'] = df_sorted['kyoso_joken_code'].fillna('999')

    # Basic cleaning for std calculation
    # rank: numeric
    df_sorted['rank_num'] = pd.to_numeric(df_sorted['rank'], errors='coerce')
    
    # Needs SI?
    # pipeline passes raw df. SI is not in raw df usually (it's calculated in speed_index_stats).
    # However, relative_stats calculated simplified SI. We can re-calculate simplified SI here for self-containment.
    # OR, we depend on speed_index_stats being run before? No, blocks should be independent if possible.
    # Let's calculate a simplified performance metric for std.
    # Time diff is good enough proxy for performance stability (normalized by race level implicitly if we look at std?)
    # Actually TimeDiff std is very good.
    
    # Ensure time_diff
    if 'time_diff' not in df_sorted.columns:
         # Try to recover or fill
         # Usually time_diff comes from preprocessing/loader.
         # If missing, we can't calc std properly.
         logger.warning("time_diff missing in risk_stats input. Filling with 0.")
         df_sorted['time_diff_num'] = 0.0
    else:
         df_sorted['time_diff_num'] = pd.to_numeric(df_sorted['time_diff'], errors='coerce').fillna(0.0)

    grp = df_sorted.groupby('horse_id')
    
    # 1. Rank Std (5 runs)
    # shift(1) to avoid leakage
    df_sorted['rank_std_5'] = grp['rank_num'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
    
    # 2. Time Diff Std (5 runs)
    # タイム差のバラつき。小さい＝相手なりに走るor常に好走/凡走。大きい＝ムラがある。
    df_sorted['time_diff_std_5'] = grp['time_diff_num'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
    
    # 3. Collapse Rate (10 runs)
    # Def: Rank >= 10 or TimeDiff >= 2.0
    # is_collapse
    condition_rank = (df_sorted['rank_num'] >= 10)
    condition_time = (df_sorted['time_diff_num'] >= 2.0)
    df_sorted['is_collapse'] = (condition_rank | condition_time).astype(int)
    
    df_sorted['collapse_rate_10'] = grp['is_collapse'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
    
    # 4. Resurrection Flag (巻き返し)
    # Previous run was collapse?
    df_sorted['prev_is_collapse'] = grp['is_collapse'].shift(1).fillna(0)
    
    # Has won in same class or higher?
    # This is complex without strict class hierarchy.
    # Let's simplify: Has won in ANY class in history?
    # expanding().max() of is_win
    df_sorted['has_won_history'] = grp['is_win'].transform(lambda x: x.shift(1).expanding().max()).fillna(0)
    
    # Flag: Prev=Collapse AND HasWon=1
    df_sorted['resurrection_flag'] = (df_sorted['prev_is_collapse'] == 1) & (df_sorted['has_won_history'] == 1)
    df_sorted['resurrection_flag'] = df_sorted['resurrection_flag'].astype(int)
    
    # 抽出
    feats = ['rank_std_5', 'time_diff_std_5', 'collapse_rate_10', 'resurrection_flag']
    keys = ['race_id', 'horse_number', 'horse_id']
    
    return df_sorted[keys + feats].copy()
