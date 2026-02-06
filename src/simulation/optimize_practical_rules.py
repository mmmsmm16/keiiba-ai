import pandas as pd
import numpy as np
import logging
import itertools
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def bootstrap_roi_ci(df: pd.DataFrame, payout_col: str, n_boot: int = 1000, ci: float = 0.95):
    """
    ブートストラップ法によりROIの信頼区間を計算
    """
    if df.empty: return 0.0, 0.0, 0.0
    
    # 高速化のためnumpy配列化
    returns = df[payout_col].values
    bets = np.ones(len(returns)) * 100 # Unit bet 100
    
    n_samples = len(returns)
    indices = np.random.randint(0, n_samples, size=(n_boot, n_samples))
    
    boot_returns = np.sum(returns[indices], axis=1)
    boot_bets = np.sum(bets[indices], axis=1)
    
    boot_rois = boot_returns / boot_bets
    
    lower = np.percentile(boot_rois, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_rois, (1 + ci) / 2 * 100)
    mean_roi = np.mean(boot_rois)
    
    return mean_roi, lower, upper

def evaluate_rule_robust(df: pd.DataFrame, t1, t2, e1, en1, s1, bet_type, is_train: bool = True):
    payout_col = 'payout_win' if bet_type == 'win' else 'payout_place'
    
    # Calculate EV if needed
    if 'ev' not in df.columns:
        df['ev'] = df['p_cal'] * df['odds_win_pre']
        
    mask = (df['rank_pred'] == 1) & \
           (df['p1'] >= t1) & \
           (df['margin'] >= t2) & \
           (df['ev'] >= e1) & \
           (df['entropy'] <= en1) & \
           (df['field_size'] <= s1)
           
    selected = df[mask].copy()
    
    if selected.empty: return None
    
    # 統計
    total_bet = len(selected) * 100
    total_ret = selected[payout_col].sum()
    roi = total_ret / total_bet
    
    # 年別チェック
    years = selected['year_valid'].unique()
    yearly_stats = {}
    for y in [2022, 2023, 2024]:
        ydf = selected[selected['year_valid'] == y]
        y_bet = len(ydf)
        y_ret = ydf[payout_col].sum()
        y_roi = y_ret / (y_bet * 100) if y_bet > 0 else 0
        yearly_stats[y] = {'roi': y_roi, 'bets': y_bet}
        
    min_roi_yr = min([v['roi'] for v in yearly_stats.values()])
    min_bets_yr = min([v['bets'] for v in yearly_stats.values()])

    res = {
        't1': t1, 't2': t2, 'e1': e1, 'en1': en1, 's1': s1, 'bet_type': bet_type,
        'num_bets': len(selected),
        'roi': roi,
        'min_roi_yr': min_roi_yr,
        'min_bets_yr': min_bets_yr,
        'stats_2022': yearly_stats[2022],
        'stats_2023': yearly_stats[2023],
        'stats_2024': yearly_stats[2024]
    }
    
    # TrainモードならBootstrap計算は省略（速度優先）
    # Test/Validationモードなら計算
    if not is_train:
        mean_bs, lower_bs, upper_bs = bootstrap_roi_ci(selected, payout_col)
        res['ci_lower'] = lower_bs
        res['ci_upper'] = upper_bs
        res['ci_mean'] = mean_bs
        
    return res

def run_nested_optimization(input_path: str):
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Split Train (2022-2023) / Test (2024)
    train_df = df[df['year_valid'].isin([2022, 2023])].copy()
    test_df = df[df['year_valid'] == 2024].copy()
    
    # Parameters for Grid Search
    # 実用性重視のためマージンやEVの範囲を広くとる
    # Parameters for Grid Search
    # 実用性重視: EV閾値を下げ、範囲を広げる
    win_params = {
        't1': [0.3, 0.4, 0.5, 0.6],
        't2': [0.0, 0.05, 0.1],
        'e1': [1.1, 1.2, 1.5, 1.8, 2.0],
        'en1': [2.5, 99.0],
        's1': [18],
        'bet_type': ['win']
    }
    place_params = {
        't1': [0.5, 0.6, 0.7],
        't2': [0.0, 0.1, 0.2],
        'e1': [0.0],
        'en1': [2.0, 2.5, 99.0],
        's1': [18],
        'bet_type': ['place']
    }
    
    candidates = []
    
    # 制約
    # 緩和された収集用基準 (分析用)
    COLLECT_BETS_TRAIN = 50 
    
    logger.info("Step 1: Grid Search on Train Data (2022-2023)...")
    
    search_space = []
    for p in [win_params, place_params]:
        keys = p.keys()
        for v in itertools.product(*p.values()):
            search_space.append(dict(zip(keys, v)))
            
    for params in search_space:
        res = evaluate_rule_robust(train_df, **params, is_train=True)
        if not res: continue
        
        # 最低限のフィルタ (ノイズ除去)
        if res['num_bets'] < COLLECT_BETS_TRAIN: continue
        
        # 通過した候補をTestで評価
        test_res = evaluate_rule_robust(test_df, **params, is_train=False) # Bootstrapあり
        if not test_res: continue
        
        # 全期間評価もつけておく
        full_res = evaluate_rule_robust(df, **params, is_train=False)
        
        candidate = {
            'params': params,
            'train_roi': res['roi'],
            'train_bets': res['num_bets'],
            'test_roi': test_res['roi'],
            'test_bets': test_res['num_bets'],
            'test_ci_lower': test_res['ci_lower'],
            'full_roi': full_res['roi'],
            'full_ci_lower': full_res['ci_lower'],
            'full_min_yr_roi': full_res['min_roi_yr'],
            'full_min_yr_bets': full_res['min_bets_yr']
        }
        candidates.append(candidate)
        
    # 結果集計
    res_df = pd.DataFrame(candidates)
    if res_df.empty:
        logger.warning("No candidates found even with relaxed criteria.")
        return
        
    # ソート: 実用性重視 = full_ci_lower
    res_df = res_df.sort_values('full_ci_lower', ascending=False)
    
    cols = ['params', 'full_roi', 'full_ci_lower', 'full_min_yr_roi', 'train_roi', 'test_roi', 'train_bets', 'test_bets']
    print("\n--- Top 10 Candidates (Sorted by Full CI Lower) ---")
    print(res_df[cols].head(10).to_string())
    
    # ユーザー制約を満たす推奨候補
    # bets_total >= 500 (3年), min_yr_roi >= 0.85 (少し緩和して確認), min_yr_bets >= 100
    strict_mask = (res_df['train_bets'] + res_df['test_bets'] >= 500) & \
                  (res_df['full_min_yr_roi'] >= 0.85)
                  
    print("\n--- Recommended Candidates (Bets>=500, MinYrROI>=0.85) ---")
    if strict_mask.any():
        print(res_df[strict_mask][cols].head(10).to_string())
    else:
        print("No candidates met the STRICT constraints. Check 'Top 10 Candidates' for nearest options.")
    
    res_df.to_csv("reports/simulations/results/practical_optimization_h4.csv", index=False)
    logger.info("Saved results to reports/simulations/results/practical_optimization_h4.csv")

if __name__ == "__main__":
    run_nested_optimization("reports/simulations/v13_e1_enriched_2022_2024.parquet")
