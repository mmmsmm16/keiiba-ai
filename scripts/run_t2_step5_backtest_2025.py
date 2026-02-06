
import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from tqdm import tqdm
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_predictions():
    path = "data/temp_t2/T2_meta_predictions_2025.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions not found at {path}")
    df = pd.read_parquet(path)
    # odds_10min and odds_ratio_60_10 might need to be merged back if not in path
    # Actually T2_meta_predictions_2025.parquet saved earlier might not have all columns.
    # Let's check columns.
    return df

def load_payouts(year=2025):
    loader = JraVanDataLoader()
    engine = loader.engine
    query = f"SELECT * FROM jvd_hr WHERE kaisai_nen = '{year}'"
    logger.info(f"Loading Payouts for {year}...")
    df_hr = pd.read_sql(query, engine)
    
    df_hr['race_id'] = (
        df_hr['kaisai_nen'].astype(str).str.strip() + 
        df_hr['keibajo_code'].astype(str).str.strip() + 
        df_hr['kaisai_kai'].astype(str).str.strip().str.zfill(2) + 
        df_hr['kaisai_nichime'].astype(str).str.strip().str.zfill(2) + 
        df_hr['race_bango'].astype(str).str.strip().str.zfill(2)
    )
    return df_hr

def simulate_strategy(df_preds, df_hr, strategy_name, func, **kwargs):
    race_ids = df_preds['race_id'].unique()
    total_ret = 0
    total_cost = 0
    race_count = 0
    hit_count = 0
    
    df_hr_indexed = df_hr.set_index('race_id')
    
    logger.info(f"Simulating: {strategy_name}")
    
    for rid in tqdm(race_ids, desc=strategy_name):
        if rid not in df_hr_indexed.index:
            continue
            
        row_hr = df_hr_indexed.loc[rid]
        if isinstance(row_hr, pd.DataFrame): row_hr = row_hr.iloc[0]
        
        sub_df = df_preds[df_preds['race_id'] == rid]
        
        ret, cost, hit = func(sub_df, row_hr, **kwargs)
        
        if cost > 0:
            total_ret += ret
            total_cost += cost
            race_count += 1
            if hit: hit_count += 1
            
    roi = total_ret / total_cost if total_cost > 0 else 0
    hit_rate = hit_count / race_count if race_count > 0 else 0
    
    logger.info(f"[{strategy_name}] ROI: {roi*100:.2f}%, HitRate: {hit_rate*100:.2f}%, Bets: {race_count}, Cost: {total_cost}, Ret: {total_ret}")
    return {'strategy': strategy_name, 'roi': roi, 'hit_rate': hit_rate, 'cost': total_cost, 'return': total_ret, 'n_races': race_count}

# --- Strategies ---

def strat_win_ev(sub_df, row_hr, ev_threshold=1.1, min_prob=0.1, max_odds=30.0):
    # Calculate EV
    # Use odds_final for EV calculation in backtest IF we want to see "realized" EV.
    # But usually we use odds_10min for DECISION.
    
    sub_df = sub_df.copy()
    sub_df['ev'] = sub_df['meta_prob'] * sub_df['odds_10min']
    
    # Filter
    targets = sub_df[(sub_df['ev'] >= ev_threshold) & (sub_df['meta_prob'] >= min_prob) & (sub_df['odds_10min'] <= max_odds)]
    if targets.empty: return 0, 0, False
    
    # Choose top EV horse (or ALL in case of multiple?)
    # Let's say we bet on all horses meeting criteria
    cost = len(targets) * 100
    ret = 0
    hit = False
    
    k_h = 'haraimodoshi_tansho_1a'
    k_p = 'haraimodoshi_tansho_1b'
    
    if k_h in row_hr and k_p in row_hr:
        try:
            h_win = int(row_hr[k_h])
            pay = float(row_hr[k_p])
            for _, row in targets.iterrows():
                if int(row['horse_number']) == h_win:
                    ret += pay
                    hit = True
        except: pass
        
    return ret, cost, hit

def strat_place_ev(sub_df, row_hr, ev_threshold=1.1, prob_mult=2.5, max_odds=10.0):
    """
    Simulate Place (Fukusho) betting using win_prob * multiplier.
    """
    sub_df = sub_df.copy()
    # Estimate place probability (heuristic)
    sub_df['place_prob'] = (sub_df['meta_prob'] * prob_mult).clip(upper=0.9)
    
    # We need Place Odds. If not available, we can't do this accurately.
    # JRA-VAN haraimodoshi_fukusho_1b is the payout. 
    # Let's assume we use approximate odds or a fixed dividend for screening?
    # Actually, for backtest, we can use the ACTUAL payout to find if it WOULD have been profitable.
    # But for decision, we need Place Odds 10min.
    # If odds_place_10min is missing, skip.
    if 'odds_place_10min' not in sub_df.columns:
        return 0, 0, False
        
    sub_df['ev_place'] = sub_df['place_prob'] * sub_df['odds_place_10min']
    
    targets = sub_df[(sub_df['ev_place'] >= ev_threshold) & (sub_df['odds_place_10min'] <= max_odds)]
    if targets.empty: return 0, 0, False
    
    cost = len(targets) * 100
    ret = 0
    hit = False
    
    # Place payouts: haraimodoshi_fukusho_1a (horse), 1b (money), up to 3-5
    for i in range(1, 6):
        k_h = f'haraimodoshi_fukusho_{i}a'
        k_p = f'haraimodoshi_fukusho_{i}b'
        if k_h in row_hr and k_p in row_hr:
            try:
                h_win = int(row_hr[k_h])
                pay = float(row_hr[k_p])
                for _, row in targets.iterrows():
                    if int(row['horse_number']) == h_win:
                        ret += pay
                        hit = True
            except: pass
    return ret, cost, hit

def strat_wide_ev(sub_df, row_hr, ev_threshold=1.1, n_horses=3):
    """
    Bet on all combinations of top N horses if pair EV is high.
    Pair EV approx: P(i in Top3) * P(j in Top3) * Wide_Odds
    """
    sorted_df = sub_df.sort_values('meta_prob', ascending=False).head(n_horses)
    if len(sorted_df) < 2: return 0, 0, False
    
    # Estimate Top3 prob
    sorted_df = sorted_df.copy()
    sorted_df['p_top3'] = (sorted_df['meta_prob'] * 2.5).clip(upper=0.8)
    
    selected_horses = sorted_df['horse_number'].astype(int).tolist()
    
    # Cost: nC2
    import math
    n_bets = math.comb(len(selected_horses), 2)
    cost = n_bets * 100
    ret = 0
    hit = False
    
    # Wide payouts: haraimodoshi_wide_1a (pair "0102"), 1b (money), up to 3-7
    for i in range(1, 8):
        k_h = f'haraimodoshi_wide_{i}a'
        k_p = f'haraimodoshi_wide_{i}b'
        if k_h in row_hr and k_p in row_hr:
            h_str = str(row_hr[k_h]).strip()
            p_val = str(row_hr[k_p]).strip()
            try:
                pay = float(p_val) if p_val else 0
                if len(h_str) == 4 and pay > 0:
                    h1, h2 = int(h_str[0:2]), int(h_str[2:4])
                    if h1 in selected_horses and h2 in selected_horses:
                        ret += pay
                        hit = True
            except: pass
    return ret, cost, hit

def strat_trio_nagashi_ev(sub_df, row_hr, n_followers=7, ev_threshold=1.1):
    # Sort by meta_prob
    sorted_df = sub_df.sort_values('meta_prob', ascending=False)
    if len(sorted_df) < 1 + n_followers: return 0, 0, False
    
    top1 = sorted_df.iloc[0]
    # Filter: Only bet if Top1 Win EV > threshold or similar
    if top1['meta_prob'] * top1['odds_10min'] < ev_threshold: return 0, 0, False
    
    axis = int(top1['horse_number'])
    followers = set(sorted_df.iloc[1:1+n_followers]['horse_number'].astype(int))
    
    n_bets = math.comb(len(followers), 2)
    cost = n_bets * 100
    ret = 0
    hit = False
    
    for i in range(1, 4):
        k_horse = f'haraimodoshi_sanrenpuku_{i}a'
        k_pay = f'haraimodoshi_sanrenpuku_{i}b'
        if k_horse not in row_hr or k_pay not in row_hr: continue
        h_str = str(row_hr[k_horse]).strip()
        try:
            p_val = str(row_hr[k_pay]).strip()
            pay = float(p_val) if p_val else 0
        except:
            pay = 0
        if not h_str or pay == 0: continue
        
        h_list = [int(h_str[j:j+2]) for j in range(0, 6, 2)]
        win_set = set(h_list)
        if axis in win_set and (win_set - {axis}).issubset(followers):
            ret += pay
            hit = True
            
    return ret, cost, hit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025, help="Simulation Year")
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting {args.year} High-Precision Backtest...")
    
    # Load Predictions
    df_preds = load_predictions()
    
    # Filter by year in predictions if available, else assume date matches
    if 'date' in df_preds.columns:
        df_preds = df_preds[df_preds['date'].dt.year == args.year].copy()
    
    if df_preds.empty:
        logger.error(f"No predictions for year {args.year}!")
        return

    # Merge odds_10min and odds_ratio for backtest.
    t1_path = "data/temp_t1/T1_features_2024_2025.parquet"
    if os.path.exists(t1_path):
        df_t1 = pd.read_parquet(t1_path)
        df_t1['race_id'] = df_t1['race_id'].astype(str)
        df_t1 = df_t1.drop_duplicates(['race_id', 'horse_number'])
        df_preds = pd.merge(df_preds, df_t1[['race_id', 'horse_number', 'odds_10min', 'odds_ratio_60_10']], 
                            on=['race_id', 'horse_number'], how='left')
        logger.info("Merged odds_10min and odds_ratio for backtest.")
    
    # Calculate prob_market
    df_preds['prob_market'] = 0.8 / (df_preds['odds_10min'].fillna(100) + 1e-9)
    df_preds['ev'] = df_preds['meta_prob'] * df_preds['odds_10min'].fillna(0)
    
    df_hr = load_payouts(args.year)
    
    results = []
    
    # Win EV Strategies (Low Thres for Volume)
    logger.info("--- Win EV Strategies (Volume Focus) ---")
    ev_list = [0.95, 1.0, 1.05, 1.1]
    prob_list = [0.10, 0.15, 0.20]
    for ev_th in ev_list:
        for p_th in prob_list:
            res = simulate_strategy(df_preds, df_hr, f"Win EV [th={ev_th}, p>={p_th}]", strat_win_ev, ev_threshold=ev_th, min_prob=p_th)
            results.append(res)
        
    # Market-Relative Win (Volume Focus)
    logger.info("--- Market-Relative Win (Volume Focus) ---")
    for ratio_th in [1.1, 1.2, 1.3]:
        def strat_market_rel(sub_df, row_hr, ratio=1.2):
            sub_df = sub_df.copy()
            # Buy if model is more confident than market
            sub_df['conf_ratio'] = sub_df['meta_prob'] / (sub_df['prob_market'] + 1e-9)
            targets = sub_df[sub_df['conf_ratio'] >= ratio]
            return strat_win_ev(targets, row_hr, ev_threshold=0, min_prob=0.1) # No fixed EV threshold
            
        res = simulate_strategy(df_preds, df_hr, f"Win Ratio [th={ratio_th}]", strat_market_rel, ratio=ratio_th)
        results.append(res)

    # Wide Strategies (Very High Frequency)
    logger.info("--- Wide Strategies (High Volume) ---")
    for ev_th in [1.0, 1.1]:
        for n in [3, 4, 5]:
            res = simulate_strategy(df_preds, df_hr, f"Wide Box {n} [ev_th={ev_th}]", strat_wide_ev, ev_threshold=ev_th, n_horses=n)
            results.append(res)

    # Trio Nagashi (Balanced)
    logger.info("--- Trio Nagashi (Balanced) ---")
    for ev_th in [1.0, 1.1, 1.2]:
        res = simulate_strategy(df_preds, df_hr, f"Trio Nagashi [ev_th={ev_th}]", strat_trio_nagashi_ev, n_followers=7, ev_threshold=ev_th)
        results.append(res)

    # Save
    df_res = pd.DataFrame(results)
    output_path = "reports/backtest_t2_2025_results.csv"
    os.makedirs("reports", exist_ok=True)
    df_res.to_csv(output_path, index=False)
    print("\nFinal Results:")
    print(df_res.sort_values('roi', ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
