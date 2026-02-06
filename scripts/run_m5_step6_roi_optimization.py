
import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import itertools
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Simulation Engines ---

def simulate_win(df, p_win_th, ev_th, margin_th):
    """
    Win Strategy (Max 1 per race)
    """
    # Filter and Rank
    df = df.copy()
    
    # P4. Odds Cap Guardrail (No Shrinkage)
    MAX_ODDS = 20.0
    
    df['ev_win'] = df['p_win'] * df['win_odds']
    
    # Sort and calc margin
    df = df.sort_values(['race_id', 'p_win'], ascending=[True, False])
    df['p_win_rank'] = df.groupby('race_id').cumcount() + 1
    
    # Margin (Top1 - Top2)
    top2 = df[df['p_win_rank'] <= 2][['race_id', 'p_win_rank', 'p_win']]
    top2_pivot = top2.pivot(index='race_id', columns='p_win_rank', values='p_win')
    top2_pivot['margin'] = top2_pivot[1] - top2_pivot[2]
    
    df = pd.merge(df, top2_pivot[['margin']], on='race_id', how='left')
    
    # Bet Selection
    bets = df[
        (df['p_win_rank'] == 1) &
        (df['p_win'] >= p_win_th) &
        (df['ev_win'] >= ev_th) &
        (df['margin'] >= margin_th) &
        (df['win_odds'] < MAX_ODDS) # Guardrail
    ].copy()
    
    return bets[['race_id', 'date', 'year', 'horse_number', 'p_win', 'win_odds', 'ev_win', 'win_payoff']]

def simulate_umaren(df, p_um_th, ev_um_th, max_pairs=3):
    """
    Umaren Strategy (Harville approximation)
    """
    races = []
    
    unique_races = df.groupby('race_id')
    
    for rid, group in unique_races:
        # Get raw data
        horses = group['horse_number'].values
        p_wins = group['p_win'].values
        p_top2s = group['p_top2'].values # We can use p_top2 or Harville
        date = group['date'].iloc[0]
        year = group['year'].iloc[0]
        
        try:
            um_odds = json.loads(group['umaren_odds_json'].iloc[0])
            um_pay = json.loads(group['umaren_payoff_json'].iloc[0])
        except: continue
        
        n = len(p_wins)
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                h1, h2 = horses[i], horses[j]
                pw1, pw2 = p_wins[i], p_wins[j]
                # Harville: P(i,j) = p_i*p_j/(1-p_i) + p_j*p_i/(1-p_j)
                p_pair = pw1 * (pw2 / (1.0 - pw1 + 1e-9)) + pw2 * (pw1 / (1.0 - pw2 + 1e-9))
                
                combo = f"{min(h1,h2):02}-{max(h1,h2):02}"
                odds = um_odds.get(combo, 0.0)
                if odds <= 0: continue
                
                ev = p_pair * odds
                if p_pair >= p_um_th and ev >= ev_um_th:
                    # Payoff lookup (try both formatted and unformatted)
                    combo_clean = combo.replace("-", "")
                    payout = um_pay.get(combo, um_pay.get(combo_clean, 0))
                    
                    candidates.append({
                        'race_id': rid, 'date': date, 'year': year, 'combo': combo,
                        'p_um': p_pair, 'odds': odds, 'ev': ev, 
                        'payout': payout
                    })
        
        if candidates:
            # Sort by EV and pick top K
            c_df = pd.DataFrame(candidates).sort_values('ev', ascending=False).head(max_pairs)
            races.append(c_df)
            
    if not races: return pd.DataFrame()
    return pd.concat(races)

def simulate_place(df, p_pl_th, ev_pl_th, max_bets=2):
    """
    Place Strategy (Max K per race)
    """
    df = df.copy()
    df['ev_place'] = df['p_top3'] * df['place_odds_mid']
    
    # Filter
    bets = df[
        (df['p_top3'] >= p_pl_th) &
        (df['ev_place'] >= ev_pl_th)
    ].copy()
    
    if bets.empty: return pd.DataFrame()
    
    # Sort and pick top K per race
    bets = bets.sort_values(['race_id', 'ev_place'], ascending=[True, False])
    bets = bets.groupby('race_id').head(max_bets)
    
    return bets[['race_id', 'date', 'year', 'horse_number', 'p_top3', 'place_odds_mid', 'ev_place', 'place_payoff']]

def simulate_portfolio(win_bets, um_bets, pl_bets, daily_cap=50000, race_cap=10000):
    """
    Portfolio Simulation (O4)
    - Combined bets from all strategies
    - Apply daily and race caps
    """
    # Standardize columns
    w = win_bets.rename(columns={'win_payoff': 'payout', 'horse_number': 'selection'})[['race_id', 'date', 'year', 'selection', 'payout']]
    w['type'] = 'Win'
    
    u = um_bets.rename(columns={'combo': 'selection'})[['race_id', 'date', 'year', 'selection', 'payout']]
    u['type'] = 'Umaren'
    
    p = pl_bets.rename(columns={'place_payoff': 'payout', 'horse_number': 'selection'})[['race_id', 'date', 'year', 'selection', 'payout']]
    p['type'] = 'Place'
    
    combined = pd.concat([w, u, p])
    if combined.empty: return combined
    
    # Sort by date/race to simulate chronological order
    combined = combined.sort_values(['date', 'race_id'])
    
    # Apply Race Cap (Simple: if bets in race > cap, scale down. But here we use fixed 1000 yen/bet? 
    # Let's assume 1000 yen per bet for now.)
    # If 10 bets in a race, cost = 10,000. Cap 10,000 means all ok.
    # If cost > race_cap, we truncate extra bets?
    
    # For now, let's keep it simple: 100 yen per bet (matching payout unit)
    combined['cost'] = 100
    
    # Group by race and apply cap
    combined['race_cost_rank'] = combined.groupby('race_id').cumcount()
    # If we want to strictly follow cap:
    combined = combined[combined['race_cost_rank'] < (race_cap // 100)]
    
    # Daily Cap
    combined['daily_cost_cumsum'] = combined.groupby('date')['cost'].cumsum()
    combined = combined[combined['daily_cost_cumsum'] <= daily_cap]
    
    # Consistency Check
    total_cost = combined['cost'].sum()
    total_payout = combined['payout'].sum()
    if not combined.empty:
        calc_roi = total_payout / total_cost
        # logger.info(f"Audit: Portfolio ROI {calc_roi:.4f} (Cost={total_cost}, Pay={total_payout})")
        assert total_cost == len(combined) * 100, "Cost mismatch"
    
    return combined

# --- 2. Evaluation Wrapper ---

def evaluate(bets, label="Win"):
    if bets.empty: return {f"{label}_ROI": 0, f"{label}_Count": 0, f"{label}_HitRate": 0}
    
    total_bets = len(bets)
    # payoff is per 100 yen
    payoff_col = 'payout'
    for col in ['win_payoff', 'place_payoff']:
        if col in bets.columns:
            payoff_col = col
            break
    
    total_return = bets[payoff_col].sum()
    roi = total_return / (total_bets * 100)
    
    # Race based hit rate
    hits = (bets[payoff_col] > 0).sum()
    hit_rate = hits / total_bets
    
    return {
        f"{label}_ROI": roi,
        f"{label}_Count": total_bets,
        f"{label}_HitRate": hit_rate
    }

# --- 3. Main Optimization ---

def main():
    logger.info("🚀 Starting M5 Step 6: ROI Optimization & Pareto")
    
    DATA_PATH = "reports/simulations/v24_m5_roi_dataset_2022_2024.parquet"
    if not os.path.exists(DATA_PATH):
        logger.error(f"Missing dataset: {DATA_PATH}. Run Step 5 first.")
        return
        
    df = pd.read_parquet(DATA_PATH)
    train_df = df[df['year'].isin([2022, 2023])].copy()
    test_df = df[df['year'] == 2024].copy()
    
    # 3-1. Win Optimization
    logger.info("Optimizing Win Strategy...")
    p_ths = [0.10, 0.15, 0.20, 0.25, 0.30]
    ev_ths = [1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
    m_ths = [0.0, 0.02, 0.05]
    
    win_results = []
    for p, e, m in itertools.product(p_ths, ev_ths, m_ths):
        res = simulate_win(train_df, p, e, m)
        metrics = evaluate(res, "Win")
        metrics.update({'p_th': p, 'ev_th': e, 'm_th': m})
        win_results.append(metrics)
        
    win_res_df = pd.DataFrame(win_results)
    win_res_df.to_csv("reports/simulations/win_optimization_train.csv", index=False)
    
    # 3-2. Umaren Optimization
    logger.info("Optimizing Umaren Strategy...")
    p_um_ths = [0.010, 0.015, 0.020]
    ev_um_ths = [1.2, 1.5, 2.0]
    
    um_results = []
    for p, e in itertools.product(p_um_ths, ev_um_ths):
        res = simulate_umaren(train_df, p, e, max_pairs=3)
        metrics = evaluate(res, "Umaren")
        metrics.update({'p_th': p, 'ev_th': e})
        um_results.append(metrics)
        
    um_res_df = pd.DataFrame(um_results)
    um_res_df.to_csv("reports/simulations/umaren_optimization_train.csv", index=False)
    
    # 3-3. Place Optimization
    logger.info("Optimizing Place Strategy...")
    p_pl_ths = [0.25, 0.30, 0.35, 0.40]
    ev_pl_ths = [1.1, 1.15, 1.25]
    
    pl_results = []
    for p, e in itertools.product(p_pl_ths, ev_pl_ths):
        res = simulate_place(train_df, p, e, max_bets=2)
        metrics = evaluate(res, "Place")
        metrics.update({'p_th': p, 'ev_th': e})
        pl_results.append(metrics)
        
    pl_res_df = pd.DataFrame(pl_results)
    pl_res_df.to_csv("reports/simulations/place_optimization_train.csv", index=False)
    
    # 3-4. Pareto Analysis & Verification
    # Pick "Recommended" from Win, Umaren, Place
    # (High ROI, Reasonable Frequency)
    
    def pick_best(res_df, roi_col, cnt_col):
        # Pareto: ROI >= 1.1 and max Count? Or max ROI with Cnt > 100?
        candidates = res_df[res_df[cnt_col] > 100].sort_values(roi_col, ascending=False)
        return candidates.iloc[0] if not candidates.empty else None

    best_win = pick_best(win_res_df, "Win_ROI", "Win_Count")
    best_um = pick_best(um_res_df, "Umaren_ROI", "Umaren_Count")
    best_pl = pick_best(pl_res_df, "Place_ROI", "Place_Count")
    
    logger.info(f"Best Win (Train): ROI={best_win['Win_ROI']:.3f} Cnt={best_win['Win_Count']}")
    logger.info(f"Best Umaren (Train): ROI={best_um['Umaren_ROI']:.3f} Cnt={best_um['Umaren_Count']}")
    logger.info(f"Best Place (Train): ROI={best_pl['Place_ROI']:.3f} Cnt={best_pl['Place_Count']}")
    
    # Verify on 2024
    logger.info("Verifying on 2024 (Test)...")
    v_win = simulate_win(test_df, best_win['p_th'], best_win['ev_th'], best_win['m_th'])
    v_um = simulate_umaren(test_df, best_um['p_th'], best_um['ev_th'])
    v_pl = simulate_place(test_df, best_pl['p_th'], best_pl['ev_th'])
    
    test_win_metrics = evaluate(v_win, "Win")
    test_um_metrics = evaluate(v_um, "Umaren")
    test_pl_metrics = evaluate(v_pl, "Place")
    
    # 3-5. Portfolio Simulation
    logger.info("Simulating Portfolio...")
    portfolio = simulate_portfolio(v_win, v_um, v_pl, daily_cap=50000, race_cap=10000)
    
    port_roi = portfolio['payout'].sum() / (portfolio['cost'].sum() + 1e-9)
    port_bets = len(portfolio)
    port_days = portfolio['date'].nunique()
    port_hits = (portfolio.groupby('date')['payout'].sum() > 0).sum()
    hit_day_rate = port_hits / port_days if port_days > 0 else 0
    
    # Final Summary Report Generation
    report = f"""# v24_m5_final ROI Simulation Report

## Recommended Betting Rules
- **Win**: p_win >= {best_win['p_th']:.2f}, EV >= {best_win['ev_th']:.1f}, margin >= {best_win['m_th']:.2f}
- **Umaren**: p_um >= {best_um['p_th']:.3f}, EV >= {best_um['ev_th']:.1f}, Max 3
- **Place**: p_top3 >= {best_pl['p_th']:.2f}, EV >= {best_pl['ev_th']:.1f}, Max 2

## Combined Portfolio (2024 Test)
- **Total Bets**: {port_bets}
- **ROI**: {port_roi:.3f}
- **Hit Day Rate**: {hit_day_rate:.1%} ({port_hits}/{port_days} days)
- **Total Profit**: {(portfolio['payout'].sum() - portfolio['cost'].sum()):,.0f} JPY (assuming 1,000 JPY unit)

## Individual Performance (2024 Test)
| Ticket | Condition | ROI | Count | Hit Rate |
|---|---|---|---|---|
| Win | {best_win['p_th']}, {best_win['ev_th']}, {best_win['m_th']} | {test_win_metrics['Win_ROI']:.3f} | {test_win_metrics['Win_Count']} | {test_win_metrics['Win_HitRate']:.1%} |
| Umaren | {best_um['p_th']}, {best_um['ev_th']} | {test_um_metrics['Umaren_ROI']:.3f} | {test_um_metrics['Umaren_Count']} | {test_um_metrics['Umaren_HitRate']:.1%} |
| Place | {best_pl['p_th']}, {best_pl['ev_th']} | {test_pl_metrics['Place_ROI']:.3f} | {test_pl_metrics['Place_Count']} | {test_pl_metrics['Place_HitRate']:.1%} |
"""
    with open("reports/simulations/v24_m5_roi_report.md", "w") as f:
        f.write(report)
        
    logger.info("M5 Step 6 Complete!")

if __name__ == "__main__":
    main()
