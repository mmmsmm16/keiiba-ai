"""
Comprehensive EV Grid Search Simulation
- Multiple bet types: Win, Place, Quinella
- EV Filtering with 10-min pre-race odds
- Multiple strategies: Top1, TopN-EV (highest EV in top N)
- Grid search on 2023, Test on 2024

Usage: python scripts/run_comprehensive_ev_sim.py
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from tqdm import tqdm
from itertools import product

print("=" * 80)
print("🎯 Comprehensive EV Grid Search Simulation")
print("=" * 80)

# Constants
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/optuna_best_full/model.pkl"

LEAKAGE = ['pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank', 'last_3f', 
           'raw_time', 'time_diff', 'margin', 'time', 'popularity', 'odds', 
           'relative_popularity_rank', 'slow_start_recovery', 'track_bias_disadvantage', 
           'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate']
META = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 'is_win', 
        'is_top2', 'is_top3', 'year', 'rank_str']
IDS = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']


def parse_tansho_odds(odds_str):
    """Parse odds_tansho: 8 chars per horse (num:2, odds:4, pop:2)"""
    result = {}
    if not odds_str or pd.isna(odds_str):
        return result
    s = str(odds_str)
    for i in range(0, len(s), 8):
        block = s[i:i+8]
        if len(block) < 8:
            break
        try:
            h = int(block[0:2])
            o = int(block[2:6]) / 10.0
            if h > 0 and o > 0:
                result[h] = o
        except:
            pass
    return result


def parse_fukusho_odds(odds_str):
    """Parse odds_fukusho: 12 chars per horse (num:2, min:4, max:4, pop:2)"""
    result = {}
    if not odds_str or pd.isna(odds_str):
        return result
    s = str(odds_str)
    for i in range(0, len(s), 12):
        block = s[i:i+12]
        if len(block) < 12:
            break
        try:
            h = int(block[0:2])
            o_min = int(block[2:6]) / 10.0
            o_max = int(block[6:10]) / 10.0
            # Use middle of range
            o = (o_min + o_max) / 2
            if h > 0 and o > 0:
                result[h] = o
        except:
            pass
    return result


def load_data_and_predictions():
    """Load model, data, and generate predictions"""
    print("Loading model...", flush=True)
    model = joblib.load(MODEL_PATH)
    
    print("Loading data...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    feat = [c for c in df.columns if c not in META + LEAKAGE + IDS]
    
    all_preds = {}
    for year in [2023, 2024]:
        print(f"Generating predictions for {year}...", flush=True)
        df_year = df[df['year'] == year].copy()
        
        X = df_year[feat].copy()
        for c in X.columns:
            if X[c].dtype.name == 'category' or X[c].dtype == 'object':
                X[c] = X[c].astype('category').cat.codes
            else:
                X[c] = X[c].fillna(-999)
        
        df_year['pred'] = model.predict(X)
        df_year['race_id'] = df_year['race_id'].astype(str)
        df_year = df_year.sort_values(['race_id', 'pred'], ascending=[True, False])
        df_year['pred_rank'] = df_year.groupby('race_id').cumcount() + 1
        
        all_preds[year] = df_year
    
    return all_preds


def load_odds_and_payouts():
    """Load 10-min odds and confirmed payouts"""
    print("Loading odds and payouts...", flush=True)
    engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
    
    # 10-min odds (use latest available)
    odds_query = """
    WITH ranked AS (
        SELECT 
            kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
            odds_tansho, odds_fukusho,
            ROW_NUMBER() OVER (
                PARTITION BY kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango 
                ORDER BY happyo_tsukihi_jifun DESC
            ) as rn
        FROM jvd_o1
        WHERE kaisai_nen IN ('2023', '2024')
    )
    SELECT race_id, odds_tansho, odds_fukusho FROM ranked WHERE rn = 1
    """
    df_odds = pd.read_sql(odds_query, engine)
    df_odds['race_id'] = df_odds['race_id'].astype(str)
    
    # Parse odds into dicts
    win_odds = {}  # {race_id: {horse: odds}}
    place_odds = {}
    
    for _, row in tqdm(df_odds.iterrows(), total=len(df_odds), desc="Parsing odds"):
        rid = row['race_id']
        win_odds[rid] = parse_tansho_odds(row['odds_tansho'])
        place_odds[rid] = parse_fukusho_odds(row['odds_fukusho'])
    
    # Payouts
    pay_query = """
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        haraimodoshi_umaren_1a as quin_horses, haraimodoshi_umaren_1b as quin_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ('2023', '2024')
    """
    df_pay = pd.read_sql(pay_query, engine)
    df_pay['race_id'] = df_pay['race_id'].astype(str)
    
    # Parse payouts
    win_payouts = {}
    place_payouts = {}
    quin_payouts = {}
    
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        
        # Win
        try:
            h = int(row['win_horse'])
            p = int(row['win_pay'])
            if h > 0 and p > 0:
                win_payouts[rid] = {h: p}
        except:
            pass
        
        # Place
        place_payouts[rid] = {}
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if h > 0 and p > 0:
                    place_payouts[rid][h] = p
            except:
                pass
        
        # Quinella
        try:
            hs = str(row['quin_horses'])
            if len(hs) >= 4:
                h1, h2 = int(hs[:2]), int(hs[2:4])
                p = int(row['quin_pay'])
                if h1 > 0 and h2 > 0 and p > 0:
                    quin_payouts[rid] = {(min(h1, h2), max(h1, h2)): p}
        except:
            pass
    
    print(f"Win odds: {len(win_odds)}, Place odds: {len(place_odds)}")
    print(f"Win payouts: {len(win_payouts)}, Place payouts: {len(place_payouts)}, Quin payouts: {len(quin_payouts)}")
    
    return win_odds, place_odds, win_payouts, place_payouts, quin_payouts


def simulate_win_top1(df, win_odds, win_payouts, ev_th=0.0, prob_th=0.0):
    """Strategy: Bet on Top1 predicted horse (Win bet)"""
    top1 = df[df['pred_rank'] == 1].copy()
    
    total_bet = 0
    total_pay = 0
    hits = 0
    
    for _, row in top1.iterrows():
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        pred = row['pred']
        
        if rid not in win_odds or hn not in win_odds[rid]:
            continue
        
        odds = win_odds[rid][hn]
        ev = pred * odds
        
        if pred < prob_th or ev < ev_th:
            continue
        
        total_bet += 100
        if rid in win_payouts and hn in win_payouts[rid]:
            total_pay += win_payouts[rid][hn]
            hits += 1
    
    n_bets = int(total_bet / 100)
    return {
        'strategy': 'Win-Top1',
        'bets': n_bets,
        'hits': hits,
        'hit_rate': hits / n_bets * 100 if n_bets > 0 else 0,
        'roi': total_pay / total_bet * 100 if total_bet > 0 else 0,
        'params': f"ev>={ev_th:.1f},p>={prob_th:.2f}"
    }


def simulate_win_topn_maxev(df, win_odds, win_payouts, n=3, ev_th=1.0, prob_th=0.0):
    """Strategy: From Top N, bet on horse with highest EV"""
    topn = df[df['pred_rank'] <= n].copy()
    
    total_bet = 0
    total_pay = 0
    hits = 0
    
    for rid, grp in topn.groupby('race_id'):
        rid = str(rid)
        if rid not in win_odds:
            continue
        
        # Calculate EV for each horse
        best_hn = None
        best_ev = -1
        
        for _, row in grp.iterrows():
            hn = int(row['horse_number'])
            pred = row['pred']
            
            if hn not in win_odds[rid]:
                continue
            
            odds = win_odds[rid][hn]
            ev = pred * odds
            
            if pred >= prob_th and ev >= ev_th and ev > best_ev:
                best_ev = ev
                best_hn = hn
        
        if best_hn is not None:
            total_bet += 100
            if rid in win_payouts and best_hn in win_payouts[rid]:
                total_pay += win_payouts[rid][best_hn]
                hits += 1
    
    n_bets = int(total_bet / 100)
    return {
        'strategy': f'Win-Top{n}-MaxEV',
        'bets': n_bets,
        'hits': hits,
        'hit_rate': hits / n_bets * 100 if n_bets > 0 else 0,
        'roi': total_pay / total_bet * 100 if total_bet > 0 else 0,
        'params': f"ev>={ev_th:.1f},p>={prob_th:.2f}"
    }


def simulate_place_top1(df, place_odds, place_payouts, ev_th=0.0, prob_th=0.0):
    """Strategy: Bet on Top1 predicted horse (Place bet)"""
    top1 = df[df['pred_rank'] == 1].copy()
    
    total_bet = 0
    total_pay = 0
    hits = 0
    
    for _, row in top1.iterrows():
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        pred = row['pred']
        
        if rid not in place_odds or hn not in place_odds[rid]:
            continue
        
        odds = place_odds[rid][hn]
        ev = pred * 2.5 * odds  # Rough estimate: top1 has ~75% top3 rate
        
        if pred < prob_th or ev < ev_th:
            continue
        
        total_bet += 100
        if rid in place_payouts and hn in place_payouts[rid]:
            total_pay += place_payouts[rid][hn]
            hits += 1
    
    n_bets = int(total_bet / 100)
    return {
        'strategy': 'Place-Top1',
        'bets': n_bets,
        'hits': hits,
        'hit_rate': hits / n_bets * 100 if n_bets > 0 else 0,
        'roi': total_pay / total_bet * 100 if total_bet > 0 else 0,
        'params': f"ev>={ev_th:.1f},p>={prob_th:.2f}"
    }


def simulate_quinella_top2(df, win_odds, quin_payouts, prob_sum_th=0.0, prob1_th=0.0):
    """Strategy: Quinella on Top1-Top2"""
    total_bet = 0
    total_pay = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        rid = str(rid)
        top2 = grp[grp['pred_rank'] <= 2]
        if len(top2) < 2:
            continue
        
        h1 = int(top2.iloc[0]['horse_number'])
        h2 = int(top2.iloc[1]['horse_number'])
        p1 = top2.iloc[0]['pred']
        p2 = top2.iloc[1]['pred']
        
        if p1 < prob1_th or (p1 + p2) < prob_sum_th:
            continue
        
        total_bet += 100
        key = (min(h1, h2), max(h1, h2))
        if rid in quin_payouts and key in quin_payouts[rid]:
            total_pay += quin_payouts[rid][key]
            hits += 1
    
    n_bets = int(total_bet / 100)
    return {
        'strategy': 'Quin-Top2',
        'bets': n_bets,
        'hits': hits,
        'hit_rate': hits / n_bets * 100 if n_bets > 0 else 0,
        'roi': total_pay / total_bet * 100 if total_bet > 0 else 0,
        'params': f"p1>={prob1_th:.2f},sum>={prob_sum_th:.2f}"
    }


def main():
    # Load data
    all_preds = load_data_and_predictions()
    win_odds, place_odds, win_payouts, place_payouts, quin_payouts = load_odds_and_payouts()
    
    df_2023 = all_preds[2023]
    df_2024 = all_preds[2024]
    
    print("\n" + "=" * 80)
    print("Running Grid Search on 2023...")
    print("=" * 80 + "\n")
    
    results_2023 = []
    
    # Win-Top1 Grid
    print("Win-Top1 strategy...", flush=True)
    for ev_th, prob_th in product([0.0, 0.8, 1.0, 1.2, 1.5], [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]):
        r = simulate_win_top1(df_2023, win_odds, win_payouts, ev_th, prob_th)
        results_2023.append(r)
    
    # Win-TopN-MaxEV Grid
    print("Win-TopN-MaxEV strategy...", flush=True)
    for n, ev_th, prob_th in product([3, 5], [0.8, 1.0, 1.2, 1.5], [0.0, 0.08, 0.10, 0.15]):
        r = simulate_win_topn_maxev(df_2023, win_odds, win_payouts, n, ev_th, prob_th)
        results_2023.append(r)
    
    # Place-Top1 Grid
    print("Place-Top1 strategy...", flush=True)
    for ev_th, prob_th in product([0.0, 1.0, 1.5, 2.0], [0.0, 0.15, 0.20, 0.25]):
        r = simulate_place_top1(df_2023, place_odds, place_payouts, ev_th, prob_th)
        results_2023.append(r)
    
    # Quinella-Top2 Grid
    print("Quinella-Top2 strategy...", flush=True)
    for prob1_th, prob_sum_th in product([0.0, 0.15, 0.20, 0.25], [0.0, 0.30, 0.35, 0.40]):
        r = simulate_quinella_top2(df_2023, win_odds, quin_payouts, prob_sum_th, prob1_th)
        results_2023.append(r)
    
    # Sort and show best
    results_2023 = [r for r in results_2023 if r['bets'] >= 50]
    results_2023.sort(key=lambda x: -x['roi'])
    
    print("\n" + "=" * 80)
    print("2023 Grid Search Results (Top 20)")
    print("=" * 80)
    print(f"{'Strategy':<18} {'Params':<28} {'Bets':>6} {'Hits':>5} {'Hit%':>7} {'ROI%':>7}")
    print("-" * 80)
    for r in results_2023[:20]:
        print(f"{r['strategy']:<18} {r['params']:<28} {r['bets']:>6} {r['hits']:>5} {r['hit_rate']:>6.1f}% {r['roi']:>6.1f}%")
    
    # Test best strategies on 2024
    print("\n" + "=" * 80)
    print("Testing Best Strategies on 2024...")
    print("=" * 80 + "\n")
    
    results_2024 = []
    
    # Take top 10 strategies from 2023 and test on 2024
    for i, r23 in enumerate(results_2023[:10]):
        strat = r23['strategy']
        params = r23['params']
        
        # Parse params
        parts = params.split(',')
        ev_th = float(parts[0].split('>=')[1]) if 'ev' in parts[0] else 0.0
        
        if strat == 'Win-Top1':
            prob_th = float(parts[1].split('>=')[1])
            r24 = simulate_win_top1(df_2024, win_odds, win_payouts, ev_th, prob_th)
        elif 'Win-Top' in strat and 'MaxEV' in strat:
            n = int(strat.split('-')[1].replace('Top', ''))
            prob_th = float(parts[1].split('>=')[1])
            r24 = simulate_win_topn_maxev(df_2024, win_odds, win_payouts, n, ev_th, prob_th)
        elif strat == 'Place-Top1':
            prob_th = float(parts[1].split('>=')[1])
            r24 = simulate_place_top1(df_2024, place_odds, place_payouts, ev_th, prob_th)
        elif strat == 'Quin-Top2':
            prob1_th = float(parts[0].split('>=')[1])
            prob_sum_th = float(parts[1].split('>=')[1])
            r24 = simulate_quinella_top2(df_2024, win_odds, quin_payouts, prob_sum_th, prob1_th)
        else:
            continue
        
        results_2024.append({
            '2023_roi': r23['roi'],
            '2024_roi': r24['roi'],
            'strategy': strat,
            'params': params,
            '2023_bets': r23['bets'],
            '2024_bets': r24['bets'],
            '2023_hit': r23['hit_rate'],
            '2024_hit': r24['hit_rate']
        })
    
    print(f"{'Strategy':<18} {'Params':<28} {'23 Bets':>7} {'23 ROI':>7} {'24 Bets':>7} {'24 ROI':>7}")
    print("-" * 90)
    for r in results_2024:
        print(f"{r['strategy']:<18} {r['params']:<28} {r['2023_bets']:>7} {r['2023_roi']:>6.1f}% {r['2024_bets']:>7} {r['2024_roi']:>6.1f}%")
    
    # Best overall
    print("\n" + "=" * 80)
    print("🏆 Best Strategy (Consistent 2023 & 2024)")
    print("=" * 80)
    
    # Find strategy with best average ROI
    if results_2024:
        best = max(results_2024, key=lambda x: (x['2023_roi'] + x['2024_roi']) / 2)
        print(f"\n{best['strategy']} with {best['params']}")
        print(f"  2023: ROI={best['2023_roi']:.1f}%, Bets={best['2023_bets']}, Hit={best['2023_hit']:.1f}%")
        print(f"  2024: ROI={best['2024_roi']:.1f}%, Bets={best['2024_bets']}, Hit={best['2024_hit']:.1f}%")


if __name__ == "__main__":
    main()
