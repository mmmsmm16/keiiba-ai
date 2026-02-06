"""
Extended Bet Type EV Grid Search Simulation
- All bet types: Win, Place, Quinella (Umaren), Wide, Exacta (Umatan), Trio (Sanrenpuku), Trifecta (Sanrentan)
- EV Filtering with 10-min pre-race odds
- Grid search on 2023, Test on 2024

Usage: python scripts/run_extended_bet_sim.py
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from tqdm import tqdm
from itertools import product

print("=" * 80)
print("🎯 Extended Bet Type EV Grid Search Simulation")
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


def load_all_payouts():
    """Load all bet type payouts"""
    print("Loading payouts...", flush=True)
    engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
    
    # Build query for all payout types
    query = """
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        -- Win
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        -- Place
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        -- Quinella (Umaren)
        haraimodoshi_umaren_1a as quin_horses, haraimodoshi_umaren_1b as quin_pay,
        -- Wide (multiple payouts possible)
        haraimodoshi_wide_1a as wide_1_horses, haraimodoshi_wide_1b as wide_1_pay,
        haraimodoshi_wide_2a as wide_2_horses, haraimodoshi_wide_2b as wide_2_pay,
        haraimodoshi_wide_3a as wide_3_horses, haraimodoshi_wide_3b as wide_3_pay,
        -- Exacta (Umatan)
        haraimodoshi_umatan_1a as exacta_horses, haraimodoshi_umatan_1b as exacta_pay,
        -- Trio (Sanrenpuku)
        haraimodoshi_sanrenpuku_1a as trio_horses, haraimodoshi_sanrenpuku_1b as trio_pay,
        -- Trifecta (Sanrentan)
        haraimodoshi_sanrentan_1a as trif_horses, haraimodoshi_sanrentan_1b as trif_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ('2023', '2024')
    """
    df_pay = pd.read_sql(query, engine)
    df_pay['race_id'] = df_pay['race_id'].astype(str)
    
    # Parse all payouts
    win_payouts = {}
    place_payouts = {}  # {race_id: {horse: pay}}
    quin_payouts = {}   # {race_id: {(h1,h2): pay}}
    wide_payouts = {}   # {race_id: {(h1,h2): pay}}
    exacta_payouts = {} # {race_id: {(h1,h2): pay}} - order matters
    trio_payouts = {}   # {race_id: {(h1,h2,h3): pay}}
    trif_payouts = {}   # {race_id: {(h1,h2,h3): pay}} - order matters
    
    for _, row in tqdm(df_pay.iterrows(), total=len(df_pay), desc="Parsing payouts"):
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
                    key = tuple(sorted([h1, h2]))
                    quin_payouts[rid] = {key: p}
        except:
            pass
        
        # Wide (multiple possible winners)
        wide_payouts[rid] = {}
        for i in [1, 2, 3]:
            try:
                hs = str(row[f'wide_{i}_horses'])
                if len(hs) >= 4:
                    h1, h2 = int(hs[:2]), int(hs[2:4])
                    p = int(row[f'wide_{i}_pay'])
                    if h1 > 0 and h2 > 0 and p > 0:
                        key = tuple(sorted([h1, h2]))
                        wide_payouts[rid][key] = p
            except:
                pass
        
        # Exacta (order matters: 1st-2nd)
        try:
            hs = str(row['exacta_horses'])
            if len(hs) >= 4:
                h1, h2 = int(hs[:2]), int(hs[2:4])
                p = int(row['exacta_pay'])
                if h1 > 0 and h2 > 0 and p > 0:
                    exacta_payouts[rid] = {(h1, h2): p}
        except:
            pass
        
        # Trio
        try:
            hs = str(row['trio_horses'])
            if len(hs) >= 6:
                h1, h2, h3 = int(hs[:2]), int(hs[2:4]), int(hs[4:6])
                p = int(row['trio_pay'])
                if h1 > 0 and h2 > 0 and h3 > 0 and p > 0:
                    key = tuple(sorted([h1, h2, h3]))
                    trio_payouts[rid] = {key: p}
        except:
            pass
        
        # Trifecta (order matters: 1st-2nd-3rd)
        try:
            hs = str(row['trif_horses'])
            if len(hs) >= 6:
                h1, h2, h3 = int(hs[:2]), int(hs[2:4]), int(hs[4:6])
                p = int(row['trif_pay'])
                if h1 > 0 and h2 > 0 and h3 > 0 and p > 0:
                    trif_payouts[rid] = {(h1, h2, h3): p}
        except:
            pass
    
    print(f"Win: {len(win_payouts)}, Place: {len(place_payouts)}, Quin: {len(quin_payouts)}")
    print(f"Wide: {len(wide_payouts)}, Exacta: {len(exacta_payouts)}, Trio: {len(trio_payouts)}, Trif: {len(trif_payouts)}")
    
    return {
        'win': win_payouts,
        'place': place_payouts,
        'quin': quin_payouts,
        'wide': wide_payouts,
        'exacta': exacta_payouts,
        'trio': trio_payouts,
        'trif': trif_payouts
    }


def load_10min_odds():
    """Load 10-min pre-race odds"""
    print("Loading 10-min odds...", flush=True)
    engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
    
    query = """
    WITH ranked AS (
        SELECT 
            kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
            odds_tansho,
            ROW_NUMBER() OVER (
                PARTITION BY kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango 
                ORDER BY happyo_tsukihi_jifun DESC
            ) as rn
        FROM jvd_o1
        WHERE kaisai_nen IN ('2023', '2024')
    )
    SELECT race_id, odds_tansho FROM ranked WHERE rn = 1
    """
    df_odds = pd.read_sql(query, engine)
    
    win_odds = {}
    for _, row in df_odds.iterrows():
        rid = str(row['race_id'])
        win_odds[rid] = parse_tansho_odds(row['odds_tansho'])
    
    print(f"Win odds loaded: {len(win_odds)}")
    return win_odds


# Simulation functions for each bet type
def sim_win_top1(df, win_odds, win_payouts, ev_th=0.0, prob_th=0.0):
    """Win bet on Top1"""
    top1 = df[df['pred_rank'] == 1]
    total_bet, total_pay, hits = 0, 0, 0
    
    for _, row in top1.iterrows():
        rid, hn, pred = str(row['race_id']), int(row['horse_number']), row['pred']
        if rid not in win_odds or hn not in win_odds[rid]:
            continue
        ev = pred * win_odds[rid][hn]
        if pred < prob_th or ev < ev_th:
            continue
        total_bet += 100
        if rid in win_payouts and hn in win_payouts[rid]:
            total_pay += win_payouts[rid][hn]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Win-Top1', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"ev>={ev_th:.1f},p>={prob_th:.2f}"}


def sim_place_top1(df, place_payouts, prob_th=0.0):
    """Place bet on Top1"""
    top1 = df[df['pred_rank'] == 1]
    total_bet, total_pay, hits = 0, 0, 0
    
    for _, row in top1.iterrows():
        rid, hn, pred = str(row['race_id']), int(row['horse_number']), row['pred']
        if pred < prob_th:
            continue
        total_bet += 100
        if rid in place_payouts and hn in place_payouts[rid]:
            total_pay += place_payouts[rid][hn]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Place-Top1', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"p>={prob_th:.2f}"}


def sim_wide_top2(df, wide_payouts, prob_th=0.0):
    """Wide bet on Top1-Top2"""
    total_bet, total_pay, hits = 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        rid = str(rid)
        top2 = grp[grp['pred_rank'] <= 2]
        if len(top2) < 2:
            continue
        
        h1, h2 = int(top2.iloc[0]['horse_number']), int(top2.iloc[1]['horse_number'])
        p1, p2 = top2.iloc[0]['pred'], top2.iloc[1]['pred']
        
        if p1 < prob_th:
            continue
        
        total_bet += 100
        key = tuple(sorted([h1, h2]))
        if rid in wide_payouts and key in wide_payouts[rid]:
            total_pay += wide_payouts[rid][key]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Wide-Top2', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"p1>={prob_th:.2f}"}


def sim_quin_top2(df, quin_payouts, prob_th=0.0, prob_sum_th=0.0):
    """Quinella bet on Top1-Top2"""
    total_bet, total_pay, hits = 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        rid = str(rid)
        top2 = grp[grp['pred_rank'] <= 2]
        if len(top2) < 2:
            continue
        
        h1, h2 = int(top2.iloc[0]['horse_number']), int(top2.iloc[1]['horse_number'])
        p1, p2 = top2.iloc[0]['pred'], top2.iloc[1]['pred']
        
        if p1 < prob_th or (p1 + p2) < prob_sum_th:
            continue
        
        total_bet += 100
        key = tuple(sorted([h1, h2]))
        if rid in quin_payouts and key in quin_payouts[rid]:
            total_pay += quin_payouts[rid][key]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Quin-Top2', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"p1>={prob_th:.2f},sum>={prob_sum_th:.2f}"}


def sim_exacta_top2(df, exacta_payouts, prob_th=0.0):
    """Exacta bet on Top1-Top2 (in order)"""
    total_bet, total_pay, hits = 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        rid = str(rid)
        top2 = grp[grp['pred_rank'] <= 2]
        if len(top2) < 2:
            continue
        
        h1, h2 = int(top2.iloc[0]['horse_number']), int(top2.iloc[1]['horse_number'])
        p1 = top2.iloc[0]['pred']
        
        if p1 < prob_th:
            continue
        
        total_bet += 100
        key = (h1, h2)  # Order matters!
        if rid in exacta_payouts and key in exacta_payouts[rid]:
            total_pay += exacta_payouts[rid][key]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Exacta-Top2', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"p1>={prob_th:.2f}"}


def sim_trio_top3(df, trio_payouts, prob_th=0.0):
    """Trio bet on Top1-Top2-Top3"""
    total_bet, total_pay, hits = 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        rid = str(rid)
        top3 = grp[grp['pred_rank'] <= 3]
        if len(top3) < 3:
            continue
        
        h1 = int(top3.iloc[0]['horse_number'])
        h2 = int(top3.iloc[1]['horse_number'])
        h3 = int(top3.iloc[2]['horse_number'])
        p1 = top3.iloc[0]['pred']
        
        if p1 < prob_th:
            continue
        
        total_bet += 100
        key = tuple(sorted([h1, h2, h3]))
        if rid in trio_payouts and key in trio_payouts[rid]:
            total_pay += trio_payouts[rid][key]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Trio-Top3', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"p1>={prob_th:.2f}"}


def sim_trif_top3(df, trif_payouts, prob_th=0.0):
    """Trifecta bet on Top1-Top2-Top3 (in order)"""
    total_bet, total_pay, hits = 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        rid = str(rid)
        top3 = grp[grp['pred_rank'] <= 3]
        if len(top3) < 3:
            continue
        
        h1 = int(top3.iloc[0]['horse_number'])
        h2 = int(top3.iloc[1]['horse_number'])
        h3 = int(top3.iloc[2]['horse_number'])
        p1 = top3.iloc[0]['pred']
        
        if p1 < prob_th:
            continue
        
        total_bet += 100
        key = (h1, h2, h3)  # Order matters!
        if rid in trif_payouts and key in trif_payouts[rid]:
            total_pay += trif_payouts[rid][key]
            hits += 1
    
    n = int(total_bet / 100)
    return {'strategy': 'Trif-Top3', 'bets': n, 'hits': hits,
            'hit_rate': hits/n*100 if n > 0 else 0,
            'roi': total_pay/total_bet*100 if total_bet > 0 else 0,
            'params': f"p1>={prob_th:.2f}"}


def main():
    # Load data
    all_preds = load_data_and_predictions()
    payouts = load_all_payouts()
    win_odds = load_10min_odds()
    
    df_2023 = all_preds[2023]
    df_2024 = all_preds[2024]
    
    print("\n" + "=" * 80)
    print("Running Grid Search on 2023...")
    print("=" * 80 + "\n")
    
    results_2023 = []
    
    # Win-Top1 with EV filter
    print("Win-Top1...", flush=True)
    for ev_th, prob_th in product([0.0, 1.0, 1.2, 1.5, 2.0], [0.0, 0.10, 0.15, 0.20, 0.25]):
        r = sim_win_top1(df_2023, win_odds, payouts['win'], ev_th, prob_th)
        results_2023.append(r)
    
    # Place-Top1
    print("Place-Top1...", flush=True)
    for prob_th in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
        r = sim_place_top1(df_2023, payouts['place'], prob_th)
        results_2023.append(r)
    
    # Wide-Top2
    print("Wide-Top2...", flush=True)
    for prob_th in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
        r = sim_wide_top2(df_2023, payouts['wide'], prob_th)
        results_2023.append(r)
    
    # Quinella-Top2
    print("Quin-Top2...", flush=True)
    for prob_th, prob_sum_th in product([0.0, 0.15, 0.20], [0.0, 0.30, 0.40]):
        r = sim_quin_top2(df_2023, payouts['quin'], prob_th, prob_sum_th)
        results_2023.append(r)
    
    # Exacta-Top2
    print("Exacta-Top2...", flush=True)
    for prob_th in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
        r = sim_exacta_top2(df_2023, payouts['exacta'], prob_th)
        results_2023.append(r)
    
    # Trio-Top3
    print("Trio-Top3...", flush=True)
    for prob_th in [0.0, 0.10, 0.15, 0.20, 0.25]:
        r = sim_trio_top3(df_2023, payouts['trio'], prob_th)
        results_2023.append(r)
    
    # Trifecta-Top3
    print("Trif-Top3...", flush=True)
    for prob_th in [0.0, 0.10, 0.15, 0.20, 0.25]:
        r = sim_trif_top3(df_2023, payouts['trif'], prob_th)
        results_2023.append(r)
    
    # Filter and sort
    results_2023 = [r for r in results_2023 if r['bets'] >= 50]
    results_2023.sort(key=lambda x: -x['roi'])
    
    print("\n" + "=" * 80)
    print("2023 Grid Search Results (Top 25)")
    print("=" * 80)
    print(f"{'Strategy':<14} {'Params':<28} {'Bets':>6} {'Hits':>5} {'Hit%':>7} {'ROI%':>7}")
    print("-" * 80)
    for r in results_2023[:25]:
        print(f"{r['strategy']:<14} {r['params']:<28} {r['bets']:>6} {r['hits']:>5} {r['hit_rate']:>6.1f}% {r['roi']:>6.1f}%")
    
    # Test top strategies on 2024
    print("\n" + "=" * 80)
    print("Testing Best Strategies on 2024...")
    print("=" * 80 + "\n")
    
    results_compare = []
    
    for r23 in results_2023[:15]:
        strat = r23['strategy']
        params = r23['params']
        
        # Run same strategy on 2024
        if strat == 'Win-Top1':
            parts = params.split(',')
            ev_th = float(parts[0].split('>=')[1])
            prob_th = float(parts[1].split('>=')[1])
            r24 = sim_win_top1(df_2024, win_odds, payouts['win'], ev_th, prob_th)
        elif strat == 'Place-Top1':
            prob_th = float(params.split('>=')[1])
            r24 = sim_place_top1(df_2024, payouts['place'], prob_th)
        elif strat == 'Wide-Top2':
            prob_th = float(params.split('>=')[1])
            r24 = sim_wide_top2(df_2024, payouts['wide'], prob_th)
        elif strat == 'Quin-Top2':
            parts = params.split(',')
            prob_th = float(parts[0].split('>=')[1])
            prob_sum_th = float(parts[1].split('>=')[1])
            r24 = sim_quin_top2(df_2024, payouts['quin'], prob_th, prob_sum_th)
        elif strat == 'Exacta-Top2':
            prob_th = float(params.split('>=')[1])
            r24 = sim_exacta_top2(df_2024, payouts['exacta'], prob_th)
        elif strat == 'Trio-Top3':
            prob_th = float(params.split('>=')[1])
            r24 = sim_trio_top3(df_2024, payouts['trio'], prob_th)
        elif strat == 'Trif-Top3':
            prob_th = float(params.split('>=')[1])
            r24 = sim_trif_top3(df_2024, payouts['trif'], prob_th)
        else:
            continue
        
        results_compare.append({
            'strategy': strat,
            'params': params,
            '23_bets': r23['bets'],
            '23_roi': r23['roi'],
            '24_bets': r24['bets'],
            '24_roi': r24['roi'],
            'avg_roi': (r23['roi'] + r24['roi']) / 2
        })
    
    results_compare.sort(key=lambda x: -x['avg_roi'])
    
    print(f"{'Strategy':<14} {'Params':<28} {'23 Bets':>7} {'23 ROI':>7} {'24 Bets':>7} {'24 ROI':>7} {'Avg':>7}")
    print("-" * 90)
    for r in results_compare:
        print(f"{r['strategy']:<14} {r['params']:<28} {r['23_bets']:>7} {r['23_roi']:>6.1f}% {r['24_bets']:>7} {r['24_roi']:>6.1f}% {r['avg_roi']:>6.1f}%")
    
    # Best overall
    print("\n" + "=" * 80)
    print("🏆 Best Strategies (by Average ROI)")
    print("=" * 80)
    for r in results_compare[:5]:
        print(f"\n{r['strategy']} with {r['params']}")
        print(f"  2023: ROI={r['23_roi']:.1f}%, Bets={r['23_bets']}")
        print(f"  2024: ROI={r['24_roi']:.1f}%, Bets={r['24_bets']}")
        print(f"  Average: {r['avg_roi']:.1f}%")


if __name__ == "__main__":
    main()
