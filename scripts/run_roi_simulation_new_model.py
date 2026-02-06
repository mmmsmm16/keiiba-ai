"""
ROI Simulation with New Optuna-Best Model (Leak-Free)
- Multiple bet types: Win, Place, Quinella
- Progress output for visibility
- Optimized for speed

Usage: python scripts/run_roi_simulation_new_model.py
"""
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"

# Leakage exclusions
LEAKAGE_COLS = [
    'pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank',
    'last_3f', 'raw_time', 'time_diff', 'margin',
    'time', 'popularity', 'odds', 'relative_popularity_rank',
    'slow_start_recovery', 'track_bias_disadvantage',
    'outer_frame_disadv', 'wide_run',
    'mean_time_diff_5', 'horse_wide_run_rate',
]


def get_db_engine():
    return create_engine(
        f"postgresql://postgres:postgres@host.docker.internal:5433/pckeiba"
    )


def load_payout_data(engine, years):
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        haraimodoshi_umaren_1a as quinella_horses, haraimodoshi_umaren_1b as quinella_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    """Parse payout data into dictionaries"""
    win_payouts = {}  # {race_id: {horse_num: payout}}
    place_payouts = {}
    quinella_payouts = {}
    
    for _, row in df_pay.iterrows():
        rid = str(row['race_id'])
        
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
            combo_str = str(row['quinella_horses'])
            if len(combo_str) == 4:
                h1, h2 = int(combo_str[:2]), int(combo_str[2:])
                p = int(row['quinella_pay'])
                quinella_payouts[rid] = {(min(h1,h2), max(h1,h2)): p}
        except:
            pass
    
    return win_payouts, place_payouts, quinella_payouts


def generate_predictions(df, model, feature_cols):
    """Generate predictions for a dataframe"""
    X = df[feature_cols].copy()
    
    for col in X.columns:
        if X[col].dtype.name == 'category' or X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
        else:
            X[col] = X[col].fillna(-999)
    
    # Cast to float64 to ensure compatibility
    X_vals = X.values.astype(np.float64)
    preds = model.predict(X_vals)
    
    result = df[['race_id', 'horse_number', 'date']].copy()
    result['pred_prob'] = preds
    
    # Normalize Probabilities per race (CRITICAL for EV accuracy)
    result['pred_prob_norm'] = result.groupby('race_id')['pred_prob'].transform(lambda x: x / x.sum())
    
    # Get actual odds from original df
    if 'odds_final' in df.columns:
        result['odds'] = df['odds_final'].values
        result['odds'] = result['odds'].fillna(10.0)
    elif 'odds' in df.columns:
        result['odds'] = df['odds'].values
        result['odds'] = result['odds'].fillna(10.0)
    else:
        result['odds'] = 10.0
        
    return result

# ... (simulate_win_strategy, simulate_place_strategy, simulate_quinella_strategy remain same)

def simulate_ev_win_strategy(df, win_payouts, ev_th=1.5, prob_th=0.10):
    """EV-based Win strategy simulation (EV = ProbNorm * Odds)"""
    df = df.copy()
    df['race_id'] = df['race_id'].astype(str)
    
    # Calculate EV
    df['ev'] = df['pred_prob_norm'] * df['odds']
    
    # Filter
    mask = (df['ev'] >= ev_th) & (df['pred_prob_norm'] >= prob_th)
    bets = df[mask].copy()
    
    # Calculate actual payouts
    bets['payout'] = 0
    for idx, row in bets.iterrows():
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        if rid in win_payouts and hn in win_payouts[rid]:
            bets.loc[idx, 'payout'] = win_payouts[rid][hn]
    
    total_bet = len(bets) * 100
    total_payout = bets['payout'].sum()
    hits = (bets['payout'] > 0).sum()
    
    return {
        'bet_type': 'Win(EV)',
        'bets': len(bets),
        'hits': int(hits),
        'hit_rate': hits / len(bets) * 100 if len(bets) > 0 else 0,
        'roi': total_payout / total_bet * 100 if total_bet > 0 else 0,
        'params': f"EV>{ev_th:.1f}, p>{prob_th:.2f}"
    }

def simulate_top1_ev_win_strategy(df, win_payouts, ev_th=1.5, prob_th=0.10):
    """
    EV-based Top 1 Strategy
    Only bets on the horse with the highest EV in the race,
    provided it meets the EV and Probability thresholds.
    """
    df = df.copy()
    df['race_id'] = df['race_id'].astype(str)
    
    # Calculate EV
    df['ev'] = df['pred_prob_norm'] * df['odds']
    
    # Sort by EV descending within race
    df = df.sort_values(['race_id', 'ev'], ascending=[True, False])
    
    # Rank by EV
    df['ev_rank'] = df.groupby('race_id').cumcount() + 1
    
    # Filter: Rank 1 AND Thresholds
    mask = (df['ev_rank'] == 1) & (df['ev'] >= ev_th) & (df['pred_prob_norm'] >= prob_th)
    bets = df[mask].copy()
    
    # Calculate actual payouts
    bets['payout'] = 0
    for idx, row in bets.iterrows():
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        if rid in win_payouts and hn in win_payouts[rid]:
            bets.loc[idx, 'payout'] = win_payouts[rid][hn]
    
    total_bet = len(bets) * 100
    total_payout = bets['payout'].sum()
    hits = (bets['payout'] > 0).sum()
    
    return {
        'bet_type': 'Win(TopEV)',
        'bets': len(bets),
        'hits': int(hits),
        'hit_rate': hits / len(bets) * 100 if len(bets) > 0 else 0,
        'roi': total_payout / total_bet * 100 if total_bet > 0 else 0,
        'params': f"EV>{ev_th:.1f}, p>{prob_th:.2f}"
    }

def simulate_top1_prob_strategy(df, win_payouts):
    """
    Top 1 Probability Strategy
    Bets on the horse with the highest normalized probability in each race, 
    no matter what. (Flat buy)
    """
    df = df.copy()
    df['race_id'] = df['race_id'].astype(str)
    
    # Sort by normalized probability descending within race
    df = df.sort_values(['race_id', 'pred_prob_norm'], ascending=[True, False])
    
    # Rank by Probability
    df['prob_rank'] = df.groupby('race_id').cumcount() + 1
    
    # Filter: Rank 1 only
    mask = (df['prob_rank'] == 1)
    bets = df[mask].copy()
    
    # Calculate actual payouts
    bets['payout'] = 0
    for idx, row in bets.iterrows():
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        if rid in win_payouts and hn in win_payouts[rid]:
            bets.loc[idx, 'payout'] = win_payouts[rid][hn]
    
    total_bet = len(bets) * 100
    total_payout = bets['payout'].sum()
    hits = (bets['payout'] > 0).sum()
    
    return {
        'bet_type': 'Win(TopProb)',
        'bets': len(bets),
        'hits': int(hits),
        'hit_rate': hits / len(bets) * 100 if len(bets) > 0 else 0,
        'roi': total_payout / total_bet * 100 if total_bet > 0 else 0,
        'params': "Top1 Prob"
    }

def simulate_dynamic_betting_strategy(df, win_payouts, ev_th=1.2, prob_th=0.05, sizing_method='linear_ev', base_amount=100):
    """
    Dynamic Betting Strategy
    Varies bet amount based on EV or Probability.
    
    Methods:
    - linear_ev: amount = base * (EV - 1.0) * 10 (e.g. EV 1.5 -> 500 yen)
    - linear_prob: amount = base * prob * 100 (e.g. Prob 0.3 -> 3000 yen?) -> let's say base=10000, amt = 10000 * prob
    - kelly: amount = base * (prob * odds - 1) / (odds - 1) 
    """
    df = df.copy()
    df['race_id'] = df['race_id'].astype(str)
    
    # Calculate EV
    df['ev'] = df['pred_prob_norm'] * df['odds']
    
    # Filter first (only bet on positive EV or specific threshold)
    mask = (df['ev'] >= ev_th) & (df['pred_prob_norm'] >= prob_th)
    bets = df[mask].copy()
    
    if len(bets) == 0:
        return {
            'bet_type': f"Win({sizing_method})",
            'bets': 0, 'hits': 0, 'hit_rate': 0, 'roi': 0,
            'params': f"EV>{ev_th}, {sizing_method}"
        }

    # Calculate Bet Amounts
    if sizing_method == 'linear_ev':
        # EV 1.2 -> 200 yen, EV 2.0 -> 1000 yen ?
        # Formula: 100 * (EV) ? or (EV-1)?
        # Let's try: Amount = 100 * (EV)  (Higher EV = Higher Bet)
        bets['bet_amount'] = (bets['ev'] * base_amount).astype(int)
        
    elif sizing_method == 'linear_prob':
        # Prob 0.1 -> 1000 yen, Prob 0.5 -> 5000 yen
        # Amount = Prob * 10000
        bets['bet_amount'] = (bets['pred_prob_norm'] * 10000).astype(int)
        
    elif sizing_method == 'kelly':
        # Full Kelly is too risky, use fractional (e.g. 0.1 Kelly)
        # Fraction = (b*p - q) / b  where b = odds-1
        b = bets['odds'] - 1
        p = bets['pred_prob_norm']
        q = 1 - p
        f = (b * p - q) / b
        # Clip negative
        f = f.clip(lower=0)
        # Bankroll 10000 per race?
        bets['bet_amount'] = (f * 10000 * 0.1).astype(int) # 10% Kelly
        
    # Ensure min bet 100
    bets['bet_amount'] = bets['bet_amount'].clip(lower=100)
    # Round to 100
    bets['bet_amount'] = (bets['bet_amount'] // 100) * 100
    
    # Calculate payouts
    bets['payout'] = 0
    for idx, row in bets.iterrows():
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        amt = int(row['bet_amount'])
        
        if rid in win_payouts and hn in win_payouts[rid]:
            # Payout is per 100 yen in JVD data
            unit_payout = win_payouts[rid][hn]
            bets.loc[idx, 'payout'] = (unit_payout / 100) * amt
            
    total_bet = bets['bet_amount'].sum()
    total_payout = bets['payout'].sum()
    hits = (bets['payout'] > 0).sum()
    
    return {
        'bet_type': f"Win({sizing_method})",
        'bets': len(bets),
        'hits': int(hits),
        'hit_rate': hits / len(bets) * 100 if len(bets) > 0 else 0,
        'roi': total_payout / total_bet * 100 if total_bet > 0 else 0,
        'params': f"EV>{ev_th}, {sizing_method}"
    }

def main():
    print("=" * 80)
    print("🎯 ROI Simulation with T2 Refined v3 (Corrected EV)")
    print("=" * 80)
    
    # ... (Loading model/data/payouts code)
    print("Loading model...", flush=True)
    model = joblib.load(MODEL_PATH)
    
    print("Loading data...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Prepare features
    # Use model's expected features
    print("Aligning features...", flush=True)
    required_features = model.feature_name()
    print(f"Model expects {len(required_features)} features.", flush=True)
    
    # Check for missing
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        print(f"WARNING: Missing {len(missing)} features: {missing}", flush=True)
        # Fill missing with 0 or NaN?
        for c in missing:
            df[c] = np.nan
            
    feature_cols = required_features
    
    print(f"Features: {len(feature_cols)}", flush=True)
    
    df_2023 = df[df['year'] == 2023].copy()
    df_2024 = df[df['year'] == 2024].copy()
    
    print(f"2023: {len(df_2023)} rows, 2024: {len(df_2024)} rows", flush=True)

    print("\nGenerating predictions...", flush=True)
    preds_2023 = generate_predictions(df_2023, model, feature_cols)
    preds_2024 = generate_predictions(df_2024, model, feature_cols)
    print("Predictions done!", flush=True)

    print("\nLoading payouts...", flush=True)
    engine = get_db_engine()
    df_pay_2023 = load_payout_data(engine, [2023])
    df_pay_2024 = load_payout_data(engine, [2024])
    win_2023, _, _ = parse_payouts(df_pay_2023)
    win_2024, _, _ = parse_payouts(df_pay_2024)
    print("Payouts loaded.", flush=True)

    results = []
    
    # EV Strategy Grid
    print("\n🎰 Win EV Strategy Grid Search...", flush=True)
    ev_params = [
        (1.0, 0.05), (1.2, 0.05), (1.5, 0.05), (1.8, 0.05), (2.0, 0.05),
        (1.0, 0.10), (1.2, 0.10), (1.5, 0.10), (1.8, 0.10), (2.0, 0.10),
        (1.5, 0.15),
    ]
    
    for ev_th, prob_th in tqdm(ev_params, desc="EV"):
        # Multi Bet
        r23 = simulate_ev_win_strategy(preds_2023, win_2023, ev_th, prob_th)
        r24 = simulate_ev_win_strategy(preds_2024, win_2024, ev_th, prob_th)
        results.append({'year': 2023, **r23})
        results.append({'year': 2024, **r24})

        # Top 1 Bet
        r23_top = simulate_top1_ev_win_strategy(preds_2023, win_2023, ev_th, prob_th)
        r24_top = simulate_top1_ev_win_strategy(preds_2024, win_2024, ev_th, prob_th)
        results.append({'year': 2023, **r23_top})
        results.append({'year': 2024, **r24_top})

    
    # Dynamic Sizing Strategy (Linear EV)
    print("🎰 Dynamic Sizing Strategy (Linear EV)...", flush=True)
    for ev_th, prob_th in tqdm(ev_params, desc="Dynamic EV"):
        r23 = simulate_dynamic_betting_strategy(preds_2023, win_2023, ev_th, prob_th, 'linear_ev')
        r24 = simulate_dynamic_betting_strategy(preds_2024, win_2024, ev_th, prob_th, 'linear_ev')
        results.append({'year': 2023, **r23})
        results.append({'year': 2024, **r24})

    # Linear Prob and Kelly (Just a few thresholds)
    print("🎰 Dynamic Sizing Strategy (Prob / Kelly)...", flush=True)
    dyn_params = [(1.0, 0.05), (1.2, 0.05), (1.5, 0.10)]
    for ev_th, prob_th in dyn_params:
        # Linear Prob
        r23 = simulate_dynamic_betting_strategy(preds_2023, win_2023, ev_th, prob_th, 'linear_prob')
        r24 = simulate_dynamic_betting_strategy(preds_2024, win_2024, ev_th, prob_th, 'linear_prob')
        results.append({'year': 2023, **r23})
        results.append({'year': 2024, **r24})
        
        # Kelly
        r23k = simulate_dynamic_betting_strategy(preds_2023, win_2023, ev_th, prob_th, 'kelly')
        r24k = simulate_dynamic_betting_strategy(preds_2024, win_2024, ev_th, prob_th, 'kelly')
        results.append({'year': 2023, **r23k})
        results.append({'year': 2024, **r24k})

    # ... (Print results)
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results[df_results['bets'] >= 30]
    
    for year in [2023, 2024]:
        print(f"\n📅 {year} Results (sorted by ROI):")
        print("-" * 70)
        year_results = df_results[df_results['year'] == year].sort_values('roi', ascending=False)
        print(f"{'BetType':<10} {'Params':<35} {'Bets':>5} {'Hits':>5} {'Hit%':>6} {'ROI%':>7}")
        print("-" * 70)
        for _, r in year_results.head(15).iterrows():
            print(f"{r['bet_type']:<10} {r['params']:<35} {r['bets']:>5} {r['hits']:>5} {r['hit_rate']:>5.1f}% {r['roi']:>6.1f}%")

    print("\nDone.")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
