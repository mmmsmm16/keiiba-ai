"""
ROI Simulation with Calibration + EV Filtering (10-min Pre-Race Odds)

Features:
- Isotonic calibration of predicted probabilities
- EV filtering using 10-min-before odds from jvd_o1
- Actual payouts from jvd_hr for ROI calculation

Usage: python scripts/run_roi_simulation_calibrated.py
"""
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/optuna_best_full/model.pkl"

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


def load_10min_odds(engine, years):
    """Load 10-minute-before odds from jvd_o1"""
    year_list = ','.join([f"'{y}'" for y in years])
    
    # Build race_id and get odds before deadline (10 min before race)
    query = f"""
    WITH race_info AS (
        SELECT DISTINCT
            kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
            kaisai_nen,
            kaisai_tsukihi
        FROM jvd_ra
        WHERE kaisai_nen IN ({year_list})
    ),
    odds_with_ts AS (
        SELECT 
            o.kaisai_nen || o.keibajo_code || o.kaisai_kai || o.kaisai_nichime || o.race_bango as race_id,
            o.happyo_tsukihi_jifun,
            o.odds_tansho
        FROM jvd_o1 o
        WHERE o.kaisai_nen IN ({year_list})
    )
    SELECT 
        race_id,
        odds_tansho
    FROM (
        SELECT 
            race_id,
            odds_tansho,
            ROW_NUMBER() OVER (PARTITION BY race_id ORDER BY happyo_tsukihi_jifun DESC) as rn
        FROM odds_with_ts
    ) sub
    WHERE rn = 1
    """
    
    # Simpler query - just get latest odds per race
    simple_query = f"""
    WITH latest_odds AS (
        SELECT 
            kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
            odds_tansho,
            ROW_NUMBER() OVER (
                PARTITION BY kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango 
                ORDER BY happyo_tsukihi_jifun DESC
            ) as rn
        FROM jvd_o1
        WHERE kaisai_nen IN ({year_list})
    )
    SELECT race_id, odds_tansho
    FROM latest_odds
    WHERE rn = 1
    """
    
    try:
        df = pd.read_sql(simple_query, engine)
        df['race_id'] = df['race_id'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading odds: {e}")
        return pd.DataFrame()


def parse_odds_array(odds_str):
    """Parse odds_tansho string into dictionary {horse_num: odds}
    
    Format: 8 chars per horse
    - bytes 0-1: horse number (e.g., "01", "02")
    - bytes 2-5: odds x10 (e.g., "0036" = 3.6倍)
    - bytes 6-7: popularity (ignored for our purpose)
    """
    result = {}
    if not odds_str or pd.isna(odds_str):
        return result
    
    try:
        s = str(odds_str)
        chunk_size = 8
        for i in range(0, len(s), chunk_size):
            block = s[i:i+chunk_size]
            if len(block) < chunk_size:
                break
            try:
                horse_num = int(block[0:2])
                odds_val = int(block[2:6]) / 10.0
                if horse_num > 0 and odds_val > 0:
                    result[horse_num] = odds_val
            except:
                pass
    except:
        pass
    
    return result


def load_payout_data(engine, years):
    """Load confirmed payouts from jvd_hr"""
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    """Parse payout data into dictionaries"""
    win_payouts = {}
    place_payouts = {}
    
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
    
    return win_payouts, place_payouts


def calibrate_probabilities(y_train, probs_train, probs_test):
    """Apply isotonic calibration"""
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(probs_train, y_train)
    return iso_reg.predict(probs_test)


def generate_predictions(df, model, feature_cols):
    """Generate predictions for a dataframe"""
    X = df[feature_cols].copy()
    
    for col in X.columns:
        if X[col].dtype.name == 'category' or X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
        else:
            X[col] = X[col].fillna(-999)
    
    preds = model.predict(X)
    
    result = df[['race_id', 'horse_number', 'date', 'rank']].copy()
    result['pred_prob'] = preds
    
    return result


def simulate_win_ev_strategy(df, win_payouts, odds_dict, ev_th=1.0, prob_th=0.10, gap_th=0.03):
    """Win strategy with EV filtering using pre-race odds"""
    df = df.copy()
    df['race_id'] = df['race_id'].astype(str)
    
    # Add odds
    def get_odds(row):
        rid = str(row['race_id'])
        hn = int(row['horse_number'])
        if rid in odds_dict:
            race_odds = odds_dict[rid]
            return race_odds.get(hn, 10.0)
        return 10.0
    
    df['odds_10min'] = df.apply(get_odds, axis=1)
    
    # Sort and rank
    df = df.sort_values(['race_id', 'pred_prob'], ascending=[True, False])
    df['race_rank'] = df.groupby('race_id').cumcount() + 1
    
    # Get top2
    top1 = df[df['race_rank'] == 1].set_index('race_id')
    top2 = df[df['race_rank'] == 2].set_index('race_id')
    
    top1['top2_prob'] = top2['pred_prob']
    top1['gap'] = top1['pred_prob'] - top1['top2_prob']
    top1['ev'] = top1['pred_prob'] * top1['odds_10min']
    
    # Apply filters
    mask = (top1['pred_prob'] >= prob_th) & (top1['gap'] >= gap_th) & (top1['ev'] >= ev_th)
    bets = top1[mask].copy()
    
    # Calculate actual payouts
    bets['payout'] = 0
    for rid, row in bets.iterrows():
        hn = int(row['horse_number'])
        if rid in win_payouts and hn in win_payouts[rid]:
            bets.loc[rid, 'payout'] = win_payouts[rid][hn]
    
    total_bet = len(bets) * 100
    total_payout = bets['payout'].sum()
    hits = (bets['payout'] > 0).sum()
    
    return {
        'bet_type': 'Win-EV',
        'bets': len(bets),
        'hits': int(hits),
        'hit_rate': hits / len(bets) * 100 if len(bets) > 0 else 0,
        'roi': total_payout / total_bet * 100 if total_bet > 0 else 0,
        'params': f"ev>{ev_th:.1f},p>{prob_th:.2f},gap>{gap_th:.2f}"
    }


def main():
    print("=" * 80)
    print("🎯 Calibrated ROI Simulation with 10-min Pre-Race Odds EV Filtering")
    print("=" * 80)
    
    # Load model
    print("Loading model...", flush=True)
    model = joblib.load(MODEL_PATH)
    
    # Load data
    print("Loading data...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Prepare features
    meta_cols = [
        'race_id', 'horse_number', 'date', 'rank', 'odds_final', 
        'is_win', 'is_top2', 'is_top3', 'year', 'rank_str'
    ]
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    exclude_all = meta_cols + LEAKAGE_COLS + id_cols
    feature_cols = [c for c in df.columns if c not in exclude_all]
    
    print(f"Features: {len(feature_cols)}", flush=True)
    
    # Split data
    df_train = df[(df['year'] >= 2019) & (df['year'] <= 2022)].copy()
    df_2023 = df[df['year'] == 2023].copy()
    df_2024 = df[df['year'] == 2024].copy()
    
    print(f"Train: {len(df_train)}, 2023: {len(df_2023)}, 2024: {len(df_2024)}", flush=True)
    
    # Generate predictions
    print("\nGenerating predictions...", flush=True)
    preds_train = generate_predictions(df_train, model, feature_cols)
    preds_2023 = generate_predictions(df_2023, model, feature_cols)
    preds_2024 = generate_predictions(df_2024, model, feature_cols)
    
    y_train = (df_train['rank'] == 1).astype(int).values
    
    # Calibration
    print("Calibrating probabilities...", flush=True)
    preds_2023['pred_prob_cal'] = calibrate_probabilities(
        y_train, preds_train['pred_prob'].values, preds_2023['pred_prob'].values
    )
    preds_2024['pred_prob_cal'] = calibrate_probabilities(
        y_train, preds_train['pred_prob'].values, preds_2024['pred_prob'].values
    )
    
    # Check calibration
    print("\nCalibration check:")
    for name, preds in [('2023', preds_2023), ('2024', preds_2024)]:
        y_true = (preds['rank'] == 1).astype(int)
        prob_true, prob_pred = calibration_curve(y_true, preds['pred_prob_cal'], n_bins=10)
        print(f"  {name}: Mean predicted={preds['pred_prob_cal'].mean():.4f}, "
              f"Actual win rate={y_true.mean():.4f}")
    
    # Replace pred_prob with calibrated version
    preds_2023['pred_prob'] = preds_2023['pred_prob_cal']
    preds_2024['pred_prob'] = preds_2024['pred_prob_cal']
    
    # Load odds
    print("\nLoading 10-min pre-race odds...", flush=True)
    engine = get_db_engine()
    
    df_odds_2023 = load_10min_odds(engine, [2023])
    df_odds_2024 = load_10min_odds(engine, [2024])
    
    print(f"Loaded odds: 2023={len(df_odds_2023)}, 2024={len(df_odds_2024)}", flush=True)
    
    # Parse odds into dictionaries
    odds_dict_2023 = {}
    odds_dict_2024 = {}
    
    for _, row in df_odds_2023.iterrows():
        rid = row['race_id']
        odds_dict_2023[rid] = parse_odds_array(row['odds_tansho'])
    
    for _, row in df_odds_2024.iterrows():
        rid = row['race_id']
        odds_dict_2024[rid] = parse_odds_array(row['odds_tansho'])
    
    print(f"Parsed odds: 2023={len(odds_dict_2023)}, 2024={len(odds_dict_2024)}", flush=True)
    
    # Load payouts
    print("\nLoading payouts...", flush=True)
    df_pay_2023 = load_payout_data(engine, [2023])
    df_pay_2024 = load_payout_data(engine, [2024])
    win_2023, place_2023 = parse_payouts(df_pay_2023)
    win_2024, place_2024 = parse_payouts(df_pay_2024)
    print(f"Payouts: 2023={len(win_2023)}, 2024={len(win_2024)}", flush=True)
    
    # Run simulations
    print("\n" + "=" * 80)
    print("Running EV Strategy Simulations (Calibrated)...")
    print("=" * 80 + "\n")
    
    results = []
    
    # EV Grid Search
    ev_params = [
        # (ev_th, prob_th, gap_th)
        (1.0, 0.08, 0.02), (1.0, 0.10, 0.03), (1.0, 0.12, 0.03),
        (1.1, 0.10, 0.03), (1.1, 0.12, 0.05), (1.1, 0.15, 0.05),
        (1.2, 0.10, 0.03), (1.2, 0.12, 0.05), (1.2, 0.15, 0.05),
        (1.3, 0.12, 0.05), (1.3, 0.15, 0.08), (1.3, 0.18, 0.08),
        (1.5, 0.15, 0.08), (1.5, 0.18, 0.10), (1.5, 0.20, 0.10),
        (1.8, 0.18, 0.10), (1.8, 0.20, 0.12), (2.0, 0.20, 0.12),
    ]
    
    print("🎰 Win-EV Strategy Grid Search...", flush=True)
    for ev, prob, gap in tqdm(ev_params, desc="Win-EV"):
        r23 = simulate_win_ev_strategy(preds_2023, win_2023, odds_dict_2023, ev, prob, gap)
        r24 = simulate_win_ev_strategy(preds_2024, win_2024, odds_dict_2024, ev, prob, gap)
        results.append({'year': 2023, **r23})
        results.append({'year': 2024, **r24})
    
    # Print results
    print("\n" + "=" * 80)
    print("Results Summary (Calibrated + EV Filtering)")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results[df_results['bets'] >= 30]
    
    for year in [2023, 2024]:
        print(f"\n📅 {year} Results (sorted by ROI):")
        print("-" * 75)
        year_results = df_results[df_results['year'] == year].sort_values('roi', ascending=False)
        
        print(f"{'BetType':<10} {'Params':<35} {'Bets':>6} {'Hits':>5} {'Hit%':>6} {'ROI%':>7}")
        print("-" * 75)
        
        for _, r in year_results.head(15).iterrows():
            print(f"{r['bet_type']:<10} {r['params']:<35} {r['bets']:>6} {r['hits']:>5} {r['hit_rate']:>5.1f}% {r['roi']:>6.1f}%")
    
    # Best strategies
    print("\n" + "=" * 80)
    print("🏆 Best Strategies")
    print("=" * 80)
    
    for year in [2023, 2024]:
        year_df = df_results[df_results['year'] == year]
        if len(year_df) > 0:
            best = year_df.sort_values('roi', ascending=False).iloc[0]
            print(f"\n{year} Best: {best['bet_type']} with {best['params']}")
            print(f"   ROI={best['roi']:.1f}%, Bets={best['bets']}, Hit={best['hit_rate']:.1f}%")


if __name__ == "__main__":
    main()
