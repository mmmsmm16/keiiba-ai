"""
EV and Combined Filtering Simulation
=====================================
Tests:
1. EV-only filtering
2. Combined Prob + EV filtering

Usage:
  python scripts/simulate_ev_filtered.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
MODELS = {
    "Win": "models/experiments/exp_t2_refined_v3/model.pkl",
    "Top2": "models/experiments/exp_t2_refined_v3_top2/model.pkl",
    "Top3": "models/experiments/exp_t2_refined_v3_top3/model.pkl",
}


def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "pckeiba")
    return create_engine(f'postgresql://{user}:{pw}@{host}:{port}/{db}')


def load_payout_data(engine, years):
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay
    FROM jvd_hr WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}}
        try:
            h, p = int(row['win_horse']), int(row['win_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        for i in [1, 2, 3]:
            try:
                h, p = int(row[f'place_{i}_horse']), int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        payout_dict[rid] = pay
    return payout_dict


def load_test_data():
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['race_id'] = df['race_id'].astype(str)
    
    targets = pd.read_parquet(TARGET_PATH)
    targets['race_id'] = targets['race_id'].astype(str)
    
    if 'rank' not in df.columns:
        df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                      on=['race_id', 'horse_number'], how='left')
    
    df_test = df[df['date'].dt.year == 2024].copy()
    logger.info(f"Test set (2024): {len(df_test)} records")
    return df_test


def predict_with_model(model_path, df_test):
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    feature_names = model.feature_name()
    X = df_test[feature_names].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
    X = X.astype(np.float64)
    return model.predict(X.values)


def simulate_win_ev(df_test, preds, payout_dict, ev_threshold=0.0, prob_threshold=0.0):
    """単勝: EV閾値 + 確率閾値でフィルタ"""
    df = df_test.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        prob = top['pred_norm']
        horse = int(top['horse_number'])
        
        # Get odds from dataframe
        if 'odds_final' in top and pd.notna(top['odds_final']) and top['odds_final'] > 0:
            odds = top['odds_final']
        else:
            continue
        
        ev = prob * odds
        
        # Apply filters
        if prob < prob_threshold: continue
        if ev < ev_threshold: continue
        
        cost += 100
        bets += 1
        if horse in pay['win']:
            returns += pay['win'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_place_ev(df_test, preds, payout_dict, ev_threshold=0.0, prob_threshold=0.0):
    """複勝: EV閾値 + 確率閾値でフィルタ"""
    df = df_test.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        prob = top['pred_norm']
        horse = int(top['horse_number'])
        
        if 'odds_final' in top and pd.notna(top['odds_final']) and top['odds_final'] > 0:
            odds = top['odds_final']
        else:
            continue
        
        ev = prob * odds
        
        if prob < prob_threshold: continue
        if ev < ev_threshold: continue
        
        cost += 100
        bets += 1
        if horse in pay['place']:
            returns += pay['place'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def main():
    logger.info("=" * 70)
    logger.info("EV Filtered Betting Simulation")
    logger.info("=" * 70)
    
    df_test = load_test_data()
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2024])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    predictions = {}
    for name, path in MODELS.items():
        logger.info(f"Predicting with {name} model...")
        preds = predict_with_model(path, df_test)
        if preds is not None:
            predictions[name] = preds
    
    ev_thresholds = [0.0, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
    prob_thresholds = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" Part 1: 単勝 - EV Only Filtering (Win Model)")
    print("=" * 90)
    print(f"{'EV閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    
    for ev_th in ev_thresholds:
        r = simulate_win_ev(df_test, predictions['Win'], payout_dict, ev_th, 0.0)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{ev_th:<10.1f} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" Part 2: 複勝 - EV Only Filtering (Top3 Model)")
    print("=" * 90)
    print(f"{'EV閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    
    for ev_th in ev_thresholds:
        r = simulate_place_ev(df_test, predictions['Top3'], payout_dict, ev_th, 0.0)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{ev_th:<10.1f} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" Part 3: 単勝 - Combined Prob + EV Filtering (Win Model)")
    print("=" * 90)
    print(f"{'Prob閾値':<10} | {'EV閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'ROI':<10}")
    print("-" * 70)
    
    for prob_th in prob_thresholds:
        for ev_th in [1.0, 1.2, 1.5]:
            r = simulate_win_ev(df_test, predictions['Win'], payout_dict, ev_th, prob_th)
            if r['bets'] < 10: continue
            roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
            hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
            print(f"{prob_th:<10.0%} | {ev_th:<10.1f} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" Part 4: 複勝 - Combined Prob + EV Filtering (Top3 Model)")
    print("=" * 90)
    print(f"{'Prob閾値':<10} | {'EV閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'ROI':<10}")
    print("-" * 70)
    
    for prob_th in prob_thresholds:
        for ev_th in [1.0, 1.2, 1.5]:
            r = simulate_place_ev(df_test, predictions['Top3'], payout_dict, ev_th, prob_th)
            if r['bets'] < 10: continue
            roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
            hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
            print(f"{prob_th:<10.0%} | {ev_th:<10.1f} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" Part 5: Grid Search - Best Configurations (Min 50 bets)")
    print("=" * 90)
    
    results = []
    
    for prob_th in prob_thresholds:
        for ev_th in ev_thresholds:
            # Win - 単勝
            r = simulate_win_ev(df_test, predictions['Win'], payout_dict, ev_th, prob_th)
            if r['bets'] >= 50:
                roi = r['returns'] / r['cost'] * 100
                results.append({
                    'bet': '単勝(Win)', 'prob': prob_th, 'ev': ev_th,
                    'roi': roi, 'bets': r['bets'], 'profit': r['returns'] - r['cost']
                })
            
            # Place - 複勝
            r = simulate_place_ev(df_test, predictions['Top3'], payout_dict, ev_th, prob_th)
            if r['bets'] >= 50:
                roi = r['returns'] / r['cost'] * 100
                results.append({
                    'bet': '複勝(Top3)', 'prob': prob_th, 'ev': ev_th,
                    'roi': roi, 'bets': r['bets'], 'profit': r['returns'] - r['cost']
                })
    
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'Rank':<6} | {'Bet Type':<15} | {'Prob':<8} | {'EV':<8} | {'ROI':<10} | {'Bets':<8} | {'Profit':<12}")
    print("-" * 80)
    for i, r in enumerate(results[:20], 1):
        sign = '+' if r['profit'] >= 0 else ''
        print(f"{i:<6} | {r['bet']:<15} | {r['prob']:<7.0%} | {r['ev']:<8.1f} | {r['roi']:<9.1f}% | {r['bets']:<8} | {sign}¥{r['profit']:,}")


if __name__ == "__main__":
    main()
