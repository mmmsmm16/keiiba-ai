"""
Multi-Bet ROI Simulation (Model-Specific)
=========================================
Each model is tested with the bet type that matches its strength:
- Win Model → 単勝 (Win bet on top-1 predicted horse)
- Top2 Model → 馬連 (Quinella on top-2 predicted horses)
- Top3 Model → 複勝/ワイド (Place on top-1, Wide BOX on top-3)

No EV or probability thresholds - pure predictive power evaluation.

Usage:
  python scripts/simulate_optimal_bets.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from sqlalchemy import create_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
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
    """Load all payout types"""
    year_list = ','.join([f"'{y}'" for y in years])
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay,
        haraimodoshi_fukusho_1a as place_1_horse, haraimodoshi_fukusho_1b as place_1_pay,
        haraimodoshi_fukusho_2a as place_2_horse, haraimodoshi_fukusho_2b as place_2_pay,
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        haraimodoshi_umaren_1a as quinella_horses, haraimodoshi_umaren_1b as quinella_pay,
        haraimodoshi_wide_1a as wide_1_horses, haraimodoshi_wide_1b as wide_1_pay,
        haraimodoshi_wide_2a as wide_2_horses, haraimodoshi_wide_2b as wide_2_pay,
        haraimodoshi_wide_3a as wide_3_horses, haraimodoshi_wide_3b as wide_3_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    """Parse payout data into comprehensive dict"""
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
        
        # Win
        try:
            h = int(row['win_horse'])
            p = int(row['win_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        
        # Place (top 3 finishers get payouts)
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        
        # Quinella (馬連)
        try:
            horses_str = str(row['quinella_horses']).zfill(4)
            h1, h2 = int(horses_str[:2]), int(horses_str[2:])
            p = int(row['quinella_pay'])
            if p > 0:
                pay['quinella'][(min(h1,h2), max(h1,h2))] = p
        except: pass
        
        # Wide (ワイド - up to 3 combinations)
        for i in [1, 2, 3]:
            try:
                horses_str = str(row[f'wide_{i}_horses']).zfill(4)
                h1, h2 = int(horses_str[:2]), int(horses_str[2:])
                p = int(row[f'wide_{i}_pay'])
                if p > 0:
                    pay['wide'][(min(h1,h2), max(h1,h2))] = p
            except: pass
        
        payout_dict[rid] = pay
    return payout_dict


def load_test_data():
    """Load 2024 test data"""
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
    """Generate predictions"""
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


def simulate_win(df_test, preds, payout_dict):
    """Win bet: Bet on top-1 predicted horse"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        
        top_horse = int(grp.iloc[0]['horse_number'])
        
        cost += 100
        bets += 1
        
        if top_horse in pay['win']:
            returns += pay['win'][top_horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_place(df_test, preds, payout_dict):
    """Place bet: Bet on top-1 predicted horse"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        
        top_horse = int(grp.iloc[0]['horse_number'])
        
        cost += 100
        bets += 1
        
        if top_horse in pay['place']:
            returns += pay['place'][top_horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_quinella(df_test, preds, payout_dict):
    """Quinella bet: Bet on top-2 predicted horses combination"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        
        if len(grp) < 2:
            continue
        
        h1 = int(grp.iloc[0]['horse_number'])
        h2 = int(grp.iloc[1]['horse_number'])
        key = (min(h1, h2), max(h1, h2))
        
        cost += 100
        bets += 1
        
        if key in pay['quinella']:
            returns += pay['quinella'][key]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_wide_box(df_test, preds, payout_dict):
    """Wide BOX bet: Buy all combinations of top-3 predicted horses"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        
        if len(grp) < 3:
            continue
        
        top_horses = [int(grp.iloc[i]['horse_number']) for i in range(3)]
        
        # 3 combinations: (1,2), (1,3), (2,3)
        for h1, h2 in combinations(top_horses, 2):
            key = (min(h1, h2), max(h1, h2))
            cost += 100
            bets += 1
            
            if key in pay['wide']:
                returns += pay['wide'][key]
                hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def main():
    logger.info("=" * 70)
    logger.info("Multi-Bet ROI Simulation (No Filters)")
    logger.info("=" * 70)
    
    # Load data
    df_test = load_test_data()
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2024])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Generate predictions
    predictions = {}
    for name, path in MODELS.items():
        logger.info(f"Predicting with {name} model...")
        preds = predict_with_model(path, df_test)
        if preds is not None:
            predictions[name] = preds
    
    # Create ensemble
    weights = {'Win': 0.7, 'Top2': 0.2, 'Top3': 0.1}
    ensemble_preds = (weights['Win'] * predictions['Win'] + 
                      weights['Top2'] * predictions['Top2'] + 
                      weights['Top3'] * predictions['Top3'])
    predictions['Ensemble'] = ensemble_preds
    
    # ==========================================
    # Part 1: Each model with its optimal bet type
    # ==========================================
    print("\n" + "=" * 70)
    print(" Part 1: Model-Specific Optimal Bet Types (No Filters)")
    print("=" * 70)
    
    results = {}
    
    # Win Model → 単勝
    r = simulate_win(df_test, predictions['Win'], payout_dict)
    roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
    hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
    results['Win→単勝'] = {'roi': roi, 'hit_rate': hit_rate, **r}
    
    # Top2 Model → 馬連
    r = simulate_quinella(df_test, predictions['Top2'], payout_dict)
    roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
    hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
    results['Top2→馬連'] = {'roi': roi, 'hit_rate': hit_rate, **r}
    
    # Top3 Model → 複勝
    r = simulate_place(df_test, predictions['Top3'], payout_dict)
    roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
    hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
    results['Top3→複勝'] = {'roi': roi, 'hit_rate': hit_rate, **r}
    
    # Top3 Model → ワイドBOX
    r = simulate_wide_box(df_test, predictions['Top3'], payout_dict)
    roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
    hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
    results['Top3→ワイドBOX'] = {'roi': roi, 'hit_rate': hit_rate, **r}
    
    print(f"\n{'Strategy':<20} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<20} | {r['bets']:<8} | {r['hits']:<8} | {r['hit_rate']:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {r['roi']:<9.1f}%")

    # ==========================================
    # Part 2: All models × All bet types (comparison matrix)
    # ==========================================
    print("\n" + "=" * 70)
    print(" Part 2: All Models × All Bet Types Matrix")
    print("=" * 70)
    
    bet_funcs = {
        '単勝': simulate_win,
        '複勝': simulate_place,
        '馬連': simulate_quinella,
        'ワイドBOX': simulate_wide_box,
    }
    
    print(f"\n{'Model':<12}", end="")
    for bet_name in bet_funcs.keys():
        print(f" | {bet_name:<12}", end="")
    print()
    print("-" * 70)
    
    for model_name in ['Win', 'Top2', 'Top3', 'Ensemble']:
        print(f"{model_name:<12}", end="")
        for bet_name, bet_func in bet_funcs.items():
            r = bet_func(df_test, predictions[model_name], payout_dict)
            roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
            print(f" | {roi:<11.1f}%", end="")
        print()
    
    # ==========================================
    # Part 3: Best combinations
    # ==========================================
    print("\n" + "=" * 70)
    print(" Part 3: Best Combinations (Sorted by ROI)")
    print("=" * 70)
    
    all_results = []
    for model_name in ['Win', 'Top2', 'Top3', 'Ensemble']:
        for bet_name, bet_func in bet_funcs.items():
            r = bet_func(df_test, predictions[model_name], payout_dict)
            roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
            hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
            all_results.append({
                'model': model_name,
                'bet': bet_name,
                'roi': roi,
                'hit_rate': hit_rate,
                'bets': r['bets'],
                'profit': r['returns'] - r['cost']
            })
    
    all_results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'Rank':<6} | {'Model':<12} | {'Bet':<12} | {'ROI':<10} | {'HitRate':<10} | {'Profit':<12}")
    print("-" * 70)
    for i, r in enumerate(all_results[:10], 1):
        sign = '+' if r['profit'] >= 0 else ''
        print(f"{i:<6} | {r['model']:<12} | {r['bet']:<12} | {r['roi']:<9.1f}% | {r['hit_rate']:<9.1f}% | {sign}¥{r['profit']:,}")


if __name__ == "__main__":
    main()
