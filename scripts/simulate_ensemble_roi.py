"""
Ensemble ROI Simulation
=======================
Compare ROI of single models vs ensemble on 2024 test set.
Simulates Win bets with various EV thresholds.

Usage:
  python scripts/simulate_ensemble_roi.py
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

# Paths
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
MODELS = {
    "Win": "models/experiments/exp_t2_refined_v3/model.pkl",
    "Top2": "models/experiments/exp_t2_refined_v3_top2/model.pkl",
    "Top3": "models/experiments/exp_t2_refined_v3_top3/model.pkl",
}

# Ensemble weights from grid search
ENSEMBLE_WEIGHTS = {
    'win': {'Win': 0.7, 'Top2': 0.2, 'Top3': 0.1},
    'top2': {'Win': 0.4, 'Top2': 0.4, 'Top3': 0.2},
    'top3': {'Win': 0.3, 'Top2': 0.3, 'Top3': 0.4},
}


def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "pckeiba")
    return create_engine(f'postgresql://{user}:{pw}@{host}:{port}/{db}')


def load_payout_data(engine, years):
    """Load win/place payout data"""
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
    """Parse payout data into dict"""
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}}
        try:
            h = int(row['win_horse'])
            p = int(row['win_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        payout_dict[rid] = pay
    return payout_dict


def load_test_data():
    """Load 2024 test data"""
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['race_id'] = df['race_id'].astype(str)
    
    # Load targets for rank
    targets = pd.read_parquet(TARGET_PATH)
    targets['race_id'] = targets['race_id'].astype(str)
    
    if 'rank' not in df.columns:
        df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                      on=['race_id', 'horse_number'], how='left')
    
    # Filter to 2024
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


def simulate_roi(df_test, preds, payout_dict, ev_threshold=1.5, prob_threshold=0.10, bet_type='win'):
    """Simulate ROI for given predictions"""
    df = df_test.copy()
    df['pred'] = preds
    
    # Normalize predictions per race
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        
        # Get top prediction
        top_row = grp.iloc[0]
        pred_prob = top_row['pred_norm']
        horse_num = int(top_row['horse_number'])
        
        # Use odds_final from dataframe (pre-race odds)
        if 'odds_final' in top_row and pd.notna(top_row['odds_final']) and top_row['odds_final'] > 0:
            odds = top_row['odds_final']
        else:
            continue  # Skip if no odds available
        
        # Calculate EV
        ev = pred_prob * odds
        
        # Bet decision
        if top_row['pred'] >= prob_threshold and ev >= ev_threshold:
            cost += 100
            bets += 1
            
            if bet_type == 'win':
                if horse_num in pay['win']:
                    returns += pay['win'][horse_num]
                    hits += 1
            elif bet_type == 'place':
                if horse_num in pay['place']:
                    returns += pay['place'][horse_num]
                    hits += 1
    
    roi = (returns / cost * 100) if cost > 0 else 0
    hit_rate = (hits / bets * 100) if bets > 0 else 0
    
    return {
        'roi': roi,
        'bets': bets,
        'hits': hits,
        'hit_rate': hit_rate,
        'cost': cost,
        'returns': returns
    }


def main():
    logger.info("=" * 60)
    logger.info("Ensemble ROI Simulation")
    logger.info("=" * 60)
    
    # Load data
    df_test = load_test_data()
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2024])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Generate predictions for each model
    predictions = {}
    for name, path in MODELS.items():
        logger.info(f"Predicting with {name} model...")
        preds = predict_with_model(path, df_test)
        if preds is not None:
            predictions[name] = preds
    
    if len(predictions) != 3:
        logger.error("Failed to load all models")
        return
    
    # Create ensemble predictions
    weights = ENSEMBLE_WEIGHTS['win']
    ensemble_preds = (weights['Win'] * predictions['Win'] + 
                      weights['Top2'] * predictions['Top2'] + 
                      weights['Top3'] * predictions['Top3'])
    predictions['Ensemble'] = ensemble_preds
    
    # Simulate with different EV thresholds
    ev_thresholds = [1.0, 1.2, 1.5, 1.8, 2.0]
    
    print("\n" + "=" * 90)
    print(" Win Bet ROI Simulation (2024 Test Set)")
    print("=" * 90)
    
    for ev_th in ev_thresholds:
        print(f"\n--- EV Threshold: {ev_th} ---")
        print(f"{'Model':<12} | {'Bets':<8} | {'Hits':<8} | {'Hit%':<8} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
        print("-" * 90)
        
        for name in ['Win', 'Top2', 'Top3', 'Ensemble']:
            result = simulate_roi(df_test, predictions[name], payout_dict, 
                                  ev_threshold=ev_th, prob_threshold=0.10)
            print(f"{name:<12} | {result['bets']:<8} | {result['hits']:<8} | {result['hit_rate']:<7.1f}% | ¥{result['cost']:<10,} | ¥{result['returns']:<10,} | {result['roi']:<9.1f}%")
    
    # Summary
    print("\n" + "=" * 90)
    print(" Best Configuration Summary")
    print("=" * 90)
    
    # Find best ROI for each model
    best_results = {}
    for name in ['Win', 'Top2', 'Top3', 'Ensemble']:
        best_roi = 0
        best_ev = 0
        for ev_th in ev_thresholds:
            result = simulate_roi(df_test, predictions[name], payout_dict, 
                                  ev_threshold=ev_th, prob_threshold=0.10)
            if result['roi'] > best_roi and result['bets'] >= 50:  # Minimum 50 bets
                best_roi = result['roi']
                best_ev = ev_th
        best_results[name] = {'ev': best_ev, 'roi': best_roi}
    
    print(f"\n{'Model':<12} | {'Best EV':<10} | {'Best ROI':<10}")
    print("-" * 40)
    for name, res in best_results.items():
        print(f"{name:<12} | {res['ev']:<10} | {res['roi']:<9.1f}%")


if __name__ == "__main__":
    main()
