"""
Calibrated ROI Simulation
=========================
1. Train calibrator on 2023 validation set
2. Apply to 2024 test set
3. Run ROI simulation with calibrated probabilities

Usage:
  python scripts/simulate_calibrated_roi.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
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


def load_data():
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    targets['race_id'] = targets['race_id'].astype(str)
    df = df.merge(targets[['race_id', 'horse_number', 'rank']], on=['race_id', 'horse_number'], how='left')
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def predict_with_model(model_path, df):
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    feature_names = model.feature_name()
    X = df[feature_names].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
    return model.predict(X.values.astype(np.float64))


def train_calibrator(df_calib, preds, target_col):
    """Train isotonic regression calibrator"""
    df = df_calib.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum())
    df['target'] = (df['rank'] <= int(target_col[-1])).astype(int) if 'top' in target_col.lower() else (df['rank'] == 1).astype(int)
    
    # Train isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(df['pred_norm'].values, df['target'].values)
    
    return ir


def apply_calibration(df, preds, calibrator):
    """Apply calibration to predictions"""
    df = df.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum())
    df['pred_calib'] = calibrator.predict(df['pred_norm'].values)
    
    return df['pred_calib'].values


def simulate_win(df_test, preds, payout_dict, prob_threshold=0.0):
    """Win bet simulation"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        if top['pred'] < prob_threshold: continue
        
        horse = int(top['horse_number'])
        cost += 100
        bets += 1
        if horse in pay['win']:
            returns += pay['win'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_place(df_test, preds, payout_dict, prob_threshold=0.0):
    """Place bet simulation"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        if top['pred'] < prob_threshold: continue
        
        horse = int(top['horse_number'])
        cost += 100
        bets += 1
        if horse in pay['place']:
            returns += pay['place'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_ev(df_test, preds, payout_dict, ev_threshold=0.0, prob_threshold=0.0):
    """EV-filtered win bet simulation"""
    df = df_test.copy()
    df['pred'] = preds
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        prob = top['pred']
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
        if horse in pay['win']:
            returns += pay['win'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def main():
    logger.info("=" * 70)
    logger.info("Calibrated ROI Simulation")
    logger.info("=" * 70)
    
    # Load data
    df = load_data()
    df_calib = df[df['date'].dt.year == 2023].copy()  # Calibration set
    df_test = df[df['date'].dt.year == 2024].copy()   # Test set
    
    logger.info(f"Calibration set (2023): {len(df_calib)} records")
    logger.info(f"Test set (2024): {len(df_test)} records")
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2024])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # ========================================
    # Win Model - Calibration
    # ========================================
    logger.info("Processing Win model...")
    win_model = joblib.load(MODELS['Win'])
    
    # Predict on calibration and test sets
    win_preds_calib = predict_with_model(MODELS['Win'], df_calib)
    win_preds_test = predict_with_model(MODELS['Win'], df_test)
    
    # Train calibrator
    logger.info("Training Win calibrator on 2023 data...")
    win_calibrator = train_calibrator(df_calib, win_preds_calib, 'win')
    
    # Apply calibration
    win_calib_test = apply_calibration(df_test, win_preds_test, win_calibrator)
    
    # Normalize for comparison
    df_test_copy = df_test.copy()
    df_test_copy['win_raw'] = win_preds_test
    df_test_copy['win_raw_norm'] = df_test_copy.groupby('race_id')['win_raw'].transform(lambda x: x / x.sum())
    df_test_copy['win_calib'] = win_calib_test
    
    # ========================================
    # Top3 Model - Calibration
    # ========================================
    logger.info("Processing Top3 model...")
    top3_preds_calib = predict_with_model(MODELS['Top3'], df_calib)
    top3_preds_test = predict_with_model(MODELS['Top3'], df_test)
    
    logger.info("Training Top3 calibrator on 2023 data...")
    top3_calibrator = train_calibrator(df_calib, top3_preds_calib, 'top3')
    
    top3_calib_test = apply_calibration(df_test, top3_preds_test, top3_calibrator)
    
    df_test_copy['top3_raw'] = top3_preds_test
    df_test_copy['top3_raw_norm'] = df_test_copy.groupby('race_id')['top3_raw'].transform(lambda x: x / x.sum())
    df_test_copy['top3_calib'] = top3_calib_test
    
    # ========================================
    # Calibration Comparison
    # ========================================
    print("\n" + "=" * 80)
    print(" Calibration Verification (2024 Test Set)")
    print("=" * 80)
    
    # Check if calibration improved
    df_test_copy['is_win'] = (df_test_copy['rank'] == 1).astype(int)
    
    # Raw vs Calibrated comparison
    print("\nWin Model:")
    for col in ['win_raw_norm', 'win_calib']:
        df_test_copy['pred_bin'] = pd.cut(df_test_copy[col], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
        calib_check = df_test_copy.groupby('pred_bin', observed=True).agg(
            count=('is_win', 'count'),
            actual=('is_win', 'mean'),
            pred=('win_raw_norm' if 'raw' in col else 'win_calib', 'mean')
        )
        label = "Raw" if 'raw' in col else "Calibrated"
        print(f"\n  {label}:")
        for idx, row in calib_check.iterrows():
            gap = abs(row['pred'] - row['actual']) * 100
            print(f"    {idx}: Pred={row['pred']*100:.1f}%, Actual={row['actual']*100:.1f}%, Gap={gap:.1f}%")
    
    # ========================================
    # ROI Simulation
    # ========================================
    print("\n" + "=" * 80)
    print(" ROI Simulation: Raw vs Calibrated")
    print("=" * 80)
    
    prob_thresholds = [0.0, 0.15, 0.20, 0.25, 0.30]
    
    # Win bets
    print("\n--- 単勝 (Win Model) ---")
    print(f"{'Type':<12} | {'Prob閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'ROI':<10}")
    print("-" * 70)
    
    for prob_th in prob_thresholds:
        # Raw
        r = simulate_win(df_test_copy, df_test_copy['win_raw_norm'], payout_dict, prob_th)
        if r['bets'] > 0:
            roi = r['returns'] / r['cost'] * 100
            hit_rate = r['hits'] / r['bets'] * 100
            print(f"{'Raw':<12} | {prob_th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")
        
        # Calibrated
        r = simulate_win(df_test_copy, df_test_copy['win_calib'], payout_dict, prob_th)
        if r['bets'] > 0:
            roi = r['returns'] / r['cost'] * 100
            hit_rate = r['hits'] / r['bets'] * 100
            print(f"{'Calibrated':<12} | {prob_th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")
    
    # Place bets
    print("\n--- 複勝 (Top3 Model) ---")
    print(f"{'Type':<12} | {'Prob閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'ROI':<10}")
    print("-" * 70)
    
    for prob_th in prob_thresholds:
        # Raw
        r = simulate_place(df_test_copy, df_test_copy['top3_raw_norm'], payout_dict, prob_th)
        if r['bets'] > 0:
            roi = r['returns'] / r['cost'] * 100
            hit_rate = r['hits'] / r['bets'] * 100
            print(f"{'Raw':<12} | {prob_th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")
        
        # Calibrated
        r = simulate_place(df_test_copy, df_test_copy['top3_calib'], payout_dict, prob_th)
        if r['bets'] > 0:
            roi = r['returns'] / r['cost'] * 100
            hit_rate = r['hits'] / r['bets'] * 100
            print(f"{'Calibrated':<12} | {prob_th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")
    
    # EV-filtered with calibrated probs
    print("\n--- 単勝 + EV Filter (Calibrated) ---")
    print(f"{'EV閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'ROI':<10}")
    print("-" * 60)
    
    df_test_copy['pred_for_ev'] = df_test_copy['win_calib']
    for ev_th in [0.8, 1.0, 1.2, 1.5, 2.0]:
        r = simulate_ev(df_test_copy, df_test_copy['win_calib'], payout_dict, ev_th, 0.0)
        if r['bets'] > 0:
            roi = r['returns'] / r['cost'] * 100
            hit_rate = r['hits'] / r['bets'] * 100
            print(f"{ev_th:<10.1f} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | {roi:<9.1f}%")


if __name__ == "__main__":
    main()
