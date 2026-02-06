"""
Kelly Criterion Betting Simulation
==================================
Uses Kelly criterion for optimal bet sizing:
  f = (p * odds - 1) / (odds - 1)
  
Where:
  - f = fraction of bankroll to bet
  - p = calibrated win probability
  - odds = decimal odds

Also tests Fractional Kelly (1/2, 1/4, 1/8) for reduced variance.

Usage:
  python scripts/simulate_kelly.py
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
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"

INITIAL_BANKROLL = 100000  # ¥100,000 starting capital


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
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay
    FROM jvd_hr WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        try:
            h, p = int(row['win_horse']), int(row['win_pay'])
            if p > 0:
                payout_dict[rid] = {'horse': h, 'payout': p}
        except:
            pass
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


def train_calibrator(df_calib, preds):
    """Train isotonic regression calibrator"""
    df = df_calib.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum())
    df['target'] = (df['rank'] == 1).astype(int)
    
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(df['pred_norm'].values, df['target'].values)
    
    return ir


def apply_calibration(df, preds, calibrator):
    """Apply calibration to predictions"""
    df = df.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum())
    df['pred_calib'] = calibrator.predict(df['pred_norm'].values)
    
    return df


def kelly_fraction(prob, odds):
    """
    Calculate Kelly fraction
    f = (p * odds - 1) / (odds - 1)
    
    Returns 0 if negative (no edge)
    """
    if odds <= 1:
        return 0
    
    f = (prob * odds - 1) / (odds - 1)
    return max(0, f)  # Don't bet if negative edge


def simulate_kelly(df_test, payout_dict, kelly_multiplier=1.0, min_kelly=0.0, max_kelly=0.25):
    """
    Simulate Kelly betting
    
    Args:
        kelly_multiplier: Fractional Kelly (e.g., 0.5 for half Kelly)
        min_kelly: Minimum Kelly fraction to bet (filter out tiny edges)
        max_kelly: Maximum bet size as fraction of bankroll
    """
    bankroll = INITIAL_BANKROLL
    history = []
    
    # Sort by date and race for chronological simulation
    race_ids = df_test.groupby('race_id').first().sort_values('date').index.tolist()
    
    total_bets = 0
    wins = 0
    total_wagered = 0
    
    for rid in race_ids:
        if rid not in payout_dict:
            continue
        
        grp = df_test[df_test['race_id'] == rid].copy()
        grp = grp.sort_values('pred_calib', ascending=False)
        
        top = grp.iloc[0]
        prob = top['pred_calib']
        horse = int(top['horse_number'])
        
        # Get pre-race odds
        if 'odds_final' not in top or pd.isna(top['odds_final']) or top['odds_final'] <= 0:
            continue
        odds = top['odds_final']
        
        # Calculate Kelly fraction
        f = kelly_fraction(prob, odds)
        f = f * kelly_multiplier
        
        # Apply min/max constraints
        if f < min_kelly:
            continue
        f = min(f, max_kelly)
        
        # Calculate bet size
        bet_size = int(bankroll * f / 100) * 100  # Round to 100 yen
        if bet_size < 100:
            continue
        
        total_bets += 1
        total_wagered += bet_size
        
        # Check result
        actual = payout_dict[rid]
        if actual['horse'] == horse:
            # Winner!
            payout = bet_size * actual['payout'] / 100
            bankroll = bankroll - bet_size + payout
            wins += 1
        else:
            # Loser
            bankroll = bankroll - bet_size
        
        history.append({
            'race_id': rid,
            'date': top['date'],
            'prob': prob,
            'odds': odds,
            'kelly': f,
            'bet_size': bet_size,
            'bankroll': bankroll,
            'won': actual['horse'] == horse
        })
        
        # Bankruptcy check
        if bankroll < 100:
            break
    
    final_roi = bankroll / INITIAL_BANKROLL * 100
    hit_rate = wins / total_bets * 100 if total_bets > 0 else 0
    
    return {
        'final_bankroll': bankroll,
        'roi': final_roi,
        'bets': total_bets,
        'wins': wins,
        'hit_rate': hit_rate,
        'total_wagered': total_wagered,
        'history': history
    }


def simulate_fixed_bet(df_test, payout_dict, bet_amount=100):
    """Fixed bet simulation for comparison"""
    bankroll = INITIAL_BANKROLL
    history = []
    
    race_ids = df_test.groupby('race_id').first().sort_values('date').index.tolist()
    
    total_bets = 0
    wins = 0
    
    for rid in race_ids:
        if rid not in payout_dict:
            continue
        
        grp = df_test[df_test['race_id'] == rid].copy()
        grp = grp.sort_values('pred_calib', ascending=False)
        
        top = grp.iloc[0]
        horse = int(top['horse_number'])
        
        if bankroll < bet_amount:
            break
        
        total_bets += 1
        
        actual = payout_dict[rid]
        if actual['horse'] == horse:
            payout = bet_amount * actual['payout'] / 100
            bankroll = bankroll - bet_amount + payout
            wins += 1
        else:
            bankroll = bankroll - bet_amount
        
        history.append({'bankroll': bankroll, 'won': actual['horse'] == horse})
    
    return {
        'final_bankroll': bankroll,
        'roi': bankroll / INITIAL_BANKROLL * 100,
        'bets': total_bets,
        'wins': wins,
        'hit_rate': wins / total_bets * 100 if total_bets > 0 else 0,
        'history': history
    }


def main():
    logger.info("=" * 70)
    logger.info("Kelly Criterion Betting Simulation")
    logger.info("=" * 70)
    
    # Load data
    df = load_data()
    df_calib = df[df['date'].dt.year == 2023].copy()
    df_test = df[df['date'].dt.year == 2024].copy()
    
    logger.info(f"Calibration set (2023): {len(df_calib)} records")
    logger.info(f"Test set (2024): {len(df_test)} records")
    
    # Load payouts
    engine = get_db_engine()
    df_pay = load_payout_data(engine, [2024])
    payout_dict = parse_payouts(df_pay)
    logger.info(f"Loaded {len(payout_dict)} payout records")
    
    # Predict and calibrate
    logger.info("Predicting and calibrating...")
    preds_calib = predict_with_model(MODEL_PATH, df_calib)
    preds_test = predict_with_model(MODEL_PATH, df_test)
    
    calibrator = train_calibrator(df_calib, preds_calib)
    df_test = apply_calibration(df_test, preds_test, calibrator)
    
    print(f"\n{'='*80}")
    print(f" Kelly Criterion Simulation (Initial Bankroll: ¥{INITIAL_BANKROLL:,})")
    print(f"{'='*80}")
    
    # ========================================
    # Baseline: Fixed Betting
    # ========================================
    print("\n--- Baseline: Fixed ¥100 Betting ---")
    r = simulate_fixed_bet(df_test, payout_dict, 100)
    print(f"Final Bankroll: ¥{r['final_bankroll']:,.0f}")
    print(f"ROI: {r['roi']:.1f}%")
    print(f"Bets: {r['bets']}, Wins: {r['wins']}, Hit Rate: {r['hit_rate']:.1f}%")
    
    # ========================================
    # Kelly Variations
    # ========================================
    print("\n--- Kelly Criterion Variations ---")
    print(f"{'Kelly Type':<20} | {'Final ¥':<15} | {'ROI':<10} | {'Bets':<8} | {'Wins':<8} | {'Hit%':<10}")
    print("-" * 80)
    
    kelly_configs = [
        ('Full Kelly', 1.0, 0.01, 1.0),
        ('Half Kelly', 0.5, 0.005, 0.5),
        ('Quarter Kelly', 0.25, 0.0025, 0.25),
        ('1/8 Kelly', 0.125, 0.001, 0.125),
        ('1/16 Kelly', 0.0625, 0.0005, 0.0625),
        ('Full w/ 10% cap', 1.0, 0.01, 0.10),
        ('Half w/ 5% cap', 0.5, 0.005, 0.05),
        ('Quarter w/ 3% cap', 0.25, 0.0025, 0.03),
    ]
    
    results = []
    for name, mult, min_k, max_k in kelly_configs:
        r = simulate_kelly(df_test, payout_dict, mult, min_k, max_k)
        print(f"{name:<20} | ¥{r['final_bankroll']:>12,.0f} | {r['roi']:>8.1f}% | {r['bets']:<8} | {r['wins']:<8} | {r['hit_rate']:<9.1f}%")
        results.append({'name': name, **r})
    
    # ========================================
    # Best Result Analysis
    # ========================================
    best = max(results, key=lambda x: x['final_bankroll'])
    print(f"\n🏆 Best Strategy: {best['name']}")
    print(f"   Final Bankroll: ¥{best['final_bankroll']:,.0f}")
    print(f"   ROI: {best['roi']:.1f}%")
    print(f"   Total Bets: {best['bets']}, Wins: {best['wins']}")
    
    # ========================================
    # Profitable strategies (ROI > 100%)
    # ========================================
    profitable = [r for r in results if r['roi'] > 100]
    if profitable:
        print(f"\n✅ Profitable Strategies (ROI > 100%):")
        for r in sorted(profitable, key=lambda x: x['roi'], reverse=True):
            print(f"   - {r['name']}: ROI {r['roi']:.1f}%, ¥{r['final_bankroll']:,.0f}")
    else:
        print(f"\n❌ No profitable strategies found (all ROI < 100%)")
    
    # ========================================
    # Bankroll trajectory for best strategy
    # ========================================
    if best['history']:
        print(f"\n--- Bankroll Trajectory ({best['name']}) ---")
        hist = best['history']
        n = len(hist)
        checkpoints = [0, n//4, n//2, 3*n//4, n-1]
        for i in checkpoints:
            h = hist[min(i, n-1)]
            print(f"  After {i+1} bets: ¥{h['bankroll']:,.0f}")


if __name__ == "__main__":
    main()
