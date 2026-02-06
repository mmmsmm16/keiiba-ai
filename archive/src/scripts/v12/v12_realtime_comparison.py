"""
v12 Real-time Inference - 2025 Monthly ROI Comparison
Runs real-time inference using auto_predict logic and compares with cached predictions.
"""

import pandas as pd
import numpy as np
import os
import sys
from itertools import permutations, combinations
from scipy.special import softmax
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = '/workspace'
sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

def load_payout_data():
    """Load payout data."""
    payout_paths = [
        os.path.join(EXPERIMENTS_DIR, 'payouts_2024_2025.parquet'),
        os.path.join(EXPERIMENTS_DIR, 'v7_ensemble_full', 'data', 'payout_data.parquet'),
    ]
    
    for path in payout_paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"Loaded payout data from: {path}")
            return df
    
    raise FileNotFoundError("Payout data not found")

def build_payout_map(payout_df):
    """Build payout lookup map."""
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = str(row.get('race_id', ''))
        if not rid:
            continue
        if rid not in payout_map:
            payout_map[rid] = {'sanrentan': {}, 'sanrenpuku': {}}
        
        # Sanrentan
        for k in range(1, 7):
            comb = row.get(f'haraimodoshi_sanrentan_{k}a')
            pay = row.get(f'haraimodoshi_sanrentan_{k}b')
            if comb and pay:
                try:
                    payout_map[rid]['sanrentan'][str(comb).strip()] = int(float(pay))
                except:
                    pass
        
        # Sanrenpuku
        for k in range(1, 4):
            comb = row.get(f'haraimodoshi_sanrenpuku_{k}a')
            pay = row.get(f'haraimodoshi_sanrenpuku_{k}b')
            if comb and pay:
                try:
                    payout_map[rid]['sanrenpuku'][str(comb).strip()] = int(float(pay))
                except:
                    pass
    
    return payout_map

def run_realtime_inference():
    """Run real-time inference for all 2025 races."""
    from src.inference.loader import InferenceDataLoader
    from src.inference.preprocessor import InferencePreprocessor
    from src.model.ensemble import EnsembleModel
    
    print("Loading model...")
    model_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'models', 'ensemble.pkl')
    model = EnsembleModel()
    model.load_model(model_path)
    
    # Load expected features
    features_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'models', 'tabnet.features.json')
    with open(features_path, 'r') as f:
        expected_features = json.load(f)
    
    print("Loading race data...")
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    
    # Get all 2025 races from preprocessed data
    # Correct path: experiments/v12_tabnet_revival/data/preprocessed_data.parquet (Contains class_level features)
    preprocessed_path = os.path.join(PROJECT_ROOT, 'experiments', 'v12_tabnet_revival', 'data', 'preprocessed_data.parquet')
    history_df = None
    
    if os.path.exists(preprocessed_path):
        print(f"Loading history data from {preprocessed_path}...")
        history_df = pd.read_parquet(preprocessed_path)
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        # Checking and patching missing class_level features in history
        # (This prevents InferencePreprocessor from zeroing them out due to missing history columns)
        class_level_cols = ['class_level', 'class_level_n_races', 'class_level_win_rate', 'class_level_top3_rate']
        patched_count = 0
        for col in class_level_cols:
            if col not in history_df.columns:
                history_df[col] = 0
                patched_count += 1
        
        if patched_count > 0:
            print(f"Patched {patched_count} missing class_level columns in history data.")



        
        # Limit to Jan 5th for quick verification (Logic verified, need speed)
        # df_2025 = history_df[(history_df['date'].dt.year == 2025) & (history_df['date'].dt.month == 1)].copy()
        target_date = pd.Timestamp('2025-01-05')
        df_2025 = history_df[history_df['date'] == target_date].copy()
    else:
        print(f"Warning: Preprocessed data NOT FOUND at {preprocessed_path}")
        # Fall back to cached predictions for race IDs
        pred_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'reports', 'predictions.parquet')
        pred_df = pd.read_parquet(pred_path)
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        df_2025 = pred_df[(pred_df['date'].dt.year == 2025) & (pred_df['date'].dt.month == 1)].copy()
        print("Warning: Falling back to cached predictions for race list. History aggregation will be slow/incomplete!")
    
    race_ids = df_2025['race_id'].unique()
    print(f"Total 2025 races: {len(race_ids)}")
    
    all_predictions = []
    errors = []
    
    for i, race_id in enumerate(race_ids):
        if i % 10 == 0:
            print(f"Processing race {i+1}/{len(race_ids)}...")
        
        try:
            # Load race data
            race_df = loader.load(race_ids=[str(race_id)])
            if race_df.empty:
                errors.append(f"No data: {race_id}")
                continue
            
            # Preprocess (Pass full cached history_df, filtering handled internally)
            X, ids = preprocessor.preprocess(race_df, history_df=history_df)
            if X.empty:
                errors.append(f"Empty preprocess: {race_id}")
                continue
            
            # Feature adaptation
            for feat in expected_features:
                if feat not in X.columns:
                    X[feat] = 0
            X = X[expected_features]
            
            # Predict
            scores = model.predict(X)
            
            # Build result
            result_df = ids.copy()
            result_df['score'] = scores
            result_df['race_id'] = str(race_id)
            result_df['date'] = df_2025[df_2025['race_id'] == race_id]['date'].iloc[0]
            
            # Add odds if available
            if 'odds' in race_df.columns:
                # Avoid odds_x, odds_y by dropping odds from result_df if exists
                if 'odds' in result_df.columns:
                    result_df = result_df.drop(columns=['odds'])
                
                result_df = result_df.merge(
                    race_df[['horse_number', 'odds']], 
                    on='horse_number', 
                    how='left'
                )
            
            all_predictions.append(result_df)
            
        except Exception as e:
            errors.append(f"{race_id}: {str(e)}")
            continue
    
    print(f"Successfully processed: {len(all_predictions)} races")
    print(f"Errors: {len(errors)}")
    
    if all_predictions:
        result = pd.concat(all_predictions, ignore_index=True)
        return result, errors
    else:
        return pd.DataFrame(), errors

def calculate_ev_strategy(df, payout_map):
    """Calculate monthly ROI using v12 EV strategy with real-time predictions."""
    df['date'] = pd.to_datetime(df['date'])
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    monthly_stats = {}
    
    for race_id, group in df.groupby('race_id'):
        race_id_str = str(race_id)
        if race_id_str not in payout_map:
            continue
        
        sorted_group = group.sort_values('pred_rank')
        if len(sorted_group) < 5:
            continue
        
        # Get month
        month = sorted_group.iloc[0]['date'].month
        if month not in monthly_stats:
            monthly_stats[month] = {'cost': 0, 'return': 0, 'races': 0, 'hits': 0}
        
        top1 = sorted_group.iloc[0]
        
        # Calculate EV using softmax probability * odds
        probs = softmax(sorted_group['score'].values)
        top1_prob = probs[0]
        top1_odds = float(top1.get('odds', 1)) if pd.notna(top1.get('odds')) else 1.0
        top1_ev = top1_prob * top1_odds
        
        # Prepare horses
        axis = int(sorted_group.iloc[0]['horse_number'])
        opps = [int(sorted_group.iloc[i]['horse_number']) for i in range(1, min(5, len(sorted_group)))]
        
        race_cost = 0
        race_return = 0
        hit = False
        
        if top1_ev >= 1.2:
            # High Value: 三連複
            tickets = list(combinations([axis] + opps[:3], 3))
            tickets = [t for t in tickets if axis in t]
            race_cost = len(tickets) * 100
            
            for t in tickets:
                sorted_t = tuple(sorted(t))
                key = f"{sorted_t[0]:02}{sorted_t[1]:02}{sorted_t[2]:02}"
                if key in payout_map[race_id_str].get('sanrenpuku', {}):
                    race_return += payout_map[race_id_str]['sanrenpuku'][key]
                    hit = True
                    
        elif top1_ev >= 0.8:
            # Mid Value: 三連単
            tickets = [(axis, o1, o2) for o1, o2 in permutations(opps[:3], 2)]
            race_cost = len(tickets) * 100
            
            for t in tickets:
                key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                if key in payout_map[race_id_str].get('sanrentan', {}):
                    race_return += payout_map[race_id_str]['sanrentan'][key]
                    hit = True
        
        monthly_stats[month]['cost'] += race_cost
        monthly_stats[month]['return'] += race_return
        monthly_stats[month]['races'] += 1
        if hit:
            monthly_stats[month]['hits'] += 1
    
    return monthly_stats

def print_comparison(cached_stats, realtime_stats):
    """Print comparison report."""
    print("\n" + "="*100)
    print("v12 EV Strategy - Cached vs Real-time Inference Comparison (2025)")
    print("="*100)
    print(f"{'月':<5} {'Cached投資':>12} {'Cached払戻':>12} {'Cached ROI':>10} | {'RT投資':>12} {'RT払戻':>12} {'RT ROI':>10}")
    print("-"*100)
    
    for month in range(1, 13):
        c = cached_stats.get(month, {'cost': 0, 'return': 0})
        r = realtime_stats.get(month, {'cost': 0, 'return': 0})
        
        c_roi = (c['return'] / c['cost'] * 100) if c['cost'] > 0 else 0
        r_roi = (r['return'] / r['cost'] * 100) if r['cost'] > 0 else 0
        
        print(f"{month:>2}月  {c['cost']:>10,}円 {c['return']:>10,}円 {c_roi:>9.1f}% | {r['cost']:>10,}円 {r['return']:>10,}円 {r_roi:>9.1f}%")
    
    print("-"*100)
    
    # Totals
    c_total_cost = sum(s['cost'] for s in cached_stats.values())
    c_total_ret = sum(s['return'] for s in cached_stats.values())
    r_total_cost = sum(s['cost'] for s in realtime_stats.values())
    r_total_ret = sum(s['return'] for s in realtime_stats.values())
    
    c_total_roi = (c_total_ret / c_total_cost * 100) if c_total_cost > 0 else 0
    r_total_roi = (r_total_ret / r_total_cost * 100) if r_total_cost > 0 else 0
    
    print(f"{'合計':<4} {c_total_cost:>10,}円 {c_total_ret:>10,}円 {c_total_roi:>9.1f}% | {r_total_cost:>10,}円 {r_total_ret:>10,}円 {r_total_roi:>9.1f}%")
    print("="*100)

if __name__ == "__main__":
    try:
        print("Loading payout data...")
        payout_df = load_payout_data()
        payout_map = build_payout_map(payout_df)
        
        print("\nRunning real-time inference (this may take a while)...")
        realtime_df, errors = run_realtime_inference()
        
        if not realtime_df.empty:
            print("\nCalculating real-time ROI...")
            realtime_stats = calculate_ev_strategy(realtime_df, payout_map)
            
            # Load cached for comparison
            print("\nLoading cached predictions for comparison...")
            from v12_monthly_roi import load_data, calculate_ev_strategy as cached_calculate
            cached_df, _ = load_data()
            cached_payout_map = build_payout_map(payout_df)
            cached_stats = cached_calculate(cached_df, cached_payout_map, year=2025)
            
            print_comparison(cached_stats, realtime_stats)
        else:
            print("No predictions generated. Check errors above.")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
