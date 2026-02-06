"""
Hybrid Betting with Probability Filtering
==========================================
Tests filtering by predicted probability thresholds.

Usage:
  python scripts/simulate_filtered_bets.py
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
        haraimodoshi_fukusho_3a as place_3_horse, haraimodoshi_fukusho_3b as place_3_pay,
        haraimodoshi_umaren_1a as quinella_horses, haraimodoshi_umaren_1b as quinella_pay,
        haraimodoshi_wide_1a as wide_1_horses, haraimodoshi_wide_1b as wide_1_pay,
        haraimodoshi_wide_2a as wide_2_horses, haraimodoshi_wide_2b as wide_2_pay,
        haraimodoshi_wide_3a as wide_3_horses, haraimodoshi_wide_3b as wide_3_pay
    FROM jvd_hr WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
        
        try:
            h, p = int(row['win_horse']), int(row['win_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        
        for i in [1, 2, 3]:
            try:
                h, p = int(row[f'place_{i}_horse']), int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        
        try:
            hs = str(row['quinella_horses']).zfill(4)
            h1, h2 = int(hs[:2]), int(hs[2:])
            p = int(row['quinella_pay'])
            if p > 0: pay['quinella'][(min(h1,h2), max(h1,h2))] = p
        except: pass
        
        for i in [1, 2, 3]:
            try:
                hs = str(row[f'wide_{i}_horses']).zfill(4)
                h1, h2 = int(hs[:2]), int(hs[2:])
                p = int(row[f'wide_{i}_pay'])
                if p > 0: pay['wide'][(min(h1,h2), max(h1,h2))] = p
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


def simulate_win_filtered(df_test, preds, payout_dict, prob_threshold=0.0):
    """単勝: 確率閾値以上の馬のみベット"""
    df = df_test.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        if top['pred_norm'] < prob_threshold: continue
        
        horse = int(top['horse_number'])
        cost += 100
        bets += 1
        if horse in pay['win']:
            returns += pay['win'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_place_filtered(df_test, preds, payout_dict, prob_threshold=0.0):
    """複勝: 確率閾値以上の馬のみベット"""
    df = df_test.copy()
    df['pred'] = preds
    df['pred_norm'] = df.groupby('race_id')['pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        grp = grp.sort_values('pred', ascending=False)
        top = grp.iloc[0]
        
        if top['pred_norm'] < prob_threshold: continue
        
        horse = int(top['horse_number'])
        cost += 100
        bets += 1
        if horse in pay['place']:
            returns += pay['place'][horse]
            hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_quinella_nagashi_filtered(df_test, win_preds, top2_preds, payout_dict, 
                                        prob_threshold=0.0, n_second=3):
    """馬連ながし: 軸馬の確率閾値でフィルタ"""
    df = df_test.copy()
    df['win_pred'] = win_preds
    df['top2_pred'] = top2_preds
    df['win_norm'] = df.groupby('race_id')['win_pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        grp_win = grp.sort_values('win_pred', ascending=False)
        axis = grp_win.iloc[0]
        
        if axis['win_norm'] < prob_threshold: continue
        
        axis_horse = int(axis['horse_number'])
        grp_top2 = grp[grp['horse_number'] != axis_horse].sort_values('top2_pred', ascending=False)
        seconds = [int(grp_top2.iloc[i]['horse_number']) for i in range(min(n_second, len(grp_top2)))]
        
        for h2 in seconds:
            key = (min(axis_horse, h2), max(axis_horse, h2))
            cost += 100
            bets += 1
            if key in pay['quinella']:
                returns += pay['quinella'][key]
                hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_wide_nagashi_filtered(df_test, win_preds, top3_preds, payout_dict,
                                    prob_threshold=0.0, n_targets=4):
    """ワイドながし: 軸馬の確率閾値でフィルタ"""
    df = df_test.copy()
    df['win_pred'] = win_preds
    df['top3_pred'] = top3_preds
    df['win_norm'] = df.groupby('race_id')['win_pred'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    
    cost, returns, bets, hits = 0, 0, 0, 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict: continue
        pay = payout_dict[rid]
        
        grp_win = grp.sort_values('win_pred', ascending=False)
        axis = grp_win.iloc[0]
        
        if axis['win_norm'] < prob_threshold: continue
        
        axis_horse = int(axis['horse_number'])
        grp_top3 = grp[grp['horse_number'] != axis_horse].sort_values('top3_pred', ascending=False)
        targets = [int(grp_top3.iloc[i]['horse_number']) for i in range(min(n_targets, len(grp_top3)))]
        
        for h2 in targets:
            key = (min(axis_horse, h2), max(axis_horse, h2))
            cost += 100
            bets += 1
            if key in pay['wide']:
                returns += pay['wide'][key]
                hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def main():
    logger.info("=" * 70)
    logger.info("Probability Filtered Betting Simulation")
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
    
    prob_thresholds = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" 単勝 (Win Model) - Probability Filtering")
    print("=" * 90)
    print(f"{'Prob閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    
    for th in prob_thresholds:
        r = simulate_win_filtered(df_test, predictions['Win'], payout_dict, th)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" 複勝 (Top3 Model) - Probability Filtering")
    print("=" * 90)
    print(f"{'Prob閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    
    for th in prob_thresholds:
        r = simulate_place_filtered(df_test, predictions['Top3'], payout_dict, th)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" 馬連ながし (Win→Top2, 3頭) - Probability Filtering")
    print("=" * 90)
    print(f"{'Prob閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    
    for th in prob_thresholds:
        r = simulate_quinella_nagashi_filtered(df_test, predictions['Win'], predictions['Top2'], 
                                               payout_dict, th, 3)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
    
    # ==========================================
    print("\n" + "=" * 90)
    print(" ワイドながし (Win→Top3, 4頭) - Probability Filtering")
    print("=" * 90)
    print(f"{'Prob閾値':<10} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 90)
    
    for th in prob_thresholds:
        r = simulate_wide_nagashi_filtered(df_test, predictions['Win'], predictions['Top3'],
                                           payout_dict, th, 4)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{th:<10.0%} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
    
    # ==========================================
    # Grid search for best combination
    # ==========================================
    print("\n" + "=" * 90)
    print(" Best Configurations (Grid Search)")
    print("=" * 90)
    
    results = []
    
    # Grid search: bet type × prob threshold
    for th in prob_thresholds:
        # 単勝
        r = simulate_win_filtered(df_test, predictions['Win'], payout_dict, th)
        if r['bets'] >= 100:
            roi = r['returns'] / r['cost'] * 100
            results.append({'bet': '単勝', 'prob': th, 'roi': roi, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
        
        # 複勝
        r = simulate_place_filtered(df_test, predictions['Top3'], payout_dict, th)
        if r['bets'] >= 100:
            roi = r['returns'] / r['cost'] * 100
            results.append({'bet': '複勝', 'prob': th, 'roi': roi, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
        
        # 馬連ながし
        r = simulate_quinella_nagashi_filtered(df_test, predictions['Win'], predictions['Top2'], payout_dict, th, 3)
        if r['bets'] >= 100:
            roi = r['returns'] / r['cost'] * 100
            results.append({'bet': '馬連ながし', 'prob': th, 'roi': roi, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
        
        # ワイドながし
        r = simulate_wide_nagashi_filtered(df_test, predictions['Win'], predictions['Top3'], payout_dict, th, 4)
        if r['bets'] >= 100:
            roi = r['returns'] / r['cost'] * 100
            results.append({'bet': 'ワイドながし', 'prob': th, 'roi': roi, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
    
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'Rank':<6} | {'Bet Type':<15} | {'Prob閾値':<10} | {'ROI':<10} | {'Bets':<8} | {'Profit':<12}")
    print("-" * 70)
    for i, r in enumerate(results[:15], 1):
        sign = '+' if r['profit'] >= 0 else ''
        print(f"{i:<6} | {r['bet']:<15} | {r['prob']:<10.0%} | {r['roi']:<9.1f}% | {r['bets']:<8} | {sign}¥{r['profit']:,}")


if __name__ == "__main__":
    main()
