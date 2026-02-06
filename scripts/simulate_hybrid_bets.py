"""
Hybrid Multi-Model Betting Simulation
=====================================
Combines different models for different parts of the bet:
- Win model for 1st place selection
- Top2 model for 2nd place candidates
- Top3 model for 3rd place candidates

Strategies:
1. 馬連ながし: Win(1着固定) → Top2(2着上位N頭)
2. ワイドながし: Win(1着固定) → Top3(相手上位N頭)
3. 3連複ながし: Win(軸) ← Top2(2着) ← Top3(3着)

Usage:
  python scripts/simulate_hybrid_bets.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from itertools import combinations, product
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
    """Load all payout types including trifecta"""
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
        haraimodoshi_wide_3a as wide_3_horses, haraimodoshi_wide_3b as wide_3_pay,
        haraimodoshi_sanrenpuku_1a as trio_horses, haraimodoshi_sanrenpuku_1b as trio_pay
    FROM jvd_hr 
    WHERE kaisai_nen IN ({year_list})
    """
    df = pd.read_sql(query, engine)
    df['race_id'] = df['race_id'].astype(str)
    return df


def parse_payouts(df_pay):
    """Parse payout data"""
    payout_dict = {}
    for _, row in df_pay.iterrows():
        rid = row['race_id']
        pay = {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}, 'trio': set()}
        
        # Win
        try:
            h = int(row['win_horse'])
            p = int(row['win_pay'])
            if p > 0: pay['win'][h] = p
        except: pass
        
        # Place
        for i in [1, 2, 3]:
            try:
                h = int(row[f'place_{i}_horse'])
                p = int(row[f'place_{i}_pay'])
                if p > 0: pay['place'][h] = p
            except: pass
        
        # Quinella
        try:
            horses_str = str(row['quinella_horses']).zfill(4)
            h1, h2 = int(horses_str[:2]), int(horses_str[2:])
            p = int(row['quinella_pay'])
            if p > 0:
                pay['quinella'][(min(h1,h2), max(h1,h2))] = p
        except: pass
        
        # Wide
        for i in [1, 2, 3]:
            try:
                horses_str = str(row[f'wide_{i}_horses']).zfill(4)
                h1, h2 = int(horses_str[:2]), int(horses_str[2:])
                p = int(row[f'wide_{i}_pay'])
                if p > 0:
                    pay['wide'][(min(h1,h2), max(h1,h2))] = p
            except: pass
        
        # Trio (3連複) - just store the winning combination
        try:
            horses_str = str(row['trio_horses']).zfill(6)
            h1, h2, h3 = int(horses_str[:2]), int(horses_str[2:4]), int(horses_str[4:])
            p = int(row['trio_pay'])
            if p > 0:
                pay['trio'] = {'horses': tuple(sorted([h1, h2, h3])), 'pay': p}
        except:
            pay['trio'] = None
        
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


def simulate_quinella_nagashi(df_test, win_preds, top2_preds, payout_dict, n_second=3):
    """
    馬連ながし: Win model picks 1st, Top2 model picks 2nd candidates
    Buy: Win_top1 → Top2_top(n_second) (excluding the axis horse)
    """
    df = df_test.copy()
    df['win_pred'] = win_preds
    df['top2_pred'] = top2_preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        
        # Win model: select axis horse (1st place prediction)
        grp_win = grp.sort_values('win_pred', ascending=False)
        axis_horse = int(grp_win.iloc[0]['horse_number'])
        
        # Top2 model: select 2nd place candidates (excluding axis)
        grp_top2 = grp[grp['horse_number'] != axis_horse].sort_values('top2_pred', ascending=False)
        second_horses = [int(grp_top2.iloc[i]['horse_number']) for i in range(min(n_second, len(grp_top2)))]
        
        # Buy quinella tickets: axis → each second candidate
        for h2 in second_horses:
            key = (min(axis_horse, h2), max(axis_horse, h2))
            cost += 100
            bets += 1
            
            if key in pay['quinella']:
                returns += pay['quinella'][key]
                hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_wide_nagashi(df_test, win_preds, top3_preds, payout_dict, n_targets=3):
    """
    ワイドながし: Win model picks axis, Top3 model picks target candidates
    Buy: Win_top1 → Top3_top(n_targets)
    """
    df = df_test.copy()
    df['win_pred'] = win_preds
    df['top3_pred'] = top3_preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        
        # Win model: select axis horse
        grp_win = grp.sort_values('win_pred', ascending=False)
        axis_horse = int(grp_win.iloc[0]['horse_number'])
        
        # Top3 model: select target candidates (excluding axis)
        grp_top3 = grp[grp['horse_number'] != axis_horse].sort_values('top3_pred', ascending=False)
        target_horses = [int(grp_top3.iloc[i]['horse_number']) for i in range(min(n_targets, len(grp_top3)))]
        
        # Buy wide tickets
        for h2 in target_horses:
            key = (min(axis_horse, h2), max(axis_horse, h2))
            cost += 100
            bets += 1
            
            if key in pay['wide']:
                returns += pay['wide'][key]
                hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_trio_formation(df_test, win_preds, top2_preds, top3_preds, payout_dict, n_second=2, n_third=3):
    """
    3連複フォーメーション: Win(1着軸) × Top2(2着候補) × Top3(3着候補)
    """
    df = df_test.copy()
    df['win_pred'] = win_preds
    df['top2_pred'] = top2_preds
    df['top3_pred'] = top3_preds
    
    cost = 0
    returns = 0
    bets = 0
    hits = 0
    
    for rid, grp in df.groupby('race_id'):
        if rid not in payout_dict:
            continue
        
        pay = payout_dict[rid]
        if pay['trio'] is None:
            continue
        
        # Win model: 1st place axis
        grp_win = grp.sort_values('win_pred', ascending=False)
        first_horse = int(grp_win.iloc[0]['horse_number'])
        
        # Top2 model: 2nd place candidates
        grp_top2 = grp[grp['horse_number'] != first_horse].sort_values('top2_pred', ascending=False)
        second_horses = [int(grp_top2.iloc[i]['horse_number']) for i in range(min(n_second, len(grp_top2)))]
        
        # Top3 model: 3rd place candidates (excluding 1st and 2nd selections)
        exclude_set = {first_horse} | set(second_horses)
        grp_top3 = grp[~grp['horse_number'].isin(exclude_set)].sort_values('top3_pred', ascending=False)
        third_horses = [int(grp_top3.iloc[i]['horse_number']) for i in range(min(n_third, len(grp_top3)))]
        
        # Generate all trio combinations
        winning_trio = pay['trio']['horses']
        
        for h2 in second_horses:
            for h3 in third_horses:
                combo = tuple(sorted([first_horse, h2, h3]))
                cost += 100
                bets += 1
                
                if combo == winning_trio:
                    returns += pay['trio']['pay']
                    hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def simulate_single_model_quinella_nagashi(df_test, preds, payout_dict, n_second=3):
    """Single model quinella nagashi for comparison: Top1 → Top(2 to n_second+1)"""
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
        
        if len(grp) < n_second + 1:
            continue
        
        axis_horse = int(grp.iloc[0]['horse_number'])
        second_horses = [int(grp.iloc[i]['horse_number']) for i in range(1, n_second + 1)]
        
        for h2 in second_horses:
            key = (min(axis_horse, h2), max(axis_horse, h2))
            cost += 100
            bets += 1
            
            if key in pay['quinella']:
                returns += pay['quinella'][key]
                hits += 1
    
    return {'cost': cost, 'returns': returns, 'bets': bets, 'hits': hits}


def main():
    logger.info("=" * 70)
    logger.info("Hybrid Multi-Model Betting Simulation")
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
    
    print("\n" + "=" * 80)
    print(" Hybrid Strategies: Combining Multiple Models")
    print("=" * 80)
    
    results = []
    
    # ==========================================
    # 馬連ながし: Win(軸) → Top2(相手N頭)
    # ==========================================
    print("\n--- 馬連ながし: Win(1着軸) → Top2(2着候補) ---")
    print(f"{'N頭':<8} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 80)
    
    for n in [2, 3, 4, 5]:
        r = simulate_quinella_nagashi(df_test, predictions['Win'], predictions['Top2'], payout_dict, n)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{n}頭     | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
        results.append({'strategy': f'馬連ながしWin→Top2({n}頭)', 'roi': roi, 'hit_rate': hit_rate, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
    
    # Comparison: Single model quinella nagashi
    print("\n--- 比較: 単一モデル馬連ながし (3頭) ---")
    for model_name in ['Win', 'Top2', 'Top3']:
        r = simulate_single_model_quinella_nagashi(df_test, predictions[model_name], payout_dict, 3)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{model_name:<8} | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
        results.append({'strategy': f'馬連ながし{model_name}(3頭)', 'roi': roi, 'hit_rate': hit_rate, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
    
    # ==========================================
    # ワイドながし: Win(軸) → Top3(相手N頭)
    # ==========================================
    print("\n--- ワイドながし: Win(軸) → Top3(相手候補) ---")
    print(f"{'N頭':<8} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 80)
    
    for n in [2, 3, 4, 5]:
        r = simulate_wide_nagashi(df_test, predictions['Win'], predictions['Top3'], payout_dict, n)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{n}頭     | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
        results.append({'strategy': f'ワイドながしWin→Top3({n}頭)', 'roi': roi, 'hit_rate': hit_rate, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
    
    # ==========================================
    # 3連複フォーメーション
    # ==========================================
    print("\n--- 3連複フォーメーション: Win(1着) × Top2(2着) × Top3(3着) ---")
    print(f"{'2着×3着':<12} | {'Bets':<8} | {'Hits':<8} | {'HitRate':<10} | {'Cost':<12} | {'Returns':<12} | {'ROI':<10}")
    print("-" * 80)
    
    for n2, n3 in [(2, 2), (2, 3), (3, 3), (2, 4), (3, 4)]:
        r = simulate_trio_formation(df_test, predictions['Win'], predictions['Top2'], predictions['Top3'], payout_dict, n2, n3)
        roi = r['returns'] / r['cost'] * 100 if r['cost'] > 0 else 0
        hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
        print(f"{n2}頭×{n3}頭    | {r['bets']:<8} | {r['hits']:<8} | {hit_rate:<9.1f}% | ¥{r['cost']:<10,} | ¥{r['returns']:<10,} | {roi:<9.1f}%")
        results.append({'strategy': f'3連複フォーメーション({n2}×{n3})', 'roi': roi, 'hit_rate': hit_rate, 'bets': r['bets'], 'profit': r['returns'] - r['cost']})
    
    # ==========================================
    # Best Results
    # ==========================================
    print("\n" + "=" * 80)
    print(" Top 10 Strategies by ROI")
    print("=" * 80)
    
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'Rank':<6} | {'Strategy':<35} | {'ROI':<10} | {'HitRate':<10} | {'Profit':<12}")
    print("-" * 80)
    for i, r in enumerate(results[:10], 1):
        sign = '+' if r['profit'] >= 0 else ''
        print(f"{i:<6} | {r['strategy']:<35} | {r['roi']:<9.1f}% | {r['hit_rate']:<9.1f}% | {sign}¥{r['profit']:,}")


if __name__ == "__main__":
    main()
