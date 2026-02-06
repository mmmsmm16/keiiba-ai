
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import sys
import pickle
import itertools
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/exp_lambdarank/model.pkl"

def get_db_engine():
    import os
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'postgres')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payouts(year=2024):
    logger.info(f"Loading payouts for {year} (Win/Umaren/Umatan/Sanrenpuku/Sanrentan)...")
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_umaren_1a as umaren_comb,
        haraimodoshi_umaren_1b as umaren_pay,
        haraimodoshi_umatan_1a as umatan_comb,
        haraimodoshi_umatan_1b as umatan_pay,
        haraimodoshi_sanrenpuku_1a as sanrenpuku_comb,
        haraimodoshi_sanrenpuku_1b as sanrenpuku_pay,
        haraimodoshi_sanrentan_1a as sanrentan_comb,
        haraimodoshi_sanrentan_1b as sanrentan_pay
    FROM jvd_hr
    WHERE kaisai_nen = '{year}'
    """
    engine = get_db_engine()
    try:
        payouts = pd.read_sql(query, engine)
        cols = ['umaren_pay', 'umatan_pay', 'sanrenpuku_pay', 'sanrentan_pay']
        for c in cols:
            payouts[c] = pd.to_numeric(payouts[c], errors='coerce').fillna(0)
        return payouts
    except Exception as e:
        logger.error(f"Failed to load payouts: {e}")
        return pd.DataFrame()

def generate_combinations(rank_df, strategy_name):
    """
    Generate purchased combinations for a single race based on strategy.
    rank_df: DataFrame with 'horse_number', 'pred_rank' (1-based)
    Returns: List of dicts {'type': 'umaren', 'comb': '0102'}
    """
    # Sort by rank
    ranks = rank_df.sort_values('pred_rank')
    
    # Helpers
    def fmt2(n): return str(int(n)).zfill(2)
    def to_umaren(h1, h2): return "".join(sorted([fmt2(h1), fmt2(h2)]))
    def to_umatan(h1, h2): return fmt2(h1) + fmt2(h2)
    def to_sanrenpuku(h1, h2, h3): return "".join(sorted([fmt2(h1), fmt2(h2), fmt2(h3)]))
    def to_sanrentan(h1, h2, h3): return fmt2(h1) + fmt2(h2) + fmt2(h3)

    purchases = []
    
    # Top horses needed
    top1 = ranks[ranks['pred_rank'] == 1]['horse_number'].values[0] if len(ranks)>=1 else None
    top2 = ranks[ranks['pred_rank'] == 2]['horse_number'].values[0] if len(ranks)>=2 else None
    top3 = ranks[ranks['pred_rank'] == 3]['horse_number'].values[0] if len(ranks)>=3 else None
    
    # Valid Top 5 list
    top_n = lambda n: ranks[ranks['pred_rank'] <= n]['horse_number'].tolist()
    
    if strategy_name == 'Umaren_Box5':
        # Box 5 horses: 5C2 = 10 bets
        horses = top_n(5)
        for h1, h2 in itertools.combinations(horses, 2):
            purchases.append({'type': 'umaren', 'comb': to_umaren(h1, h2), 'cost': 100})
            
    elif strategy_name == 'Sanrenpuku_Box5':
        # Box 5 horses: 5C3 = 10 bets
        horses = top_n(5)
        for h1, h2, h3 in itertools.combinations(horses, 3):
            purchases.append({'type': 'sanrenpuku', 'comb': to_sanrenpuku(h1, h2, h3), 'cost': 100})

    elif strategy_name == 'Umatan_Nagashi_1_to_6':
        # Axis: Rank 1, Partner: Rank 2-6 (5 bets)
        partners = top_n(6)[1:] # remove rank 1
        for p in partners:
            purchases.append({'type': 'umatan', 'comb': to_umatan(top1, p), 'cost': 100})
            
    elif strategy_name == 'Sanrentan_Nagashi_1_to_6':
        # Axis 1st: Rank 1. 2nd/3rd: Rank 2-6 (Box). 
        # 1 -> (2..6) -> (2..6). Pts: 5 * 4 = 20 bets.
        partners = top_n(6)[1:]
        for h2 in partners:
            for h3 in partners:
                if h2 == h3: continue
                purchases.append({'type': 'sanrentan', 'comb': to_sanrentan(top1, h2, h3), 'cost': 100})
                
    elif strategy_name == 'Sanrentan_Formation_12_123_12345':
        # 1st: 1,2
        # 2nd: 1,2,3
        # 3rd: 1,2,3,4,5
        # Exclude duplicate horses in same bet.
        s1 = top_n(2)
        s2 = top_n(3)
        s3 = top_n(5)
        
        for h1 in s1:
            for h2 in s2:
                if h1 == h2: continue
                for h3 in s3:
                    if h3 == h1 or h3 == h2: continue
                    purchases.append({'type': 'sanrentan', 'comb': to_sanrentan(h1, h2, h3), 'cost': 100})
                    
    return purchases

def main():
    logger.info("loading test data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df_test = df[df['date'].dt.year == 2024].copy()
    
    # Filter JRA
    df_test['venue_code'] = df_test['race_id'].astype(str).str[4:6]
    jra_mask = df_test['venue_code'].isin([str(i).zfill(2) for i in range(1, 11)])
    df_test = df_test[jra_mask]
    
    # Load Model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    feature_names = model.feature_name()
    X_test = df_test[feature_names].copy()
    
    # Preprocess
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category' or X_test[col].dtype == 'object':
             X_test[col] = X_test[col].astype('category').cat.codes
        else:
             X_test[col] = X_test[col].fillna(-999)

    logger.info("Predicting...")
    df_test['score'] = model.predict(X_test)
    
    # Rank
    df_test['race_id'] = df_test['race_id'].astype(str)
    df_test['pred_rank'] = df_test.groupby('race_id')['score'].rank(ascending=False, method='first')
    
    # Load Payouts
    payouts = load_payouts(2024)
    if payouts.empty: return

    # Strategies to Test
    strategies = [
        'Umaren_Box5', 
        'Sanrenpuku_Box5', 
        'Umatan_Nagashi_1_to_6', 
        'Sanrentan_Nagashi_1_to_6',
        'Sanrentan_Formation_12_123_12345'
    ]
    
    results = {}
    for s in strategies:
        results[s] = {'bets': 0, 'return': 0, 'hits': 0, 'races': 0}
        
    logger.info("Simulating Strategies...")
    
    # Group by race for efficiency
    grouped = df_test.groupby('race_id')
    payout_map = payouts.set_index('race_id')
    
    for race_id, group in grouped:
        if race_id not in payout_map.index: continue
        
        race_payout = payout_map.loc[race_id]
        
        for s in strategies:
            # Generate Bets
            bets = generate_combinations(group[['horse_number', 'pred_rank']], s)
            
            # Check Hits
            race_return = 0
            hits = 0
            cost = 0
            
            for bet in bets:
                cost += bet['cost']
                b_type = bet['type']
                b_comb = bet['comb']
                
                # Check match against payout 1a
                # Simplification: Only checking 1st payout (1a). 
                # Ideally check 1a, 2a, 3a... usually 1a is main payout.
                # Only Exacta/Quinella/Trio/Trifecta usually have 1 unless dead heat.
                
                pay_comb_col = f"{b_type}_comb"
                pay_amt_col = f"{b_type}_pay"
                
                if b_comb == race_payout[pay_comb_col]:
                    race_return += race_payout[pay_amt_col] * (bet['cost']/100) # payout is for 100 yen
                    hits += 1
            
            results[s]['bets'] += len(bets) # number of tickets
            results[s]['return'] += race_return
            results[s]['hits'] += hits # number of winning tickets
            results[s]['races'] += 1
            
    # Report
    print("\n" + "="*80)
    print(" 🎲 LambdaRank Combination Betting Simulation (JRA 2024)")
    print("="*80)
    print(f"{'Strategy':<35} | {'ROI':<8} | {'HitRate':<8} | {'Profit':<10} | {'Bets/Race':<5}")
    print("-" * 80)
    
    for s in strategies:
        res = results[s]
        n_tickets = res['bets']
        cost = n_tickets * 100
        ret = res['return']
        roi = (ret / cost * 100) if cost > 0 else 0
        profit = ret - cost
        hit_rate = (res['hits'] / res['races'] * 100) if res['races'] > 0 else 0 # Races with at least 1 hit? No this is hits/races ratio.
        # Actually hit rate usually means % of races won. But here we have multiple bets per race.
        # Let's show Profit mainly.
        
        # Calculate hit rate as % of races where return > 0? No, let's just stick to ROI/Profit.
        # Bets/Race
        avg_bets = n_tickets / res['races'] if res['races'] > 0 else 0
        
        print(f"{s:<35} | {roi:.2f}%   | {hit_rate:.2f}     | {profit:,.0f}    | {avg_bets:.1f}")
        
    print("="*80)

if __name__ == "__main__":
    main()
