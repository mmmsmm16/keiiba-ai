
"""
Daily Evaluation Script for Top1 and Top3 EV Strategies
Usage: python scripts/evaluate_daily_results.py --date 20240106
"""
import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants (Match production script)
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def get_db_engine():
    return create_engine(
        f"postgresql://postgres:postgres@host.docker.internal:5433/pckeiba"
    )

def load_payout_and_results(engine, target_date):
    """Load race results and payouts for specific date"""
    year = target_date[:4]
    month_day = target_date[4:]
    
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay
    FROM jvd_hr 
    WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{month_day}'
    """
    df = pd.read_sql(query, engine)
    return df

def get_race_results(engine, race_ids):
    """Get actual rank and horse names"""
    ids_str = "'" + "','".join(race_ids) + "'"
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        umaban, bamei, kakutei_chakujun as rank
    FROM jvd_se
    WHERE kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango IN ({ids_str})
    """
    df = pd.read_sql(query, engine)
    return df

VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYYMMDD")
    args = parser.parse_args()
    target_date = args.date
    
    print("="*60)
    print(f"🏁 Daily Evaluation Report: {target_date}")
    print("="*60)

    # 1. Load Data
    print("Loading data...", flush=True)
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
        
    df_all = pd.read_parquet(DATA_PATH)
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Filter by date
    target_dt = pd.to_datetime(target_date)
    df_today = df_all[df_all['date'] == target_dt].copy()
    
    if df_today.empty:
        print(f"No data found for {target_date} in parquet.")
        return
        
    race_ids = df_today['race_id'].astype(str).unique()
    print(f"Found {len(race_ids)} races, {len(df_today)} horses.")

    # 2. Load Model & Predict
    print("Loading model...", flush=True)
    model = joblib.load(MODEL_PATH)
    required_features = model.feature_name()
    
    # Prepare features
    X = df_today.reindex(columns=required_features).fillna(-999)
    for c in X.columns:
        if X[c].dtype.name == 'category' or X[c].dtype == 'object':
             X[c] = X[c].astype('category').cat.codes
             
    print("Predicting...", flush=True)
    preds = model.predict(X.values.astype(np.float64))
    df_today['pred_prob'] = preds
    df_today['pred_prob_norm'] = df_today.groupby('race_id')['pred_prob'].transform(lambda x: x / x.sum())
    
    # Odds (Use odds_final from parquet if possible, else DB might be needed but parquet is faster)
    if 'odds_final' in df_today.columns:
        df_today['odds'] = df_today['odds_final'].fillna(0.0)
    else:
        df_today['odds'] = 0.0 # Should ideally fetch from DB if missing
        
    df_today['ev'] = df_today['pred_prob_norm'] * df_today['odds']

    # 3. Load Results (Answers)
    engine = get_db_engine()
    df_payout = load_payout_and_results(engine, target_date)
    df_results = get_race_results(engine, race_ids)
    
    # Process Payouts
    payout_map = {} # race_id -> {horse_num: payout}
    for _, row in df_payout.iterrows():
        try:
            rid = row['race_id']
            h = int(row['win_horse'])
            p = int(row['win_pay'])
            if rid not in payout_map: payout_map[rid] = {}
            payout_map[rid][h] = p
        except:
            pass
            
    # Process Names/Ranks
    horse_info_map = {} # (race_id, umaban) -> {name, rank}
    for _, row in df_results.iterrows():
        rid = row['race_id']
        u = int(row['umaban'])
        horse_info_map[(rid, u)] = {'name': row['bamei'], 'rank': row['rank']}

    # 4. Evaluate & ROI
    results_top1 = [] # {race_id, bet, return}
    results_top3_ev = [] # {race_id, bet, return}
    
    # Venue Stats
    venue_stats = {} # venue_code -> {strategy: {bet, return}}

    for rid in sorted(race_ids):
        df_race = df_today[df_today['race_id'] == rid].copy()
        df_race = df_race.sort_values('pred_prob', ascending=False)
        
        venue_code = rid[4:6]
        venue_name = VENUE_MAP.get(venue_code, venue_code)
        race_num = rid[-2:]
        
        # Get Race Payout
        race_payouts = payout_map.get(rid, {})
        
        if not race_payouts:
            # Skip stats if no payout data
            print(f"Skipping {venue_name} {race_num}R (No payout data in DB)")
            continue

        print(f"\n📍 {venue_name} {race_num}R")
        print("-" * 50)
        print(f"{'Rank':<4} {'Horse':<12} {'Prob':>6} {'Odds':>6} {'EV':>5} {'Result':<6}")
        
        # Display Top 5
        top_horses = df_race.head(5)
        
        # Strategy Tracking
        # 1. Top 1 Bet
        top1_horse = df_race.iloc[0]
        h_num = int(top1_horse['horse_number'])
        
        # Log Top 1
        bet_1 = 100
        ret_1 = race_payouts.get(h_num, 0)
        results_top1.append({'bet': bet_1, 'return': ret_1})
        
        if venue_name not in venue_stats:
            venue_stats[venue_name] = {'top1': {'bet':0, 'ret':0}, 'top3_ev': {'bet':0, 'ret':0}}
        venue_stats[venue_name]['top1']['bet'] += bet_1
        venue_stats[venue_name]['top1']['ret'] += ret_1

        # 2. Top 3 EV >= 1.0 Bet
        top3_horses = df_race.head(3)
        ev_bets = []
        
        for i, (_, row) in enumerate(top3_horses.iterrows()):
            hn = int(row['horse_number'])
            info = horse_info_map.get((rid, hn), {'name': '???', 'rank': '-'})
            name = info['name']
            rank = info['rank']
            
            # Print row
            print(f"{i+1:<4} {name:<12} {row['pred_prob_norm']:>6.1%} {row['odds']:>6.1f} {row['ev']:>5.2f} {rank:<6}")
            
            # Strategy B Check
            if row['ev'] >= 1.0: # EV >= 1.0
                bet_ev = 100
                ret_ev = race_payouts.get(hn, 0)
                results_top3_ev.append({'bet': bet_ev, 'return': ret_ev})
                venue_stats[venue_name]['top3_ev']['bet'] += bet_ev
                venue_stats[venue_name]['top3_ev']['ret'] += ret_ev
                ev_bets.append(name)
        
        # Print Strategy Results for this race
        win_1 = "WIN" if ret_1 > 0 else "LOSE"
        print(f"👉 Top1 Bet: {top1_horse['horse_number']} ({win_1}) => {ret_1}")
        
        if ev_bets:
            print(f"👉 EV Bets (Top3 & EV>=1.0): {', '.join(ev_bets)}")
        else:
            print(f"👉 EV Bets: None")

    # 5. Summary
    print("\n" + "="*60)
    print("📊 Daily ROI Summary")
    print("="*60)
    
    # Total Top 1
    t1_bet = sum(r['bet'] for r in results_top1)
    t1_ret = sum(r['return'] for r in results_top1)
    t1_roi = (t1_ret / t1_bet * 100) if t1_bet > 0 else 0
    
    # Total Top 3 EV
    tev_bet = sum(r['bet'] for r in results_top3_ev)
    tev_ret = sum(r['return'] for r in results_top3_ev)
    tev_roi = (tev_ret / tev_bet * 100) if tev_bet > 0 else 0
    
    print(f"【全体】 Top 1 Flat Buy:")
    print(f"  Bet: {t1_bet:,} yen | Return: {t1_ret:,} yen | ROI: {t1_roi:.1f}%")
    
    print(f"\n【全体】 Top 3 (EV>=1.0) Buy:")
    print(f"  Bet: {tev_bet:,} yen | Return: {tev_ret:,} yen | ROI: {tev_roi:.1f}%")
    
    print("\n【会場別詳細】")
    print(f"{'Venue':<10} | {'Top1 ROI':<10} | {'EV(Top3) ROI':<10}")
    print("-" * 40)
    for venue, stats in venue_stats.items():
        s1 = stats['top1']
        sev = stats['top3_ev']
        
        r1 = (s1['ret']/s1['bet']*100) if s1['bet'] > 0 else 0
        rev = (sev['ret']/sev['bet']*100) if sev['bet'] > 0 else 0
        
        print(f"{venue:<10} | {r1:>8.1f}% | {rev:>11.1f}%")

if __name__ == "__main__":
    main()
