
import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def load_final_odds(year):
    path = f"data/odds_snapshots/{year}/odds_final.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

def analyze_results():
    # 1. Load Ledgers
    files = glob.glob('reports/phase12/ledgers/ledger_2025_*.csv')
    if not files:
        print("No ledgers found.")
        return
        
    print(f"Loading {len(files)} ledgers...")
    df_bets = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Total Bets: {len(df_bets)}")
    
    # 2. Load Odds & Truth
    print("Loading Truth Data...")
    year = 2025
    odds_df = load_final_odds(year)
    odds_df['race_id'] = odds_df['race_id'].astype(str)
    odds_df['combination'] = odds_df['combination'].astype(str)
    
    # Dedupe odds
    odds_df = odds_df.drop_duplicates(subset=['race_id', 'ticket_type', 'combination'], keep='last')
    
    # Map OK
    payout_map = dict(zip(zip(odds_df['race_id'], odds_df['ticket_type'], odds_df['combination']), odds_df['odds']))
    
    # Rank Map for Hit Check
    oof_path = 'data/predictions/v13_oof_2024_2025_with_odds_features.parquet'
    df_oof = pd.read_parquet(oof_path)
    df_oof['race_id'] = df_oof['race_id'].astype(str)
    
    rank_map = {}
    rids = df_bets['race_id'].astype(str).unique()
    df_sub = df_oof[df_oof['race_id'].isin(rids)]
    
    for rid, grp in df_sub.groupby('race_id'):
        rank_map[rid] = dict(zip(grp['horse_number'], grp['rank']))
        
    # 3. Calculate Results Row-by-Row
    print("Calculating Payouts...")
    results = []
    
    for idx, row in df_bets.iterrows():
        rid = str(row['race_id'])
        ttype = row['ticket_type']
        combo = str(row['combination'])
        amt = row['amount']
        
        # Hit Logic
        h2r = rank_map.get(rid, {})
        hit = False
        if ttype == 'win':
            if h2r.get(int(combo), 99) == 1: hit = True
        elif ttype == 'place':
            if h2r.get(int(combo), 99) <= 3: hit = True # Simplified Top 3
        elif ttype == 'umaren':
            parts = [int(x) for x in combo.split('-')]
            if h2r.get(parts[0], 99) <= 2 and h2r.get(parts[1], 99) <= 2: hit = True
            
        final_odds = payout_map.get((rid, ttype, combo), 0.0)
        payout = 0
        if hit:
            payout = amt * final_odds
            
        results.append({
            'race_id': rid,
            'ticket_type': ttype,
            'combination': combo,
            'amount': amt,
            'odds': final_odds,
            'payout': payout,
            'hit': hit
        })
        
    df_res = pd.DataFrame(results)
    
    # 4. Analysis Scenarios
    
    # Base
    base_cost = df_res['amount'].sum()
    base_rev = df_res['payout'].sum()
    base_roi = base_rev / base_cost if base_cost > 0 else 0
    print(f"Base ROI: {base_roi*100:.2f}% (Cost {base_cost:,.0f}, Rev {base_rev:,.0f})")
    
    # Slippage (0.95, 0.90)
    for slip in [0.95, 0.90]:
        rev_slip = (df_res['payout'] * slip).sum() # Simplified: apply to total payout
        roi_slip = rev_slip / base_cost
        print(f"Slippage {slip:.2f} ROI: {roi_slip*100:.2f}%")
        
    # Outlier Analysis (Remove Top 1% Wins)
    df_wins = df_res[df_res['payout'] > 0].sort_values('payout', ascending=False)
    n_remove = int(len(df_wins) * 0.01)
    df_cut = df_res.drop(df_wins.head(n_remove).index)
    
    cut_rev = df_cut['payout'].sum()
    cut_cost = df_cut['amount'].sum() # Same cost (we placed bets, just didn't get big wins? No, removing bets entirely? 
    # Usually "Remove Top 1% Payouts" means "Assuming those big wins didn't happen but we paid for them".
    # So Cost stays same, Revenue drops.
    cut_roi = cut_rev / base_cost
    print(f"Remove Top 1% Winners ROI: {cut_roi*100:.2f}% (Rev {cut_rev:,.0f})")
    
    # 5. Monthly Stats
    # Need date. map race_id -> date via OOF?
    # OOF has date.
    rid_date = df_sub[['race_id', 'date']].drop_duplicates().set_index('race_id')['date'].to_dict()
    df_res['date'] = df_res['race_id'].map(rid_date)
    df_res['month'] = pd.to_datetime(df_res['date']).dt.to_period('M')
    
    monthly = df_res.groupby('month').agg({'amount': 'sum', 'payout': 'sum'})
    monthly['roi'] = monthly['payout'] / monthly['amount']
    print("\nMonthly Stats:")
    print(monthly)
    
    # Save
    df_res.to_csv('reports/phase12/final_ledger_2025_analyzed.csv', index=False)
    monthly.to_csv('reports/phase12/final_monthly_2025.csv')

if __name__ == "__main__":
    analyze_results()
