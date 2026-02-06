
import pandas as pd
import numpy as np
import os
import argparse

def generate_monthly_detail(year, month, output_path):
    # Paths
    ledger_path = f'reports/phase12/ledgers/ledger_{year}_{month}.csv'
    oof_path = 'data/predictions/v13_oof_2024_2025_with_odds_features.parquet'
    
    if not os.path.exists(ledger_path):
        print(f"Ledger not found: {ledger_path}")
        return
        
    print(f"Loading ledger: {ledger_path}")
    df_bets = pd.read_csv(ledger_path)
    
    print(f"Loading OOF: {oof_path}")
    df_oof = pd.read_parquet(oof_path)
    
    # Filter OOF for the month
    df_oof['date'] = pd.to_datetime(df_oof['date'])
    df_m = df_oof[(df_oof['date'].dt.year == year) & (df_oof['date'].dt.month == month)]
    
    if df_m.empty:
        print("No OOF data for this month.")
        return
        
    # Prepare Output
    lines = []
    lines.append(f"# {year}年{month}月 詳細レポート")
    lines.append("各レースの予測上位、購入馬券、収支結果一覧\n")
    

    # Load Final Odds for Payout Calc
    print(f"Loading Odds for {year}...")
    odd_path = f"data/odds_snapshots/{year}/odds_final.parquet"
    if os.path.exists(odd_path):
        odds_df = pd.read_parquet(odd_path)
        odds_df['race_id'] = odds_df['race_id'].astype(str)
        odds_df['combination'] = odds_df['combination'].astype(str)
        # Unique map: (race_id, ticket_type, combination) -> odds
        odds_df = odds_df.drop_duplicates(subset=['race_id', 'ticket_type', 'combination'], keep='last')
        payout_map = dict(zip(zip(odds_df['race_id'], odds_df['ticket_type'], odds_df['combination']), odds_df['odds']))
    else:
        print("Odds file not found. Payouts will be 0.")
        payout_map = {}

    # Prepare Rank Map (from OOF) for Hit Check
    print("Preparing Rank Map...")
    rank_map = {} 
    # Also need Horse Name? OOF doesn't have it.
    # Preprocessed data has it? 'data/preprocessed_data_v11.parquet'?
    # It takes time to load big file. 
    # For now, stick to Horse Number.
    
    for rid, grp in df_m.groupby('race_id'):
        rank_map[str(rid)] = dict(zip(grp['horse_number'], grp['rank']))

    dates = sorted(df_m['date'].unique())
    total_invest_month = 0
    total_return_month = 0

    for d in dates:
        d_str = pd.Timestamp(d).strftime('%Y-%m-%d')
        lines.append(f"## {d_str}")
        
        # Races
        df_d = df_m[df_m['date'] == d]
        race_ids = sorted(df_d['race_id'].unique())
        
        daily_invest = 0
        daily_return = 0
        
        for rid in race_ids:
            rid_str = str(rid)
            df_r = df_d[df_d['race_id'] == rid].sort_values('pred_prob', ascending=False)
            
            # Race Header
            bets_r = df_bets[df_bets['race_id'].astype(str) == rid_str]
            
            r_invest = bets_r['amount'].sum() if not bets_r.empty else 0
            r_return = 0
            
            bet_details = []
            if not bets_r.empty:
                for _, row in bets_r.iterrows():
                    ttype = row['ticket_type']
                    combo = str(row['combination'])
                    amt = row['amount']
                    
                    # Check Hit
                    h2r = rank_map.get(rid_str, {})
                    hit = False
                    if ttype == 'win':
                        if h2r.get(int(combo), 99) == 1: hit = True
                    elif ttype == 'place':
                         if h2r.get(int(combo), 99) <= 3: hit = True
                    elif ttype == 'umaren':
                        parts = [int(x) for x in combo.split('-')]
                        if h2r.get(parts[0], 99) <= 2 and h2r.get(parts[1], 99) <= 2: hit = True
                    
                    payout = 0
                    actual_odds = 0.0
                    if hit:
                        actual_odds = payout_map.get((rid_str, ttype, combo), 0.0)
                        if pd.isna(actual_odds): actual_odds = 0.0
                        payout = int(amt * actual_odds)
                    
                    status = "WIN" if hit else "LOSE"
                    r_return += payout
                    
                    bet_details.append(f"- [{status}] {ttype} {combo}: {amt}円 -> {payout}円 (Odds: {actual_odds})")
            
            daily_invest += r_invest
            daily_return += r_return
            
            profit = r_return - r_invest
            roi = (r_return / r_invest * 100) if r_invest > 0 else 0
            
            # Formatting Race Block
            lines.append(f"### Race {rid_str} | Invest: {r_invest:,} | Return: {r_return:,} | Profit: {profit:,} ({roi:.1f}%)")
            
            # Top 3 Predictions
            lines.append("#### Predictions (Top 3)")
            lines.append("| No. | Prob | Rank | Odds |")
            lines.append("|:---:|:---:|:---:|:---:|")
            for _, row in df_r.head(3).iterrows():
                h = row['horse_number']
                p = row['pred_prob']
                r = row['rank']
                o = row['odds']
                lines.append(f"| {h} | {p:.4f} | {r} | {o} |")
            
            lines.append("")
            
            # Bets
            if bet_details:
                lines.append("#### Bets")
                for b in bet_details:
                    lines.append(b)
            else:
                lines.append("No Bets Placed.")
                
            lines.append("")
            
        lines.append(f"**Day Total:** Invest {daily_invest:,} | Return {daily_return:,} | Profit {daily_return - daily_invest:,}")
        lines.append("---")
        
        total_invest_month += daily_invest
        total_return_month += daily_return
        
    lines.append(f"# Month Total")
    lines.append(f"Invest: {total_invest_month:,}")
    lines.append(f"Return: {total_return_month:,}")
    lines.append(f"Profit: {total_return_month - total_invest_month:,}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--month', type=int, default=1)
    parser.add_argument('--output', type=str, default='reports/phase12/monthly_detail_report.md')
    args = parser.parse_args()
    
    generate_monthly_detail(args.year, args.month, args.output)
