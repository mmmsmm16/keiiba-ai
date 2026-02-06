import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioSimulator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(['date', 'race_id']).copy()
        
    def run_portfolio(self, core_params: dict, satellite_params: dict):
        """
        Run simulation with Core-First priority.
        """
        history = []
        
        # Pre-calculate conditions to speed up
        # Core
        df = self.df
        df['core_ev'] = df['p_cal'] * df['odds_win_pre']
        core_mask = (df['rank_pred'] == 1) & \
                    (df['p1'] >= core_params['t1']) & \
                    (df['margin'] >= core_params['t2']) & \
                    (df['core_ev'] >= core_params['e1']) & \
                    (df['entropy'] <= core_params['en1']) & \
                    (df['field_size'] <= core_params['s1'])
                    
        # Satellite
        sat_mask = (df['rank_pred'] == 1) & \
                   (df['p1'] >= satellite_params['t1']) & \
                   (df['margin'] >= satellite_params['t2']) & \
                   (df['entropy'] <= satellite_params['en1']) & \
                   (df['field_size'] <= satellite_params['s1'])
                   
        # Iterate over unique race_ids (since rank_pred=1 is unique per race)
        # However, to be safe, filtering by rank_pred=1 implies 1 row per race max
        
        # subset of rows where at least one strategy might trigger
        candidates = df[core_mask | sat_mask].copy()
        
        # Determine action per row
        candidates['is_core'] = core_mask[candidates.index]
        candidates['is_sat'] = sat_mask[candidates.index]
        
        results = []
        
        for _, row in candidates.iterrows():
            bet_amount = 100 # Unit bet
            payout = 0
            strategy_name = ""
            
            if row['is_core']:
                # Core Strategy (Win)
                strategy_name = "Core"
                payout = (row['payout_win'] / 100.0) * bet_amount
                
            elif row['is_sat']: # Only if NOT core
                # Satellite Strategy (Place)
                strategy_name = "Satellite"
                payout = (row['payout_place'] / 100.0) * bet_amount
            
            results.append({
                'race_id': row['race_id'],
                'date': row['date'],
                'year': row['year_valid'],
                'strategy': strategy_name,
                'bet': bet_amount,
                'return': payout
            })
            
        res_df = pd.DataFrame(results)
        if res_df.empty: return None
        
        # Aggregate
        total_bet = res_df['bet'].sum()
        total_ret = res_df['return'].sum()
        roi = total_ret / total_bet
        
        # Yearly
        yearly = res_df.groupby('year').agg({'bet': 'sum', 'return': 'sum', 'strategy': 'count'})
        yearly['roi'] = yearly['return'] / yearly['bet']
        
        # Drawdown
        res_df['net'] = res_df['return'] - res_df['bet']
        res_df['cumsum'] = res_df['net'].cumsum()
        res_df['running_max'] = res_df['cumsum'].cummax()
        res_df['drawdown'] = res_df['running_max'] - res_df['cumsum']
        # Note: This is absolute DD, not percentage, since we are using fixed bets.
        # For %DD, we need capital simulation, but absolute gives an idea.
        max_dd_abs = res_df['drawdown'].max()
        
        return {
            'total_roi': roi,
            'total_bets': len(res_df),
            'max_dd_abs': max_dd_abs,
            'yearly_stats': yearly,
            'core_count': res_df[res_df['strategy'] == 'Core'].shape[0],
            'sat_count': res_df[res_df['strategy'] == 'Satellite'].shape[0],
            'df': res_df
        }

def run():
    input_path = "reports/simulations/v13_e1_enriched_2022_2024.parquet"
    df = pd.read_parquet(input_path)
    sim = PortfolioSimulator(df)
    
    # Core Params (Based on previous high-EV findings: e1=3.0, t1=0.6)
    # Using e1=2.6 to get slightly more bets (from verify), or stick to 3.0?
    # User said: "p1>=0.6 & EV>=3.0 is ROI>100% but bets few" -> relax slightly?
    # Let's try matching the 116% strategy: p1=0.6, t2=0.1, e1=3.0
    core_params = {
        't1': 0.6, 't2': 0.1, 'e1': 3.0, 'en1': 99.0, 's1': 18
    }
    
    # Satellite Params (Stable Place: t1=0.6, t2=0.1 from H4 results)
    sat_params = {
        't1': 0.6, 't2': 0.1, 'en1': 99.0, 's1': 18
    }
    
    res = sim.run_portfolio(core_params, sat_params)
    
    if res:
        print("\n--- Portfolio Simulation Results (Core + Satellite) ---")
        print(f"Total ROI: {res['total_roi']*100:.2f}%")
        print(f"Total Bets: {res['total_bets']} (Core: {res['core_count']}, Satellite: {res['sat_count']})")
        print(f"Max Drawdown (Abs): {res['max_dd_abs']:.0f} JPY (unit=100)")
        print("\nYearly Stats:")
        print(res['yearly_stats'][['strategy', 'roi']])
        
        # Check constraints
        min_roi = res['yearly_stats']['roi'].min()
        print(f"\nMin Yearly ROI: {min_roi*100:.2f}%")
        if min_roi >= 0.95 and res['total_bets'] >= 500:
            print("SUCCESS: Portfolio meets practical constraints!")
        else:
            print("WARNING: Constraints not fully met.")
            
if __name__ == "__main__":
    run()
