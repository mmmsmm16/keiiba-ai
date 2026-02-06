
import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.scripts.simulate_auto_predict_2025 import SimulationPredictor, get_2025_dates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Top1WinPredictor(SimulationPredictor):
    def run_top1_simulation(self):
        logger.info(f"Starting Top1 Win Simulation for {len(self.target_dates)} dates in 2025...")
        
        total_bet = 0
        total_return = 0
        total_races = 0
        hits = 0
        
        results = []

        import warnings
        warnings.simplefilter('ignore')

        for date in tqdm(self.target_dates):
            self.target_date = date
            try:
                target_date_str = date.strftime('%Y%m%d')
                raw_df = self.loader.load(target_date=target_date_str)
                if raw_df is None or raw_df.empty:
                    continue
                
                # JRA Filtering (01-10)
                raw_df['place_code'] = raw_df['race_id'].astype(str).str[4:6].astype(int)
                raw_df = raw_df[raw_df['place_code'] <= 10]
                if raw_df.empty: continue
                
                # Preprocess & Predict
                X, ids = self.preprocessor.preprocess(raw_df, history_df=self.history_df)
                
                # Get expected features from LGBM model
                expected_features = self.lgbm.feature_name()
                
                if expected_features:
                    missing = set(expected_features) - set(X.columns)
                    if missing:
                        for col in missing: X[col] = 0.0
                    X = X[[c for c in expected_features if c in X.columns]]
                
                scores = self.model.predict(X)
                result_df = ids.copy()
                result_df['score'] = scores
                result_df['rank'] = raw_df.reset_index(drop=True)['rank'] # Assuming aligned order? 
                # Careful: ids might come from preprocess sorting? 
                # ids is just race_id, horse_number.
                # We should merge 'rank' and 'odds' from raw_df back to result_df
                
                # Merge rank/odds from raw_df
                # raw_df has 'race_id', 'horse_number', 'rank', 'odds'
                # Ensure types
                raw_df['race_id'] = raw_df['race_id'].astype(str)
                raw_df['horse_number'] = raw_df['horse_number'].astype(str) # In loader often int?
                # Loader output: horse_number is int or float?
                # Check extract_rows usage in preprocess...
                
                # Let's check result_df columns. It usually has 'odds' from feature engineering?
                # But we want FINAL odds for PnL. raw_df has final odds.
                
                # Simplify: just merge on race_id, horse_number
                raw_subset = raw_df[['race_id', 'horse_number', 'rank', 'odds']].copy()
                raw_subset['horse_number'] = raw_subset['horse_number'].astype(int)
                result_df['horse_number'] = result_df['horse_number'].astype(int)
                
                merged = pd.merge(result_df, raw_subset, on=['race_id', 'horse_number'], how='left', suffixes=('', '_final'))
                # If odds exists in result_df, it might be T-10 or Final depending on mode.
                # We use odds_final for PnL.
                
                for rid, grp in merged.groupby('race_id'):
                    # Pick Top 1
                    top1 = grp.loc[grp['score'].idxmax()]
                    
                    bet = 100
                    ret = 0
                    
                    # Check Hit
                    # rank might be NaN if race cancelled/not finished?
                    try:
                        rank = int(top1['rank']) if not pd.isna(top1['rank']) else 999
                    except:
                        rank = 999
                        
                    if rank == 1:
                        # Win
                        odds = float(top1['odds_final']) if not pd.isna(top1['odds_final']) else 1.0 # Fallback 1.0? No, assume valid.
                        ret = int(odds * 100)
                        hits += 1
                    
                    total_bet += bet
                    total_return += ret
                    total_races += 1
                    
                    results.append({
                        'date': date,
                        'race_id': rid,
                        'bet': bet,
                        'return': ret
                    })

            except Exception as e:
                logger.error(f"Error {date}: {e}")

        roi = total_return / total_bet * 100 if total_bet > 0 else 0
        hit_rate = hits / total_races * 100 if total_races > 0 else 0
        
        print("\n=== Top 1 Win (Tansho) Simulation Results (2025) ===")
        print(f"Total Races: {total_races}")
        print(f"Total Hits: {hits} ({hit_rate:.1f}%)")
        print(f"Total Bet: {total_bet:,} yen")
        print(f"Total Return: {total_return:,} yen")
        print(f"ROI: {roi:.2f}%")
        
        # Also simple print for user
        logger.info(f"Top1 Win ROI: {roi:.2f}% (Hit: {hit_rate:.1f}%)")

if __name__ == '__main__':
    dates = get_2025_dates()
    # Filter for Jan 2025 only to speed up basic check
    dates = [d for d in dates if d.month == 1]
    
    sim = Top1WinPredictor(dates)
    sim.run_top1_simulation()
