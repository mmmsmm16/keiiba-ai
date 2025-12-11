import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from tqdm import tqdm
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.scripts.auto_predict import AutoPredictor
from src.simulation.simulator import BettingSimulator
from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress other loggers
logging.getLogger('src.inference.loader').setLevel(logging.WARNING)
logging.getLogger('src.inference.preprocessor').setLevel(logging.WARNING)
logging.getLogger('src.scripts.auto_predict').setLevel(logging.WARNING)

def get_2025_dates():
    try:
        path = 'data/processed/preprocessed_data.parquet'
        if not os.path.exists(path):
            return []
        dates = pd.read_parquet(path, columns=['date'])['date'].unique()
        dates = [d for d in dates if str(d).startswith('2025')]
        return sorted(dates)
    except:
        return []

class SimulationPredictor(AutoPredictor):
    def __init__(self, target_dates):
        super().__init__(dry_run=True, target_date=None)
        self.target_dates = target_dates
        self.results = []
        self.simulator = BettingSimulator()
        
        # Load history once for speed
        logger.info("Loading history data for optimization...")
        preprocessor = InferencePreprocessor()
        if os.path.exists(preprocessor.history_path):
             self.history_df = pd.read_parquet(preprocessor.history_path)
             logger.info(f"History loaded: {len(self.history_df)} rows")
        else:
             self.history_df = None
             logger.warning("History file not found.")
             
        self.test_mode = False

    def run_simulation(self):
        logger.info(f"Starting simulation for {len(self.target_dates)} dates in 2025...")
        
        total_bet = 0
        total_return = 0
        
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
                # race_id format: YYYY(0-4) Place(4-6)
                # Filter by place code
                raw_df['place_code'] = raw_df['race_id'].astype(str).str[4:6].astype(int)
                raw_df = raw_df[raw_df['place_code'] <= 10]
                
                if raw_df.empty:
                    logger.info(f"Skipping {target_date_str}: No JRA races found.")
                    continue
                
                race_ids = raw_df['race_id'].unique().tolist()
                self.simulator.fetch_payouts(race_ids)
                logging.info(f"Fetched payouts for {len(self.simulator.payout_map)} races (Total accumulated)")
                if race_ids[0] not in self.simulator.payout_map:
                    logging.warning(f"Payout data MISSING for sample race: {race_ids[0]}")
                    # Debug Query Check?

                X, ids = self.preprocessor.preprocess(raw_df, history_df=self.history_df)
                
                if self.expected_features:
                    missing = set(self.expected_features) - set(X.columns)
                    if missing:
                        for col in missing: X[col] = 0.0
                    X = X[[c for c in self.expected_features if c in X.columns]]
                
                scores = self.model.predict(X)
                
                result_df = ids.copy()
                result_df['score'] = scores
                
                from scipy.special import softmax
                SOFTMAX_TEMPERATURE = 3.0
                result_df['prob'] = result_df.groupby('race_id')['score'].transform(
                    lambda x: softmax(x / SOFTMAX_TEMPERATURE)
                )
                
                if self.calibrator:
                    result_df['calibrated_prob'] = self.calibrator.predict(result_df['prob'].values)
                else:
                    result_df['calibrated_prob'] = result_df['prob']
                    
                race_sums = result_df.groupby('race_id')['calibrated_prob'].transform('sum')
                result_df['calibrated_prob'] = result_df['calibrated_prob'] / race_sums
                
                result_df['odds'] = result_df['odds'].replace(0, 1.0)
                result_df['expected_value'] = result_df['calibrated_prob'] * result_df['odds']
                
                processed_count = 0
                
                for race_id, df in result_df.groupby('race_id'):
                    bet_data = self._calculate_betting_data_sim(df)
                    
                    if not bet_data['bets']:
                        continue
                        
                    pay_data = self.simulator.payout_map.get(race_id, {})
                    race_bet_amount = 0
                    race_return_amount = 0
                    
                    for bet in bet_data['bets']:
                        b_type = bet['type']
                        pts = bet['points']
                        axis = bet['axis']
                        partners = bet['partners']
                        cost = pts * 100
                        race_bet_amount += cost
                        
                        import itertools
                        tickets = []
                        if b_type == 'sanrenpuku':
                            for p_pair in itertools.combinations(partners, 2):
                                combo = tuple(sorted([axis[0], p_pair[0], p_pair[1]]))
                                tickets.append(combo)
                        elif b_type.startswith('sanrentan'):
                            for p1 in partners:
                                for p2 in partners:
                                    if p1 == p2: continue
                                    tickets.append((axis[0], p1, p2))
                                    
                        for t in tickets:
                            key = ""
                            val = 0
                            if b_type == 'sanrenpuku':
                                key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                                if 'sanrenpuku' in pay_data and key in pay_data['sanrenpuku']:
                                    val = pay_data['sanrenpuku'][key]
                            elif b_type.startswith('sanrentan'):
                                key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                                if 'sanrentan' in pay_data and key in pay_data['sanrentan']:
                                    val = pay_data['sanrentan'][key]
                            if val > 0:
                                race_return_amount += val / 100.0 * 100

                    self.results.append({
                        'date': date,
                        'race_id': race_id,
                        'strategy': bet_data['strategy_type'],
                        'is_strong': bet_data['is_strong'],
                        'bet': race_bet_amount,
                        'return': race_return_amount,
                        'profit': race_return_amount - race_bet_amount
                    })
                    
                    total_bet += race_bet_amount
                    total_return += race_return_amount
                    processed_count += 1
            
                if self.test_mode and processed_count > 0:
                    logger.info(f"Test Mode: Processed {processed_count} races for {target_date_str}. Stopping.")
                    break
            
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        df_res = pd.DataFrame(self.results)
        if df_res.empty:
            logger.warning("No bets placed.")
            return

        logger.info("\n=== Simulation Summary (2025 Auto-Predict) ===")
        logger.info(f"Total Bets: {df_res['bet'].sum():,.0f} yen")
        logger.info(f"Total Return: {df_res['return'].sum():,.0f} yen")
        roi = df_res['return'].sum() / df_res['bet'].sum() * 100 if df_res['bet'].sum() > 0 else 0
        logger.info(f"Total ROI: {roi:.2f}%")
        
        logger.info("\n--- By Strategy ---")
        summary = df_res.groupby(['strategy', 'is_strong']).agg({
            'race_id': 'count',
            'bet': 'sum',
            'return': 'sum'
        })
        summary['roi'] = summary['return'] / summary['bet'] * 100
        print(summary)
        
        df_res.to_csv('reports/simulation_2025_autopredict.csv', index=False)
        logger.info("Saved detail report to reports/simulation_2025_autopredict.csv")

    def _calculate_betting_data_sim(self, df):
        """
        シミュレーション用ベッティングロジック
        Source: src/api/routers/predictions.py (Dashboard 2.0 API)
        
        Strategy (Based on Expected Value):
        1. High Value (EV >= 1.2):
           - Type: 三連複 (Sanrenpuku)
           - Formation: Top1 + Top2-4 Box (4 horses -> 4 tickets)
             * Actually code says: combinations([axis] + opps[:3], 3)
             * Note: axis is included in all combinations? Yes, because we filter `if axis in t`.
        
        2. Mid Value (0.8 <= EV < 1.2):
           - Type: 三連単 (Sanrentan)
           - Formation: 1-Axis (Top1) -> Top2-4 (6 tickets)
           
        3. Low Value (EV < 0.8):
           - Skip
        """
        sorted_df = df.sort_values('score', ascending=False)
        if len(sorted_df) < 5:
            return {"bets": [], "strategy_type": "Skip(LowCount)", "is_strong": False}
            
        top1 = sorted_df.iloc[0]
        
        # Calculate EV 
        # (Note: In run_simulation loop, we already calculate 'expected_value' based on 'odds')
        ev = float(top1.get('expected_value', 0))
        
        # Strategy Data Init
        strategy_data = {
            "top1": top1,
            "ev": ev,
            "bets": [],
            "strategy_type": "None",
            "is_strong": False
        }
        
        # Logic from src/api/routers/predictions.py
        h_num = int(top1['horse_number'])
        
        if ev >= 1.2:
            strategy_data["strategy_type"] = "HighValue(EV>=1.2)"
            strategy_data["is_strong"] = True
            
            # Opponents: Rank 2, 3, 4 (3 horses)
            opps_df = sorted_df.iloc[1:4]
            opps = [int(x) for x in opps_df['horse_number']]
            
            # Sanrenpuku 1-Axis to 3 Opps -> 3 tickets?
            # Original code: list(combinations([axis] + opps[:3], 3)) -> then fitler `if axis in t`
            # If we have Axis + 3 Opps = 4 horses. Combinations(4,3) = 4 tickets.
            # (Axis, O1, O2), (Axis, O1, O3), (Axis, O2, O3), (O1, O2, O3)
            # Filter `if axis in t` reduces this to 3 tickets:
            # (Axis, O1, O2), (Axis, O1, O3), (Axis, O2, O3)
            # So this is "Sanrenpuku 1-Axis Stream (Nagashi) to 3 partners" = 3 points.
            
            strategy_data["bets"].append({
                "type": "sanrenpuku",
                "axis": [h_num],
                "partners": opps,
                "points": 3
            })
            
        elif ev >= 0.8:
            strategy_data["strategy_type"] = "MidValue(EV>=0.8)"
            strategy_data["is_strong"] = True
            
            # Opponents: Rank 2, 3, 4 (3 horses)
            opps_df = sorted_df.iloc[1:4]
            opps = [int(x) for x in opps_df['horse_number']]
            
            # Sanrentan 1-Axis -> 3 Opps (6 points)
            strategy_data["bets"].append({
                "type": "sanrentan_1fix",
                "axis": [h_num],
                "partners": opps,
                "points": 6
            })
            
        else:
            strategy_data["strategy_type"] = "Skip(LowEV)"
            strategy_data["is_strong"] = False
            
        return strategy_data

if __name__ == '__main__':
    dates = get_2025_dates()
    if not dates:
        logger.error("No 2025 dates found in preprocessed data. Using fallback...")
        dates = []

    logger.info(f"Found {len(dates)} dates in 2025.")
    
    # --- JRA Simulation Mode ---
    # User Request: Check 1 day first, exclude NRA
    TEST_ONE_DAY = False # Full run
    
    logger.info(f"Running simulation with TEST_ONE_DAY={TEST_ONE_DAY}")
    
    # Pass full dates list, simulator will find the first valid JRA day
    sim = SimulationPredictor(dates)
    sim.test_mode = TEST_ONE_DAY
    sim.run_simulation()
    
    if sim.results:
        print("\n--- First 5 Bets Detail ---")
        for res in sim.results[:5]:
            print(f"Date: {res['date']}, Race: {res['race_id']}, St: {res['strategy']}, Strong: {res['is_strong']}, Bet: {res['bet']}, Rtn: {res['return']}")
