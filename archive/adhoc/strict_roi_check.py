
import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
from scipy.special import softmax

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import after path fix
try:
    from src.inference.loader import InferenceDataLoader
    from src.inference.preprocessor import InferencePreprocessor
    from src.scripts.auto_predict import AutoPredictor
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

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

class StrictROIChecker(AutoPredictor):
    def __init__(self, target_dates):
        # Initialize AutoPredictor base
        super().__init__(dry_run=True, target_date=None)
        self.target_dates = target_dates
        self.results = []
        
        # Load history manually to ensure it exists for preprocess
        if os.path.exists(self.preprocessor.history_path):
             self.history_df = pd.read_parquet(self.preprocessor.history_path)
             logger.info(f"History loaded: {len(self.history_df)} rows")
        else:
             self.history_df = None
             logger.warning("History file not found.")

    def run_check(self):
        logger.info(f"Starting Strict ROI Check for {len(self.target_dates)} dates...")
        
        stats = {
            'model_top1': {'bet': 0, 'return': 0, 'hits': 0, 'races': 0},
            'pop_top1': {'bet': 0, 'return': 0, 'hits': 0, 'races': 0},
            # 'model_ev_policy': ... (Complex, skip for now or implement if needed)
        }
        
        # Get expected features from Model
        expected_features = self.lgbm.feature_name()

        for date in tqdm(self.target_dates):
            date_str = date.strftime('%Y%m%d')
            try:
                # 1. Load Data
                raw_df = self.loader.load(target_date=date_str)
                if raw_df is None or raw_df.empty:
                    continue
                
                # Filter JRA (01-10)
                # Check race_id format. Typically 'YYYYMMDDVJRR' (Year, Month, Day, Venue, ..?)
                # Actually standard format in this project: YYYY(4) + Venue(2) + Kai(2) + Nichi(2) + Race(2) = 12 digits?
                # Or YYYY(4) + Venue(2) + ...
                # Let's rely on 'venue' column if available, or parse race_id carefully.
                # loader.load usually returns 'venue' column.
                if 'venue' in raw_df.columns:
                    raw_df['venue_int'] = raw_df['venue'].astype(int)
                    raw_df = raw_df[raw_df['venue_int'] <= 10]
                else:
                    # Fallback parse
                    raw_df['place_code'] = raw_df['race_id'].astype(str).str[4:6].astype(int)
                    raw_df = raw_df[raw_df['place_code'] <= 10]
                    
                if raw_df.empty: continue

                # 2. Preprocess
                X, ids = self.preprocessor.preprocess(raw_df, history_df=self.history_df)
                
                # Align Features
                if expected_features:
                    current_cols = set(X.columns)
                    missing = set(expected_features) - current_cols
                    if missing:
                        for c in missing: X[c] = 0.0
                    X = X[[c for c in expected_features if c in X.columns]]
                
                # 3. Predict (Ensemble)
                p1 = self.lgbm.predict(X)
                p2 = self.catboost.predict(X)
                scores = self.meta.predict(np.column_stack([p1, p2]))
                
                # 4. Merge Results
                pred_df = ids.copy() # race_id, horse_number
                pred_df['score'] = scores
                
                # Ensure types for merge
                pred_df['race_id'] = pred_df['race_id'].astype(str)
                pred_df['horse_number'] = pred_df['horse_number'].astype(int)
                
                raw_subset = raw_df[['race_id', 'horse_number', 'rank', 'odds', 'popularity']].copy()
                raw_subset['race_id'] = raw_subset['race_id'].astype(str)
                raw_subset['horse_number'] = raw_subset['horse_number'].astype(int)
                
                merged = pd.merge(pred_df, raw_subset, on=['race_id', 'horse_number'], how='inner')
                print(f"[DEBUG] Processing {date_str}: Merged {len(merged)} rows")
                
                # 5. Calculate Metrics per Race
                grouped = merged.groupby('race_id')
                print(f"[DEBUG] Found {len(grouped)} races")
                
                for rid, race_grp in grouped:
                    try:
                        # --- Model Top 1 ---
                        # race_grp['score'] must be float
                        race_grp['score'] = race_grp['score'].astype(float)
                        model_top1 = race_grp.loc[race_grp['score'].idxmax()]
                        
                        stats['model_top1']['bet'] += 100
                        stats['model_top1']['races'] += 1
                        
                        rank = model_top1['rank']
                        try: rank = int(rank)
                        except: rank = 999
                        
                        if rank == 1:
                            odds = float(model_top1['odds']) if not pd.isna(model_top1['odds']) else 1.0
                            stats['model_top1']['return'] += int(odds * 100)
                            stats['model_top1']['hits'] += 1

                        # --- Popularity Top 1 ---
                        race_grp['popularity'] = pd.to_numeric(race_grp['popularity'], errors='coerce').fillna(999)
                        pop_top1 = race_grp.loc[race_grp['popularity'].idxmin()]
                        
                        stats['pop_top1']['bet'] += 100
                        stats['pop_top1']['races'] += 1
                        
                        pop_rank = pop_top1['rank']
                        try: pop_rank = int(pop_rank)
                        except: pop_rank = 999
                        
                        if pop_rank == 1:
                            pop_odds = float(pop_top1['odds']) if not pd.isna(pop_top1['odds']) else 1.0
                            stats['pop_top1']['return'] += int(pop_odds * 100)
                            stats['pop_top1']['hits'] += 1
                    except Exception as loop_e:
                        print(f"Error in race loop {rid}: {loop_e}")

            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    dates = [pd.Timestamp('2025-01-05')]
    checker = StrictROIChecker(dates)
    checker.run_check()
