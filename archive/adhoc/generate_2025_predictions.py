import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from sqlalchemy import text
from tqdm import tqdm
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor
from src.model.ensemble import EnsembleModel
from src.model.lgbm import KeibaLGBM
# from model.catboost_model import KeibaCatBoost
# from model.tabnet_model import KeibaTabNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../../config/config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_2025_dates(loader):
    """Fetch all race dates in 2025."""
    try:
        query = text("SELECT DISTINCT kaisai_nen, kaisai_tsukihi FROM jvd_ra WHERE kaisai_nen = '2025' ORDER BY 1, 2")
        with loader.engine.connect() as conn:
            result = conn.execute(query)
            dates = []
            for row in result:
                nen = row[0]
                md = row[1]
                dt_str = f"{nen}-{md[:2]}-{md[2:]}"
                dates.append(dt_str)
            return dates
    except Exception as e:
        logger.error(f"Failed to fetch dates: {e}")
        return []

def main():
    config = load_config()
    model_type = config['betting'].get('model_type', 'ensemble')
    model_version = config['betting'].get('model_version', 'v4_2025')
    
    logger.info(f"Target Model: {model_type} ({model_version})")
    
    # Setup
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    
    # 1. Load History (Once)
    logger.info("Loading History Data...")
    history_path = os.path.join(os.path.dirname(__file__), '../../../data/processed/preprocessed_data.parquet')
    if os.path.exists(history_path):
        history_df = pd.read_parquet(history_path)
    else:
        logger.error("History data not found.")
        return

    # 2. Load Model
    model_dir = os.path.join(os.path.dirname(__file__), '../../../models')
    model = None
    if model_type == 'ensemble':
        model = EnsembleModel()
        path = os.path.join(model_dir, f'ensemble_{model_version}.pkl')
        if not os.path.exists(path): 
            path = os.path.join(model_dir, 'ensemble_model.pkl')
        logger.info(f"Loading Ensemble from {path}")
        model.load_model(path)
    elif model_type == 'lgbm':
        model = KeibaLGBM()
        path = os.path.join(model_dir, f'lgbm_{model_version}.pkl')
        if not os.path.exists(path): 
            path = os.path.join(model_dir, 'lgbm.pkl')
        logger.info(f"Loading LGBM from {path}")
        model.load_model(path)
        
    if not model:
        logger.error("Model load failed.")
        return

    # Extract features (Basic cleanup)
    # Note: Using a dummy implementation or attempting to infer features
    # For simplicity, we assume the model object handles prediction if we pass a matching dataframe.
    # But usually we need to filter columns.
    
    # Attempt to get features from model
    feature_cols = []
    try:
        bst = model.model
        if hasattr(bst, 'feature_name'): feature_cols = bst.feature_name()
        elif hasattr(bst, 'feature_name_'): feature_cols = bst.feature_name_
        elif hasattr(bst, 'booster_'): feature_cols = bst.booster_.feature_name()
    except: pass
    
    if not feature_cols:
        # Fallback dump
        import pickle
        dataset_path = os.path.join(os.path.dirname(__file__), '../../../data/processed/lgbm_datasets.pkl')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                datasets = pickle.load(f)
                if datasets['train']['X'] is not None:
                    feature_cols = datasets['train']['X'].columns.tolist()

    if not feature_cols:
        logger.error("Could not determine feature columns.")
        return
        
    logger.info(f"Feature count: {len(feature_cols)}")

    # 3. Monthly Batch Processing
    # Instead of daily loop, we iterate by month 01-12
    # We define a custom batch loader logic here to avoid modifying core loader
    
    loader_tbl_race = loader._get_table_name(['jvd_ra', 'race', 'jvd_race_shosai'])
    loader_tbl_entry = loader._get_table_name(['jvd_se', 'seiseki', 'jvd_ur']) # Use SE (Result) if available for rank
    loader_tbl_uma = loader._get_table_name(['jvd_um', 'uma', 'jvd_uma_master'])
    
    # Check if we need to switch entry table if 'jvd_se' is not found?
    # InferenceDataLoader usually picks best available. 
    # For backtest (2025 past), jvd_se is best.
    
    output_dir = os.path.join(os.path.dirname(__file__), '../../../experiments/predictions/ytd_2025')
    os.makedirs(output_dir, exist_ok=True)
    
    # Column mapping for export
    export_cols = [
        'race_id', 'horse_number', 'score', 'prob', 'calibrated_prob', 'pred_rank',
        'date', 'venue', 'race_number', 'horse_name', 'rank', 'odds', 'popularity',
        'surface', 'distance' # Segmentation Features
    ]

    for month in range(1, 13):
        year_month = f"2025{month:02}"
        logger.info(f"Processing Month: {year_month} ...")
        
        # Build Query for the Month
        # Copied logic from loader.py but WHERE kaisai_nen+tsukihi matches month
        
        col_title = "r.race_mei_honbun"
        col_state = "r.baba_jotai_code"
        col_horse_name = "u.bamei"
        col_sex = "u.seibetsu_code"
        col_sire = "u.fushu_ketto_toroku_bango"
        col_mare = "u.boshu_ketto_toroku_bango"
        col_futan = "e.futan_juryo"
        
        # Determine strict table names from loader logic
        # Simplification: Assume standard jra-van tables exist since loader initialized ok
        
        # Note: We need complex joins similar to loader to get all features (bloodline etc)
        # Re-implementing full query is error prone.
        # Alternative: loader.load(target_date=...) accepts a list? No.
        # But we can iterate days in month? That's ~8-10 days per month.
        # 30 days / month? No, racing is weekends. ~8 days.
        # 8 calls * 12 months = 96 calls.
        # 96 calls is manageable (much better than 365, wait... 365 includes non-race days?)
        # My previous script iterating 'dates' from get_2025_dates() found 312 dates?
        # 312 dates in 2025? That's almost every day. NAR (Local) Data included?
        # Ah, 'jvd_ra' contains NAR if imported? Or maybe just lots of JRA days?
        # Local racing happens weekdays.
        # If user wants Central (JRA), 312 seems high. JRA is usually Sat/Sun. ~100 days.
        # Let's check venue codes. if user wants analysis, JRA is primary.
        # But let's stick to batching by month to handle volume regardless.
        
        # Query to get all race dates in this month
        month_dates_query = text(f"""
            SELECT DISTINCT kaisai_nen || kaisai_tsukihi 
            FROM jvd_ra 
            WHERE kaisai_nen = '2025' AND kaisai_tsukihi LIKE '{month:02}%'
        """)
        
        dates_in_month = []
        with loader.engine.connect() as conn:
            res = conn.execute(month_dates_query)
            dates_in_month = [f"{row[0][:4]}-{row[0][4:6]}-{row[0][6:]}" for row in res]
            
        if not dates_in_month:
            continue
            
        logger.info(f"  Found {len(dates_in_month)} race days in {year_month}")
        
        # For efficiency, we CAN run processing day-by-day but skip overhead?
        # No, "Batch by Month" DF construction is best.
        
        # We will use the loader's method to generate query for specific dates? No, loader takes one date.
        # Hack: We construct a big WHERE clause and inject it?
        # Too hacky.
        # Let's just loop the dates in month. 100 days total is 100 loops.
        # The previous bottleneck might have been "Bloodline Loading" in Preprocessor.
        # If Preprocessor instance is reused (it is), cache is warm.
        # The first day took long, subsequent should be fast.
        # The log showed: "Generating: 1% ... 6.81s/it". 7s * 300 = 2100s = 35 mins.
        # Optimize: Reduce columns merged. 
        # Optimize: Save ONLY necessary columns.
        
        # Actually, let's just loop. But use the persistent preprocessor.
        
        for date_str in tqdm(dates_in_month, desc=f"Month {month}", leave=False):
            flat_date = date_str.replace('-', '')
            try:
                raw_df = loader.load(target_date=flat_date)
                if raw_df.empty: continue
                
                # Filter for JRA Only (Venue 01-10)
                # Ensure venue column is processed correctly (sometimes int/str)
                # InferenceDataLoader returns 'venue' column
                if 'venue' in raw_df.columns:
                    # Convert to int safe
                    try:
                        raw_df = raw_df[raw_df['venue'].astype(int) <= 10]
                    except:
                        pass # Keep if conversion fails, or log warning
                        
                if raw_df.empty: continue
                
                features_df, ids_df = preprocessor.preprocess(raw_df, history_df=history_df)
                if features_df.empty: continue
                
                # Align columns
                full_df = pd.concat([ids_df, features_df], axis=1)
                full_df = full_df.loc[:, ~full_df.columns.duplicated()]
                
                for c in set(feature_cols) - set(full_df.columns):
                    full_df[c] = 0
                
                # Predict
                X_in = full_df[feature_cols]
                scores = model.predict(X_in)
                
                # Prob & Calib
                # Group by race_id for softmax
                full_df['score'] = scores
                full_df['prob'] = full_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
                # Calibration (if available, mostly yes)
                # Assuming calibrator loaded if exists (add load logic if missing)
                # For now skipping explicit calibrator load in this snippet (it was in main)
                # full_df['calibrated_prob'] = full_df['prob'] 
                
                # Prepare Result
                # Merge needed info from raw_df: 'rank', 'odds', 'popularity', 'horse_name', 'venue', 'distance', 'surface', 'race_number'
                # raw_df columns from loader: 
                # race_id, horse_number, venue, race_number, distance, surface, weather, state, title
                # rank, odds, popularity, horse_name, ...
                
                cols_needed = ['race_id', 'horse_number', 'venue', 'race_number', 'distance', 'surface', 'horse_name', 'rank', 'odds', 'popularity']
                meta_df = raw_df[[c for c in cols_needed if c in raw_df.columns]].copy()
                
                # Cast IDs
                full_df['race_id'] = full_df['race_id'].astype(str)
                full_df['horse_number'] = full_df['horse_number'].astype(str)
                meta_df['race_id'] = meta_df['race_id'].astype(str)
                meta_df['horse_number'] = meta_df['horse_number'].astype(str)
                
                # Fix Surface Encoding (Map to English)
                if 'surface' in meta_df.columns:
                    def map_surface_safe(s):
                        if pd.isna(s): return 'Unknown'
                        s_str = str(s)
                        if '芝' in s_str: return 'Turf'
                        if 'ダート' in s_str: return 'Dirt'
                        # Fallback for codes if they exist
                        if s_str in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']: return 'Turf'
                        if s_str in ['23', '24', '25', '26', '27', '28', '29']: return 'Dirt'
                        return 'Other'
                    meta_df['surface'] = meta_df['surface'].apply(map_surface_safe)
                
                out_df = pd.merge(full_df[['race_id', 'horse_number', 'score', 'prob']], meta_df, on=['race_id', 'horse_number'], how='left')
                out_df['date'] = date_str
                
                # Pred Rank
                out_df['pred_rank'] = out_df.groupby('race_id')['score'].rank(method='min', ascending=False)
                
                # Save Daily CSV
                save_path = os.path.join(output_dir, f"{flat_date}_{model_type}.csv")
                out_df.to_csv(save_path, index=False)
                
            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                continue

if __name__ == "__main__":
    main()
