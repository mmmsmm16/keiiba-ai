import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sqlalchemy import text
from tqdm import tqdm
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.loader import InferenceDataLoader
from inference.preprocessor import InferencePreprocessor
from model.ensemble import EnsembleModel
from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet
from inference.strategy import WeeklyBettingStrategy
from reporting.html_generator import HTMLReportGenerator
from scripts.run_daily_backtest import evaluate_bets

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ytd_simulation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_2025_dates(loader):
    """Fetch all race dates in 2025."""
    try:
        # PC-KEIBA DB: kaisai_nen, kaisai_tsukihi (strings)
        query = text("SELECT DISTINCT kaisai_nen, kaisai_tsukihi FROM jvd_ra WHERE kaisai_nen = '2025' ORDER BY 1, 2")
        with loader.engine.connect() as conn:
            result = conn.execute(query)
            dates = []
            for row in result:
                nen = row[0]
                md = row[1]
                # Format: YYYY-MM-DD
                dt_str = f"{nen}-{md[:2]}-{md[2:]}"
                dates.append(dt_str)
            return dates
    except Exception as e:
        logger.error(f"Failed to fetch dates: {e}")
        return []

def run_simulation():
    config = load_config()
    initial_bankroll = config['betting'].get('initial_bankroll', 100000)
    current_bankroll = initial_bankroll
    
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    
    # 1. Load History (Once)
    logger.info("Loading History Data...")
    history_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    if os.path.exists(history_path):
        history_df = pd.read_parquet(history_path)
    else:
        logger.error("History data not found.")
        return

    # 2. Load Models (Once)
    logger.info("Loading Models...")
    model_type = config['betting'].get('model_type', 'lgbm')
    model_version = config['betting'].get('model_version', 'v4_emb')
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    
    model = None
    if model_type == 'ensemble':
        model = EnsembleModel()
        path = os.path.join(model_dir, f'ensemble_{model_version}.pkl')
        if not os.path.exists(path): path = os.path.join(model_dir, 'ensemble_model.pkl')
        model.load_model(path)
    elif model_type == 'lgbm':
        model = KeibaLGBM()
        path = os.path.join(model_dir, f'lgbm_{model_version}.pkl')
        if not os.path.exists(path): path = os.path.join(model_dir, 'lgbm.pkl')
        model.load_model(path)
    # Add other models if needed
    
    # 2b. Extract Feature Names (Once)
    feature_cols = None
    logger.info(f"Model internal type: {type(model.model)}")
    try:
        bst = model.model
        if hasattr(bst, 'feature_name'):
            feature_cols = bst.feature_name()
            logger.info("Retrieved feature names from bst.feature_name()")
        elif hasattr(bst, 'feature_name_'):
            feature_cols = bst.feature_name_
            logger.info("Retrieved feature names from bst.feature_name_")
        elif hasattr(bst, 'booster_'):
            feature_cols = bst.booster_.feature_name()
            logger.info("Retrieved feature names from bst.booster_.feature_name()")
    except Exception as e:
        logger.warning(f"Failed to get feature names from model: {e}")

    if not feature_cols:
        logger.info("Trying to load feature names from dataset (Fallback)...")
        dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
        if os.path.exists(dataset_path):
            import pickle
            with open(dataset_path, 'rb') as f:
                datasets = pickle.load(f)
            if datasets['train']['X'] is not None:
                feature_cols = datasets['train']['X'].columns.tolist()
                logger.info("Retrieved feature names from pickle dataset.")

    if not feature_cols:
        logger.error("CRITICAL: Failed to determine feature names. Aborting.")
        return

    logger.info(f"Feature Names Determined: {len(feature_cols)} features.")
    logger.info(f"First 5 features: {feature_cols[:5]}")


    logger.info(f"First 5 features: {feature_cols[:5]}")

    # Load Calibrator
    calibrator = None
    calib_path = os.path.join(model_dir, 'calibrator.pkl')
    if os.path.exists(calib_path):
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        calibrator.load(calib_path)
        logger.info("Calibrator loaded.")

    # 3. Get Dates
    dates = get_2025_dates(loader)
    logger.info(f"Found {len(dates)} race dates in 2025.")
    
    # 4. Simulation Loop
    results = []
    
    progress_bar = tqdm(dates, desc="Simulating YTD")
    for date_str in progress_bar:
        # logger.info(f"--- Processing {date_str} ---")
        progress_bar.set_postfix(bankroll=int(current_bankroll))
        
        # A. Load Raw
        flat_date = date_str.replace('-', '')
        raw_df = loader.load(target_date=flat_date)
        if raw_df.empty:
            continue
            
        # B. Preprocess
        # Use cached history_df
        X, ids = preprocessor.preprocess(raw_df, history_df=history_df)
        if X.empty:
            continue
            
        processed_df = pd.concat([ids, X], axis=1)
        
        # Deduplicate columns (critical for avoiding feature mismatch)
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        
        # C. Predict
        # Ensure columns exist
        missing = set(feature_cols) - set(processed_df.columns)
        if missing:
             # logger.warning(f"Missing features: {missing}. Filling with 0.")
             for c in missing: processed_df[c] = 0
        X_pred = processed_df[feature_cols]
        
        scores = model.predict(X_pred)
        processed_df['score'] = scores
        processed_df['prob'] = processed_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
        if calibrator:
            processed_df['calibrated_prob'] = calibrator.predict(processed_df['prob'].values)
        else:
            processed_df['calibrated_prob'] = processed_df['prob']
            
        processed_df['expected_value'] = processed_df['calibrated_prob'] * processed_df['odds'].fillna(0)
        
        # D. Strategy
        # Update config bankroll for Strategy? 
        # Strategy uses 'initial_bankroll' for sizing?
        # Actually WeeklyBettingStrategy uses Kelly which depends on Bankroll.
        # But 'initial_bankroll' in config is fixed.
        # If we want Dynamic Kelly (Compound Interest), we should update config['betting']['initial_bankroll'] = current_bankroll ??
        # Or better, pass bankroll to strategy.apply? Strategy.apply reads config.
        # For this simulation, let's update the config object pass to Strategy.
        current_config = config.copy()
        current_config['betting']['initial_bankroll'] = current_bankroll
        
        strategy = WeeklyBettingStrategy(current_config)
        df_bets = strategy.apply(processed_df)
        
        daily_res = {
            'date': date_str,
            'bets': 0,
            'cost': 0,
            'return': 0,
            'profit': 0,
            'bankroll': current_bankroll,
            'roi': 0
        }
        
        if not df_bets.empty:
            # Metadata for evaluate/report
            if 'title' in processed_df.columns:
                 meta = processed_df[['race_id', 'title', 'venue', 'race_number']].drop_duplicates()
                 df_bets = pd.merge(df_bets, meta, on='race_id', how='left')

            # E. Evaluate
            df_evaluated, t_cost, t_ret = evaluate_bets(df_bets, date_str)
            t_profit = t_ret - t_cost
            
            # Update Bankroll
            current_bankroll += t_profit
            
            daily_res.update({
                'bets': len(df_bets),
                'cost': t_cost,
                'return': t_ret,
                'profit': t_profit,
                'bankroll': current_bankroll,
                'roi': (t_ret / t_cost * 100) if t_cost > 0 else 0
            })
            
            # Optional: Generate HTML daily report? (Maybe too many files)
            # genes = HTMLReportGenerator(output_dir='reports/backtest/ytd_2025')
            # genes.generate(df_evaluated, current_bankroll, race_data=processed_df, date_str=date_str)
            
        results.append(daily_res)
        # logger.info(f"Result: Cost {daily_res['cost']}, Return {daily_res['return']}, Profit {daily_res['profit']}, Bankroll {daily_res['bankroll']}")

    # 5. Summary & Report
    df_results = pd.DataFrame(results)
    if df_results.empty:
        logger.warning("No results to report.")
        return

    csv_path = 'reports/backtest/ytd_simulation_2025.csv'
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved simulation results to {csv_path}")
    
    # Plot
    df_results['date'] = pd.to_datetime(df_results['date'])
    plt.figure(figsize=(12, 6))
    plt.plot(df_results['date'], df_results['bankroll'], marker='o')
    plt.title('2025 Bankroll Simulation (YTD)')
    plt.xlabel('Date')
    plt.ylabel('Bankroll (JPY)')
    plt.grid(True)
    plt.savefig('reports/backtest/ytd_simulation_chart.png')
    
    # Total Stats
    total_cost = df_results['cost'].sum()
    total_return = df_results['return'].sum()
    total_profit = df_results['profit'].sum()
    total_roi = (total_return / total_cost * 100) if total_cost > 0 else 0
    
    logger.info("=== YTD Simulation Complete ===")
    logger.info(f"Total Cost: ¥{total_cost:,}")
    logger.info(f"Total Return: ¥{total_return:,}")
    logger.info(f"Total Profit: ¥{total_profit:,}")
    logger.info(f"ROI: {total_roi:.2f}%")
    logger.info(f"Final Bankroll: ¥{current_bankroll:,}")

if __name__ == "__main__":
    run_simulation()
