
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.betting_strategy import BettingSimulator
from model.evaluate import load_payout_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Optimize Betting Strategy')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet', help='Input data path')
    parser.add_argument('--year', type=int, default=2024, help='Year to optimize on')
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data from {args.input}...")
    if not os.path.exists(args.input):
        logger.error("Input file not found.")
        return
        
    df = pd.read_parquet(args.input)
    target_df = df[df['year'] == args.year].copy()
    
    if target_df.empty:
        logger.error(f"No data for year {args.year}.")
        return

    # DEBUG: Limit data size
    # logger.info("DEBUG: Limiting data to first 1000 rows for testing.")
    # target_df = target_df.head(1000).copy()
    # Here we assume we need predictions. 
    # Ideally, we should use cross-validated predictions or a held-out set.
    # For now, let's assume 'score' and 'probs' are NOT in preprocessed_data.
    # We need to run inference or load a result file.
    # Actually, evaluate.py saves results? No.
    # Let's load the latest simulation result if possible? No, that's JSON.
    
    # We must load a model and predict to get scores for optimization.
    logger.info("Loading model for generating scores...")
    from model.ensemble import EnsembleModel
    model = EnsembleModel()
    model_path = 'models/ensemble_model.pkl'
    if not os.path.exists(model_path):
        logger.warning("Ensemble model not found, trying LGBM")
        from model.lgbm import KeibaLGBM
        model = KeibaLGBM()
        model_path = 'models/lgbm.pkl'
        
    if os.path.exists(model_path):
        model.load_model(model_path)
        
        if hasattr(model, 'lgbm') and hasattr(model.lgbm, 'model') and hasattr(model.lgbm.model, 'feature_name'):
             # EnsembleModel case (use LightGBM's feature names as they should be shared)
             feature_cols = model.lgbm.model.feature_name()
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_name'):
             # LGBM case
             feature_cols = model.model.feature_name()
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_names_'):
             # CatBoost case
             feature_cols = model.model.feature_names_

        if feature_cols:
            # Check missing
            missing = set(feature_cols) - set(target_df.columns)
            for c in missing: target_df[c] = 0
            
            X = target_df[feature_cols]
        else:
            # Fallback for Ensemble or if name extraction fails
            # Try to load dataset info or use all numeric columns
            logger.warning("Could not extract feature names from model. Using all available columns in preprocessed data matching numerical types.")
            # Exclude id and target cols
            exclude = ['race_id', 'horse_number', 'horse_id', 'jockey_id', 'trainer_id', 'rank', 'time', 'odds', 'popularity', 'year', 'expected_value', 'score', 'prob']
            X = target_df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude if c in target_df.columns])
            
        logger.info(f"Predicting with {X.shape[1]} features...")
        scores = model.predict(X)
        target_df['score'] = scores
        
        # Softmax
        from scipy.special import softmax
        target_df['prob'] = target_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
        target_df['expected_value'] = target_df['prob'] * target_df['odds'].fillna(0)
    else:
        logger.error("No model found to generate scores for optimization.")
        return

    # 2. Load Payouts
    payout_df = load_payout_data(year=args.year)
    if payout_df.empty:
        logger.error("Payout data not found.")
        return

    # 3. Simulate
    sim = BettingSimulator(target_df, payout_df)
    
    strategies = [
        {'name': 'Sanrenpuku Formation (Axis 1, Opp 2-6)', 'type': 'sanrenpuku', 'axis': 1, 'opps': [2,3,4,5,6]},
        {'name': 'Sanrenpuku Box (Top 5)', 'type': 'sanrenpuku', 'axis': None, 'opps': [1,2,3,4,5]}, # Logic handling needed
        {'name': 'Umaren Formation (Axis 1, Opp 2-5)', 'type': 'umaren', 'axis': 1, 'opps': [2,3,4,5]},
        {'name': 'Umaren Box (Top 5)', 'type': 'umaren', 'axis': None, 'opps': [1,2,3,4,5]}, 
    ]
    
    results = []
    
    for strat in strategies:
        if strat['axis'] is None:
            # Box Simulation logic inside simulate_formation? No.
            # Need simulate_box in BettingSimulator (Not implemented there yet, but logic is in evaluate.py)
            # Let's skip Box for now or add it to BettingSimulator quickly if needed.
            # Actually simulate_formation works for Box if implementation allows? No.
            continue
            
        logger.info(f"Simulating {strat['name']}...")
        res = sim.simulate_formation(
            axis_rank=strat['axis'], 
            opponent_ranks=strat['opps'], 
            ticket_type=strat['type']
        )
        res['strategy'] = strat['name']
        results.append(res)
        logger.info(f"ROI: {res['roi']:.2f}%, Hit: {res['accuracy']:.2%} (Bets: {res['bet']})")

    # Grid Search for Thresholds
    logger.info("--- Grid Search Optimization (EV Threshold) ---")
    
    thresholds = np.arange(0.8, 2.6, 0.1)
    grid_results = []
    
    # Target Strategy: Sanrenpuku Formation (Axis 1, Opp 2-6)
    target_strat = {'name': 'Sanrenpuku Formation', 'type': 'sanrenpuku', 'axis': 1, 'opps': [2,3,4,5,6]}
    
    for th in thresholds:
        logger.info(f"Checking Threshold: {th:.1f}...")
        res = sim.simulate_formation(
            axis_ev_threshold=th,
            axis_rank=target_strat['axis'],
            opponent_ranks=target_strat['opps'],
            ticket_type=target_strat['type']
        )
        res['threshold'] = round(th, 2)
        res['strategy'] = target_strat['name']
        grid_results.append(res)
        logger.info(f"Th: {th:.1f} -> ROI: {res['roi']:.2f}%, Hit: {res['accuracy']:.2%} (Bets: {res['bet']})")

    # Save
    pd.DataFrame(results).to_csv('betting_optimization_strategies.csv', index=False)
    pd.DataFrame(grid_results).to_csv('betting_optimization_grid.csv', index=False)
    logger.info("Optimization finished. Results saved.")


if __name__ == "__main__":
    main()
