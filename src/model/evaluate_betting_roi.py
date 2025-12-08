
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import entropy
from tabulate import tabulate

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.betting_strategy import BettingSimulator
from model.evaluate import load_payout_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_race_features(df):
    """
    レース単位の特徴量を計算 (Same as train_betting_model.py)
    """
    race_feats = []
    for race_id, group in df.groupby('race_id'):
        probs = group['prob'].values
        odds = group['odds'].fillna(0).values
        
        ent = entropy(probs)
        odds_std = np.std(odds)
        max_prob = np.max(probs)
        sorted_probs = sorted(probs, reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
        n_horses = len(group)
        
        race_feats.append({
            'race_id': race_id,
            'entropy': ent,
            'odds_std': odds_std,
            'max_prob': max_prob,
            'confidence_gap': gap,
            'n_horses': n_horses
        })
    return pd.DataFrame(race_feats)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Betting Model ROI')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--year', type=int, default=2024, help='Year to evaluate')
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data for year {args.year}...")
    df = pd.read_parquet(args.input)
    if 'race_id' not in df.columns:
        df = df.reset_index()
        
    valid_df = df[df['year'] == args.year].copy()
    if valid_df.empty:
        logger.error(f"No data found for year {args.year}")
        return

    # 2. Load Base Model (LGBM)
    logger.info("Loading Base Model (LGBM)...")
    from model.lgbm import KeibaLGBM
    model = KeibaLGBM()
    model_path = os.path.join(args.model_dir, 'lgbm.pkl')
    if not os.path.exists(model_path):
        logger.error("Base model not found")
        return
    model.load_model(model_path)
    
    # Predict Base Scores
    feature_cols = None
    if hasattr(model.model, 'feature_name'):
        feature_cols = model.model.feature_name()
    
    if feature_cols:
        missing = set(feature_cols) - set(valid_df.columns)
        for c in missing: valid_df[c] = 0
        X_valid = valid_df[feature_cols]
    else:
        # Fallback
        X_valid = valid_df.select_dtypes(include=[np.number])
        
    valid_scores = model.predict(X_valid)
    
    # 3. Prepare Betting Data
    clean_valid = df.loc[valid_df.index].copy()
    if 'race_id' not in clean_valid.columns: clean_valid = clean_valid.reset_index()
    clean_valid = clean_valid[['race_id', 'odds', 'horse_number']].copy()
    clean_valid['score'] = valid_scores
    clean_valid['prob'] = clean_valid.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # 4. Load Betting Model
    logger.info("Loading Betting Model...")
    betting_model_path = os.path.join(args.model_dir, 'betting_model.pkl')
    if not os.path.exists(betting_model_path):
        logger.error("Betting model not found")
        return
        
    import pickle
    with open(betting_model_path, 'rb') as f:
        bet_bst = pickle.load(f)
    
    # Calc Features
    race_feats = calculate_race_features(clean_valid)
    
    # Predict Betting Confidence
    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
    bet_preds = bet_bst.predict(race_feats[features])
    race_feats['bet_confidence'] = bet_preds
    
    # Merge Confidence back to Races? No, just use race_feats map
    confidence_map = race_feats.set_index('race_id')['bet_confidence'].to_dict()
    
    logger.info(f"Clean Valid Size: {len(clean_valid)}")
    logger.info(f"Race Feats Size: {len(race_feats)}")
    if not race_feats.empty:
         sample_key = list(confidence_map.keys())[0]
         logger.info(f"Conf Map Key Sample: {sample_key} (Type: {type(sample_key)})")
         
    sample_valid_id = clean_valid['race_id'].iloc[0]
    logger.info(f"Valid DF ID Sample: {sample_valid_id} (Type: {type(sample_valid_id)})")

    # 5. Simulation Loop
    logger.info("Running ROI Simulation...")
    payout_df = load_payout_data(year=args.year)
    if payout_df.empty:
        logger.error("No payout data")
        return
        
    sim = BettingSimulator(clean_valid, payout_df)
    logger.info(f"Payout Map Size: {len(sim.payout_map)}")
    if sim.payout_map:
         logger.info(f"Payout Map Key Sample: {list(sim.payout_map.keys())[0]} (Type: {type(list(sim.payout_map.keys())[0])})")
    
    # Grid of Thresholds
    bet_conf_thresholds = [0.0, 0.5, 0.52, 0.55, 0.6, 0.7] # Model AUC is low (0.55), so distribution might be tight around 0.5
    ev_threshold = 1.2 # Constant for now, or loop it too
    
    results = []
    
    for conf_th in bet_conf_thresholds:
        # Custom Simulation Logic: explicit control over race filtering
        total_bet = 0
        total_return = 0
        n_races_bet = 0
        debug_count = 0 
        
        # Iterate races
        for race_id, group in clean_valid.groupby('race_id'):
            if debug_count < 10 and conf_th == 0.0:
                 logger.info(f"DEBUG Process Race: {race_id} (Type: {type(race_id)})")
                 
            # Check Confidence
            conf = confidence_map.get(race_id, 0)
            if debug_count < 10 and conf_th == 0.0:
                 logger.info(f"  -> Conf: {conf} vs Th: {conf_th}")

            if conf < conf_th:
                if debug_count < 10 and conf_th == 0.0: logger.info("  -> Skipped by Conf")
                continue
                
            # Check Betting Strategy (Sanrenpuku Box/Formation?)
            # Let's use the logic from optimize_betting.py: Formation Axis(1) - Opps(2-6)
            # If Axis EV > ev_threshold
            
            sorted_horses = group.sort_values('score', ascending=False)
            if len(sorted_horses) < 6: 
                if debug_count < 10 and conf_th == 0.0: logger.info("  -> Skipped by Horses Count")
                continue
            
            axis_horse = sorted_horses.iloc[0]
            axis_ev = axis_horse['prob'] * axis_horse['odds']
            
            if debug_count < 10 and conf_th == 0.0: 
                 logger.info(f"  -> Axis EV: {axis_ev} (Prob {axis_horse['prob']:.4f} * Odds {axis_horse['odds']})")
                 debug_count += 1

            if axis_ev < ev_threshold:
                continue

            # If passed both filters, BET is placed
            if debug_count < 10 and conf_th == 0.0:
                 logger.info(f"  -> BET PLACED for {race_id}")
            
            payout_map = sim.payout_map.get(race_id, {}).get('sanrenpuku', {})
            axis_num = int(axis_horse['horse_number'])
            opps_nums = [int(h) for h in sorted_horses.iloc[1:6]['horse_number']]
            bet_cost = 1000 # 10 combinations * 100 yen
            
            race_return = 0
            
            # Check hits
            # Payout map keys are typically "010203"
            for combo_str, payout in payout_map.items():
                s = str(combo_str).zfill(6)
                try:
                    horses = [int(s[i:i+2]) for i in range(0, 6, 2)]
                except: continue
                
                if axis_num in horses:
                    others = [h for h in horses if h != axis_num]
                    if all(o in opps_nums for o in others):
                        race_return += int(payout)
            
            total_bet += bet_cost
            total_return += race_return
            n_races_bet += 1
            
        roi = total_return / total_bet * 100 if total_bet > 0 else 0
        results.append({
            'Conf >=': conf_th,
            'Bets': n_races_bet,
            'Total Bet': total_bet,
            'Return': total_return,
            'ROI (%)': roi
        })
        
    print(tabulate(results, headers="keys", floatfmt=".1f"))
    
if __name__ == "__main__":
    main()
