
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from scipy.special import softmax
from scipy.stats import entropy
from itertools import combinations

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.betting_strategy import BettingSimulator, BettingOptimizer, BankrollSimulator
from model.evaluate import load_payout_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_race_features(df):
    """
    レース単位の特徴量を計算
    """
    race_feats = []
    prob_col = 'calibrated_prob' if 'calibrated_prob' in df.columns else 'prob'
    
    for race_id, group in df.groupby('race_id'):
        probs = group[prob_col].values
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
    df_res = pd.DataFrame(race_feats)
    if df_res.empty:
         return pd.DataFrame(columns=['race_id', 'entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses'])
    return df_res

def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Stage Betting ROI with Staking')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--year', type=int, default=2024, help='Year to evaluate')
    parser.add_argument('--simulation_mode', type=str, default='flat', choices=['flat', 'kelly', 'grid_search'], help='Staking Strategy')
    parser.add_argument('--ticket_type', type=str, default='sanrenpuku', choices=['sanrenpuku', 'umaren', 'wide'], help='Betting Ticket Type')
    parser.add_argument('--kelly_fraction', type=float, default=0.25, help='Fractional Kelly (0.1-1.0)')
    parser.add_argument('--initial_bankroll', type=int, default=1000000, help='Start Bankroll')
    parser.add_argument('--model_type', type=str, default='lgbm', choices=['lgbm', 'catboost'], help='Model Type')
    parser.add_argument('--model_version', type=str, default=None, help='Model Version (e.g. v5_weighted)')
    parser.add_argument('--skip_ev', action='store_true', help='Skip EV check')
    args = parser.parse_args()

    # 1. Load Data
    # 1. Load Data
    logger.info(f"Loading data for year {args.year}...")
    df = pd.read_parquet(args.input)
    if 'race_id' not in df.columns: df = df.reset_index()
        
    valid_df = df[df['year'] == args.year].copy()
    if valid_df.empty:
        logger.error(f"No data found for year {args.year}")
        return

    # 2. Base Model Predictions
    logger.info(f"Loading Base Model ({args.model_type} {args.model_version or ''}) & Calibrator...")
    
    model_path = None
    if args.model_version:
        # Check specific version
        path = os.path.join(args.model_dir, f'{args.model_type}_{args.model_version}.pkl')
        if os.path.exists(path):
            model_path = path
        else:
            logger.warning(f"指定されたモデルが見つかりません: {path}。デフォルトロジックに戻ります。")
            
    if not model_path:
        if args.model_type == 'lgbm':
            if os.path.exists(os.path.join(args.model_dir, 'lgbm_v4_emb.pkl')):
                 model_path = os.path.join(args.model_dir, 'lgbm_v4_emb.pkl')
            else:
                 model_path = os.path.join(args.model_dir, 'lgbm.pkl')
        elif args.model_type == 'catboost':
            if os.path.exists(os.path.join(args.model_dir, 'catboost_v9_emb.pkl')):
                 model_path = os.path.join(args.model_dir, 'catboost_v9_emb.pkl')
            elif os.path.exists(os.path.join(args.model_dir, 'catboost_v9.pkl')):
                 model_path = os.path.join(args.model_dir, 'catboost_v9.pkl')
            else:
                 model_path = os.path.join(args.model_dir, 'catboost.pkl')

    if args.model_type == 'lgbm':
        from model.lgbm import KeibaLGBM
        model = KeibaLGBM()
    else:
        from model.catboost_model import KeibaCatBoost
        model = KeibaCatBoost()
        
    model.load_model(model_path)
    
    feature_cols = getattr(model.model, 'feature_name', lambda: [])()
    if not feature_cols:
        X_valid = valid_df.select_dtypes(include=[np.number])
    else:
        for c in feature_cols: 
             if c not in valid_df.columns: valid_df[c] = 0
        X_valid = valid_df[feature_cols]
        
    valid_df['score'] = model.predict(X_valid)
    valid_df['prob'] = valid_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # Calibration
    calib_name = f'calibrator_{args.model_version}.pkl' if args.model_version else 'calibrator.pkl'
    calib_path = os.path.join(args.model_dir, calib_name)
    
    if not os.path.exists(calib_path):
        # Fallback to default
        calib_path = os.path.join(args.model_dir, 'calibrator.pkl')
        
    if os.path.exists(calib_path):
        logger.info(f"Calibrator loaded from {calib_path}")
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        calibrator.load(calib_path)
        valid_df['calibrated_prob'] = calibrator.predict(valid_df['prob'].values)
    else:
        logger.warning("Calibrator not found, using raw probability.")
        valid_df['calibrated_prob'] = valid_df['prob']
        
    # 3. Betting Models for Thresholding
    # Note: These use 'calibrated_prob' derived features.
    
    # Determine Betting Model Names
    suffix = f"_{args.model_version}" if args.model_version else ""
    path_win = os.path.join(args.model_dir, f'betting_model{suffix}_win.pkl')
    path_place = os.path.join(args.model_dir, f'betting_model{suffix}_place.pkl')
    
    # Fallback
    if not os.path.exists(path_win):
        logger.warning(f"Betting Model (Win) not found at {path_win}, using default.")
        path_win = os.path.join(args.model_dir, 'betting_model_win.pkl')
    if not os.path.exists(path_place):
        logger.warning(f"Betting Model (Place) not found at {path_place}, using default.")
        path_place = os.path.join(args.model_dir, 'betting_model_place.pkl')
        
    with open(path_win, 'rb') as f:
        model_win = pickle.load(f)
    with open(path_place, 'rb') as f:
        model_place = pickle.load(f)
    logger.info(f"Loaded Betting Models: {os.path.basename(path_win)}, {os.path.basename(path_place)}")
    
    clean_valid = valid_df[['race_id', 'odds', 'horse_number', 'calibrated_prob', 'score', 'date']].copy()
    race_feats = calculate_race_features(clean_valid)
    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
    
    race_feats['conf_win'] = model_win.predict(race_feats[features])
    race_feats['conf_place'] = model_place.predict(race_feats[features])
    
    conf_map = race_feats.set_index('race_id')[['conf_win', 'conf_place']].to_dict('index')

    # Prepare for Simulation
    logger.info("Initializing Simulator...")
    payout_df = load_payout_data(years=[args.year])
    optimizer = BettingOptimizer(clean_valid, payout_df)
    
    # Optimal Thresholds (Updated via Grid Search)
    TH_WIN = 0.40
    TH_PLACE = 0.70

    # Strategy Function for Simulator
    def strategy_two_stage(optimizer, race_id, current_bankroll, **kwargs):
        confs = conf_map.get(race_id, {'conf_win':0, 'conf_place':0})
        c_win = confs['conf_win']
        c_place = confs['conf_place']
        
        group = optimizer.df[optimizer.df['race_id'] == race_id]
        if group.empty or len(group) < 6: return [], 0
        
        sorted_horses = group.sort_values('score', ascending=False)
        axis_horse = sorted_horses.iloc[0]
        
        # EV Check
        skip_ev = kwargs.get('skip_ev', False)
        if skip_ev:
             ev_check = True
        else:
             ev_check = (axis_horse['calibrated_prob'] * axis_horse['odds']) > 1.2
             
        if not ev_check:
            return [], 0
            
        # Build Bets Candidates first to count them
        temp_bets = []
        
        # Strategy A & B Combined Logic
        if (c_win >= TH_WIN) or (c_place >= TH_PLACE):
            opp_nums = [int(h) for h in sorted_horses.iloc[1:6]['horse_number']]
            axis_num = int(sorted_horses.iloc[0]['horse_number'])
            
            ticket_type = kwargs.get('ticket_type', 'sanrenpuku')
            
            if ticket_type == 'sanrenpuku':
                if len(opp_nums) >= 2:
                    for oc in combinations(opp_nums, 2):
                        c = sorted([axis_num, oc[0], oc[1]])
                        temp_bets.append({'type': 'sanrenpuku', 'combo': c})
            
            elif ticket_type == 'umaren':
                for opp in opp_nums:
                    c = sorted([axis_num, opp])
                    temp_bets.append({'type': 'umaren', 'combo': c})

            elif ticket_type == 'wide':
                for opp in opp_nums:
                    c = sorted([axis_num, opp])
                    temp_bets.append({'type': 'wide', 'combo': c})

        if not temp_bets:
            return [], 0
            
        bet_count = len(temp_bets) # Usually 10
        
        # Determine Bet Amount
        unit_stake = 100 # Default Flat
        
        if kwargs.get('mode') == 'kelly':
            # p = High Confidence (0.7~). b = Effective Odds (e.g. 4.0).
            # We calculate TOTAL budget for this opportunity.
            total_budget = optimizer.calculate_kelly_bet(c_place, 5.0, current_bankroll, fraction=kwargs.get('fraction', 0.25))
            
            # Split budget
            if bet_count > 0:
                unit_stake = int(total_budget // bet_count)
                # Round to 100
                unit_stake = (unit_stake // 100) * 100
                
            if unit_stake < 100:
                # If budget too low, skip
                return [], 0
        
        # Create Final Bets
        bets_list = []
        for b in temp_bets:
            b['amount'] = unit_stake
            bets_list.append(b)
        
        # Total cost is sum of amounts
        total_cost = sum([b['amount'] for b in bets_list])
        return bets_list, total_cost


    
    if args.simulation_mode == 'grid_search':
        logger.info(f"Running Grid Search for Optimal Thresholds (Ticket: {args.ticket_type})...")
        
        # Define Grid
        TH_WIN_GRID = np.arange(0.20, 0.60, 0.05)
        TH_PLACE_GRID = np.arange(0.40, 0.90, 0.05)
        
        results = []
        
        # Pre-calculate Race Data
        logger.info(f"Payout Map Size: {len(optimizer.payout_map)}")
        processed_races = {}
        for i, (race_id, group) in enumerate(clean_valid.groupby('race_id')):
            if i == 0: logger.info(f"Scanning Race: {race_id}, In Payouts? {race_id in optimizer.payout_map}")
            sorted_horses = group.sort_values('score', ascending=False)
            if len(sorted_horses) < 6: continue
            
            axis_horse = sorted_horses.iloc[0]
            # Opponent: Rank 2-6 (5 horses)
            opp_nums = [int(h) for h in sorted_horses.iloc[1:6]['horse_number']]
            
            # EV Check
            if args.skip_ev:
                ev_check = True
            else:
                ev_check = (axis_horse['calibrated_prob'] * axis_horse['odds']) > 1.2
            
            payouts = optimizer.payout_map.get(race_id, {})
            # Sanity check if payouts exist
            if not payouts: continue

            ret_race = 0
            cost_race = 0
            
            # --- Logic Switch based on ticket_type ---
            if args.ticket_type == 'sanrenpuku':
                # Axis 1 - Rank 2-6 (Box? No, Axis + Opponents) -> Formations
                 if len(opp_nums) >= 2:
                    sanren_map = payouts.get('sanrenpuku', {})
                    tickets = list(combinations(opp_nums, 2))
                    cost_race = len(tickets) * 100
                    for oc in tickets:
                        c = sorted([int(axis_horse['horse_number']), oc[0], oc[1]])
                        key = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                        if key in sanren_map:
                            ret_race += int(sanren_map[key])
                            
            elif args.ticket_type == 'umaren':
                # Axis 1 - Rank 2-6 (5 points)
                umaren_map = payouts.get('umaren', {})
                cost_race = len(opp_nums) * 100
                for opp in opp_nums:
                    c = sorted([int(axis_horse['horse_number']), opp])
                    key = f"{c[0]:02}{c[1]:02}"
                    if key in umaren_map:
                        ret_race += int(umaren_map[key])

            elif args.ticket_type == 'wide':
                # Axis 1 - Rank 2-6 (5 points)
                wide_map = payouts.get('wide', {})
                cost_race = len(opp_nums) * 100
                for opp in opp_nums:
                    c = sorted([int(axis_horse['horse_number']), opp])
                    key = f"{c[0]:02}{c[1]:02}"
                    if key in wide_map:
                        ret_race += int(wide_map[key])

            processed_races[race_id] = {
                'ev_check': ev_check,
                'return': ret_race,
                'cost': cost_race,
                'conf_win': conf_map[race_id]['conf_win'],
                'conf_place': conf_map[race_id]['conf_place']
            }
        
        logger.info(f"Processed Races: {len(processed_races)}")
        if len(processed_races) > 0:
            first_key = list(processed_races.keys())[0]
            logger.info(f"Sample Race Data: {processed_races[first_key]}")
            
        print(f"{'TH_WIN':<10} {'TH_PLACE':<10} {'ROI':<10} {'BETS':<10} {'PROFIT':<15}")
        print("-" * 60)
        
        for th_win in TH_WIN_GRID:
            for th_place in TH_PLACE_GRID:
                tot_bet = 0
                tot_ret = 0
                cnt = 0
                
                for rid, data in processed_races.items():
                    if not data['ev_check']: continue
                    
                    if (data['conf_win'] >= th_win) or (data['conf_place'] >= th_place):
                        tot_bet += data['cost']
                        tot_ret += data['return']
                        cnt += 1
                
                roi = tot_ret / tot_bet * 100 if tot_bet > 0 else 0
                profit = tot_ret - tot_bet
                
                if cnt > 10 and roi > 80: # Filter slightly
                     print(f"{th_win:<10.2f} {th_place:<10.2f} {roi:<10.1f} {cnt:<10} {profit:<15,}")

    elif args.simulation_mode == 'flat':
        # Flat Simulation using BettingSimulator directly or simplified loop
        # We need to print the stats as before.
        logger.info("Running Flat Bet Simulation...")
        
        # Grid for reference or just single point
        TH_WIN_GRID = [0.42] 
        TH_PLACE_GRID = [0.70]
        
        for th_win in TH_WIN_GRID:
            for th_place in TH_PLACE_GRID:
                # Use Sim
                sim = BankrollSimulator(optimizer, initial_bankroll=args.initial_bankroll)
                
                # We need a wrapper to force flat bet
                def strategy_flat(optimizer, race_id, current_bankroll, **kwargs):
                    return strategy_two_stage(optimizer, race_id, current_bankroll, mode='flat', ticket_type=args.ticket_type, skip_ev=args.skip_ev) 
                
                sorted_races = clean_valid.sort_values(['date', 'race_id'])['race_id'].unique()
                res = sim.run(sorted_races, strategy_flat)
                
                print(f"TH_Win: {th_win} | TH_Place: {th_place}")
                print(f"Final: {res['final_bankroll']:,} | ROI: {res['roi']:.2f}% | Max DD: {res['max_drawdown']:.2f}%")

    elif args.simulation_mode == 'kelly':
         # ... (Kelly Logic) ...
         # Need to ensure TH_WIN/TH_PLACE are set correctly.
         # For now using hardcoded inside strategy_two_stage, but should ideally be args.
         # We will use 0.42/0.70 as default unless updated.
         logger.info(f"Running Bankroll Simulation (Kelly Fraction={args.kelly_fraction})...")
         sim = BankrollSimulator(optimizer, initial_bankroll=args.initial_bankroll)
         sorted_races = clean_valid.sort_values(['date', 'race_id'])['race_id'].unique()
         res = sim.run(sorted_races, strategy_two_stage, mode='kelly', fraction=args.kelly_fraction, ticket_type=args.ticket_type, skip_ev=args.skip_ev)
         
         print("\n=== Kelly Simulation Results ===")
         print(f"Initial: {args.initial_bankroll:,} Yen")
         print(f"Final:   {res['final_bankroll']:,} Yen")
         print(f"Profit:  {res['final_bankroll'] - args.initial_bankroll:,} Yen")
         print(f"ROI:     {res['roi']:.2f}%")
         print(f"Max DD:  {res['max_drawdown']:.2f}%")

         # Save History
         if 'history' in res and not res['history'].empty:
             os.makedirs('experiments', exist_ok=True)
             history_path = f"experiments/simulation_kelly_{args.ticket_type}_{args.model_type}.csv"
             res['history'].to_csv(history_path, index=False)
             logger.info(f"Simulation history saved to {history_path}")

if __name__ == "__main__":
    main()
