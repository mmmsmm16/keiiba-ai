
import os
import pandas as pd
import pickle
import logging
import sys

# Add src to path if needed (usually handled by caller)
# from model.evaluate_betting_roi import calculate_race_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeeklyBettingStrategy:
    def __init__(self, config):
        self.config = config
        self.bet_conf = config.get('betting', {})
        self.model_dir = 'models'
        self.bm_win = None
        self.bm_place = None
        self._load_models()
        
    def _load_models(self):
        bm_win_path = os.path.join(self.model_dir, 'betting_model_win.pkl')
        bm_place_path = os.path.join(self.model_dir, 'betting_model_place.pkl')
        
        if os.path.exists(bm_win_path) and os.path.exists(bm_place_path):
            with open(bm_win_path, 'rb') as f: self.bm_win = pickle.load(f)
            with open(bm_place_path, 'rb') as f: self.bm_place = pickle.load(f)
            logger.info("Loaded Betting Models")
        else:
            logger.error("Betting models not found.")
            
    def apply(self, df_pred):
        if not self.bm_win or not self.bm_place:
            logger.error("Models not loaded. Cannot apply strategy.")
            return []
            
        # Helper for features (inline import to avoid circular dependency issues if any)
        from model.evaluate_betting_roi import calculate_race_features

        # Calculate Race Features
        race_feats = calculate_race_features(df_pred)
        features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
        
        # Check if features exist
        missing = [f for f in features if f not in race_feats.columns]
        if missing:
             logger.warning(f"Missing features for betting model: {missing}")
             # Fill with 0?
             for f in missing: race_feats[f] = 0
        
        race_feats['conf_win'] = self.bm_win.predict(race_feats[features])
        race_feats['conf_place'] = self.bm_place.predict(race_feats[features])
        conf_map = race_feats.set_index('race_id')[['conf_win', 'conf_place']].to_dict('index')
        
        all_bets = []
        
        # Thresholds (Phase 14 Best)
        TH_WIN = 0.40
        TH_PLACE = 0.70
        
        for race_id, group in df_pred.groupby('race_id'):
            if race_id not in conf_map: continue
            
            conf = conf_map[race_id]
            
            # Confidence Filter
            if conf['conf_win'] < TH_WIN and conf['conf_place'] < TH_PLACE:
                continue
                
            sorted_horses = group.sort_values('score', ascending=False)
            if len(sorted_horses) < 6: continue

            axis = sorted_horses.iloc[0]
            opponents = sorted_horses.iloc[1:6] # Rank 2-6
            
            # EV Check (Prob * Odds > 1.2)
            if 'odds' not in axis or pd.isna(axis['odds']):
                # Missing Odds -> Skip EV Check or Skip Bet?
                # If we skip, we miss opportunities Friday Night.
                # If we bet, we might bet on low EV.
                pass 
            elif (axis['calibrated_prob'] * axis['odds']) <= 1.2:
                 continue
                 
            # Kelly Calculation
            bankroll = self.bet_conf.get('initial_bankroll', 1000000)
            fraction = self.bet_conf.get('kelly_fraction', 0.1)
            
            odd_val = axis['odds'] if ('odds' in axis and not pd.isna(axis['odds'])) else 2.0 # Default if missing
            b = odd_val - 1
            p = axis['calibrated_prob']
            if b <= 0: b = 1.0
            
            f_star = (p * (b + 1) - 1) / b
            if f_star < 0: f_star = 0
            
            wager = bankroll * fraction * f_star
            # Round to 100 yen
            wager = max(0, int(wager / 100) * 100)
            
            if wager < 100: continue
            
            unit_bet = max(100, int((wager / 5) / 100) * 100)
            total_cost = unit_bet * 5
            
            opp_nums = opponents['horse_number'].astype(int).tolist()
            eyes_str = f"馬連ながし: {int(axis['horse_number'])} - {', '.join(map(str, opp_nums))}<br>(@ {unit_bet}円)"
            
            all_bets.append({
                'race_id': race_id,
                'race_name': group.iloc[0].get('race_name', 'Race'),
                'axis_horse_num': int(axis['horse_number']),
                'axis_horse_name': axis['horse_name'],
                'opponents': opp_nums,
                'bet_type': '馬連',
                'eyes': eyes_str,
                'cost': total_cost,
                'confidence': f"Win:{conf['conf_win']:.2f}, Place:{conf['conf_place']:.2f}",
                # Additional info for backtest
                'unit_bet': unit_bet
            })
            
        return pd.DataFrame(all_bets)
