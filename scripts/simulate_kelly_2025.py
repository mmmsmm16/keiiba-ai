"""
Kelly Criterion Calculation & ROI Simulation (2025)
2025年のデータを用いて、EVモデルによる推奨馬券をケリー基準で運用した場合の
ROIと資産推移をシミュレーションする。

Parameters:
- Initial Bankroll: 100,000 JPY
- Max Bet: 5,000 JPY per race (User specified limit)
- Strategy: Win (EV > Threshold) & Umaren (EV > Threshold)
- Bet Sizing: Kelly Criterion fraction (default=0.1 or similar conservative fraction often recommended)
  * Full Kelly is risky; User requested "Kelly Criterion", implying we should use the formula.
  * Formula: f = (bp - q) / b
    - b: Odds - 1
    - p: Probability of winning
    - q: 1 - p
  * Note: User specified "Max 5000", which acts as a cap.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.feature_engineering import FeatureEngineer
from src.data.realtime_loader import RealTimeDataLoader
from src.preprocessing.aggregators import HistoryAggregator
from src.preprocessing.advanced_features import AdvancedFeatureEngineer
from src.preprocessing.incremental_aggregators import IncrementalCategoryAggregator
from src.preprocessing.experience_features import ExperienceFeatureEngineer
from src.preprocessing.relative_features import RelativeFeatureEngineer
from src.preprocessing.opposition_features import OppositionFeatureEngineer
from src.preprocessing.rating_features import RatingFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANKER_MODEL_PATH = "models/eval/ranker_eval_v19.pkl"
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
REPORT_DIR = "reports/jra/daily"

class KellySimulator:
    def __init__(self, model_path: str, initial_bankroll=100000, max_bet=5000, kelly_fraction=0.1, threshold=1.2, min_prob=0.0, min_odds=0.0):
        self.bankroll = initial_bankroll
        self.max_bet = max_bet
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.threshold = threshold
        self.min_prob = min_prob
        self.min_odds = min_odds
        self.history = []
        
        # Load Model
        logger.info(f"Loading Ranker model from {model_path}...")
        payload = joblib.load(model_path)
        self.model = payload['model']
        self.feature_cols = payload['feature_cols']
        
        # Components
        self.loader = JraVanDataLoader()
        self.rt_loader = RealTimeDataLoader()
        self.cleanser = DataCleanser()
        self.engineer = FeatureEngineer()
        self.hist_agg = HistoryAggregator()
        self.adv_eng = AdvancedFeatureEngineer()
        self.exp_eng = ExperienceFeatureEngineer()
        self.rel_eng = RelativeFeatureEngineer()
        self.opp_eng = OppositionFeatureEngineer()
        self.rating_eng = RatingFeatureEngineer()

    def _kelly_bet(self, prob: float, odds: float) -> int:
        """
        Calculate Kelly Bet Amount.
        Result is capped by self.max_bet and adjusted to 100 yen units.
        Uses Fractional Kelly.
        """
        if prob <= 0 or odds <= 1.0:
            return 0
            
        b = odds - 1.0
        q = 1.0 - prob
        f = (b * prob - q) / b # Full Kelly Fraction
        
        # Apply fractional kelly
        f = f * self.kelly_fraction
        
        if f <= 0:
            return 0
            
        # Suggested Bet
        bet_amount = self.bankroll * f
        
        # Cap
        bet_amount = min(bet_amount, self.max_bet)
        
        # Round down to 100 yen
        bet_amount = int(bet_amount // 100) * 100
        return bet_amount

    def _calc_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        def softmax_group(group):
            s = group['score'].values
            if len(s) > 1 and s.std() > 0:
                z = (s - s.mean()) / s.std()
            else:
                z = s - s.mean()
            # Temperature scaling (Removed: T=1.5 -> T=1.0 based on calibration check)
            # exp_s = np.exp(z * 1.5) 
            exp_s = np.exp(z)
            group['win_prob'] = exp_s / exp_s.sum()
            return group
        return df.groupby('race_id', group_keys=False).apply(softmax_group)

    def _calc_umaren_probs(self, race_df: pd.DataFrame) -> Dict[str, float]:
        probs = {}
        runners = race_df[['horse_number', 'win_prob']].values
        for i in range(len(runners)):
            h1, p1 = runners[i]
            for j in range(i + 1, len(runners)):
                h2, p2 = runners[j]
                denom = 1.0 - p1
                denom2 = 1.0 - p2
                if denom > 0 and denom2 > 0:
                    pair_prob = (p1 * p2 / denom) + (p2 * p1 / denom2)
                    key = f"{int(h1):02d}-{int(h2):02d}" # standardized key
                    probs[key] = pair_prob
        return probs

    def _load_process_data(self, start_date, end_date):
        # Load batch of data to reduce overhead (e.g. monthly)
        logger.info(f"Loading data from {start_date} to {end_date}")
        df = self.loader.load(history_start_date=start_date, end_date=end_date, jra_only=True)
        if len(df) == 0: return pd.DataFrame()
        
        df = self.cleanser.cleanse(df)
        df = self.engineer.add_features(df)
        
        # Incremental components
        try:
            master_df = pd.read_parquet(CACHE_PATH)
            master_df['date'] = pd.to_datetime(master_df['date'])
            # Assuming sequential processing, we could optimize this, but for simplicity reloading cache is reliable
            inc_cat_agg = IncrementalCategoryAggregator()
            inc_cat_agg.fit(master_df[master_df['date'] < start_date])
            df = inc_cat_agg.transform_update(df)
            
            # Context for features
            ctx_start = (pd.to_datetime(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            context_df = master_df[(master_df['date'] >= ctx_start) & (master_df['date'] < start_date)]
            proc_df = pd.concat([context_df, df], ignore_index=True).sort_values(['date', 'race_id'])
            
            # Features
            proc_df = self.hist_agg.aggregate(proc_df)
            proc_df = self.adv_eng.add_features(proc_df)
            proc_df = self.exp_eng.add_features(proc_df)
            proc_df = self.rel_eng.add_features(proc_df)
            proc_df = self.opp_eng.add_features(proc_df)
            proc_df = self.rating_eng.add_features(proc_df)
            
            # Filter targeted range
            return proc_df[(proc_df['date'] >= start_date) & (proc_df['date'] <= end_date)].copy()
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def prepare_data(self, start_date, end_date):
        """
        Pre-load and process data for the entire period.
        """
        return self._load_process_data(start_date, end_date)

    def _simulate_day(self, day_df, d_str):
        rids = day_df['race_id'].unique().tolist()
        
        # Win Odds from RealTime
        win_rt = self.rt_loader.get_latest_odds(rids, 'win')
        uma_rt = self.rt_loader.get_latest_odds(rids, 'umaren')
        
        for rid, race_df in day_df.groupby('race_id'):
            # --- Win Bet ---
            odds_map = win_rt.get(rid, {})
            if not odds_map:
                odds_map = dict(zip(race_df['horse_number'].astype(str).str.zfill(2), race_df['odds']))
            
            for _, row in race_df.iterrows():
                hn = f"{int(row['horse_number']):02d}"
                prob = row['win_prob']
                odds = odds_map.get(hn, 0.0)
                
                amount = 0
                if prob >= self.min_prob and odds >= self.min_odds:
                    if prob * odds > self.threshold:
                        amount = self._kelly_bet(prob, odds)
                    
                if amount > 0:
                    rank = row['rank']
                    payoff = amount * odds if rank == 1 else 0
                    self.bankroll += (payoff - amount)
                    self.history.append({
                        'date': d_str,
                        'race_id': rid,
                        'type': 'WIN',
                        'target': hn,
                        'prob': prob,
                        'odds': odds,
                        'bet': amount,
                        'return': payoff,
                        'bankroll': self.bankroll
                    })
                    
            # --- Umaren Bet ---
            uma_probs = self._calc_umaren_probs(race_df)
            uma_odds = uma_rt.get(rid, {})
            
            if uma_odds:
                for combo, prob in uma_probs.items():
                    odds = uma_odds.get(combo, 0.0)
                    if prob >= self.min_prob and odds >= self.min_odds:
                        if prob * odds > self.threshold:
                            amount = self._kelly_bet(prob, odds)
                            if amount > 0:
                                top2 = race_df[race_df['rank'].isin([1, 2])]
                                is_hit = False
                                if len(top2) == 2:
                                    h1 = int(top2.iloc[0]['horse_number'])
                                    h2 = int(top2.iloc[1]['horse_number'])
                                    res_combo = f"{min(h1,h2):02d}-{max(h1,h2):02d}"
                                    is_hit = (combo == res_combo)
                                
                                payoff = amount * odds if is_hit else 0
                                self.bankroll += (payoff - amount)
                                self.history.append({
                                    'date': d_str,
                                    'race_id': rid,
                                    'type': 'UMA',
                                    'target': combo,
                                    'prob': prob,
                                    'odds': odds,
                                    'bet': amount,
                                    'return': payoff,
                                    'bankroll': self.bankroll
                                })

    def run_simulation(self, start_date='2025-01-01', end_date='2025-12-31', preloaded_df=None):
        current_date = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if preloaded_df is not None:
            # Use preloaded dataframe.
            full_df = preloaded_df[(preloaded_df['date'] >= start_date) & (preloaded_df['date'] <= end_date)].sort_values(['date', 'race_id'])
            
            if 'score' not in full_df.columns:
                 X = full_df[self.feature_cols].copy()
                 X = X.loc[:, ~X.columns.duplicated()]
                 X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
                 full_df['score'] = self.model.predict(X)
                 full_df = self._calc_probs(full_df)
            
            # Now simulate betting day by day
            dates = full_df['date'].unique()
            for d in dates:
                d_str = pd.to_datetime(d).strftime('%Y-%m-%d')
                day_df = full_df[full_df['date'] == d]
                self._simulate_day(day_df, d_str)
                
        else:
            # Old Logic (Month by Month loading)
            while current_date <= end_dt:
                # Process one month at a time
                next_month = current_date + pd.offsets.MonthBegin(1)
                batch_end = min(next_month - timedelta(days=1), end_dt)
                
                s_str = current_date.strftime('%Y-%m-%d')
                e_str = batch_end.strftime('%Y-%m-%d')
                
                logger.info(f"Simulating Batch: {s_str} ~ {e_str}")
                
                df = self._load_process_data(s_str, e_str)
                if df.empty:
                    current_date = next_month
                    continue
                    
                # Predict
                X = df[self.feature_cols].copy()
                X = X.loc[:, ~X.columns.duplicated()]
                X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
                df['score'] = self.model.predict(X)
                df = self._calc_probs(df)
                
                # Simulate Day
                dates = df['date'].unique()
                for d in dates:
                    d_str = pd.to_datetime(d).strftime('%Y-%m-%d')
                    day_df = df[df['date'] == d]
                    self._simulate_day(day_df, d_str)
                
                # Monthly Summary
                if self.history:
                    current_hist = [h for h in self.history if h['date'] >= s_str and h['date'] <= e_str]
                    if current_hist:
                        m_df = pd.DataFrame(current_hist)
                        m_bet = m_df['bet'].sum()
                        m_ret = m_df['return'].sum()
                        m_prof = m_ret - m_bet
                        m_roi = (m_ret / m_bet) * 100 if m_bet > 0 else 0
                        print(f"  [Month {s_str[:7]}] Bet: {m_bet}, Return: {m_ret}, Profit: {m_prof}, ROI: {m_roi:.1f}%, Bankroll: {int(self.bankroll)}")
                    else:
                        print(f"  [Month {s_str[:7]}] No bets.")

                current_date = next_month
            
        # Summary
        if not self.history:
            logger.warning("No bets made.")
            return {
                'total_bet': 0,
                'total_return': 0,
                'profit': 0,
                'roi': 0.0,
                'bet_count': 0,
                'final_bankroll': self.initial_bankroll
            }

        hist_df = pd.DataFrame(self.history)
        total_bet = hist_df['bet'].sum()
        total_ret = hist_df['return'].sum()
        roi = (total_ret / total_bet) * 100 if total_bet > 0 else 0
        profit = self.bankroll - self.initial_bankroll
        
        print("\n=== Simulation Result (2025 Kelly) ===")
        print(f"Initial Bankroll: {self.initial_bankroll}")
        print(f"Final Bankroll:   {int(self.bankroll)}")
        print(f"Profit:           {int(profit)} ({profit/self.initial_bankroll*100:.1f}%)")
        print(f"Total Bet:        {total_bet}")
        print(f"ROI:              {roi:.1f}%")
        print(f"Bet Count:        {len(hist_df)}")
        
        # Save CSV
        os.makedirs(REPORT_DIR, exist_ok=True)
        hist_df.to_csv(os.path.join(REPORT_DIR, "sim_kelly_2025.csv"), index=False)
        print(f"Detailed log saved to {os.path.join(REPORT_DIR, 'sim_kelly_2025.csv')}")
        
        return {
            'total_bet': total_bet,
            'total_return': total_ret,
            'profit': profit,
            'roi': roi,
            'bet_count': len(hist_df),
            'final_bankroll': int(self.bankroll)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default='2025-01-01')
    parser.add_argument("--end_date", type=str, default='2025-12-31')
    parser.add_argument("--fraction", type=float, default=0.1, help="Kelly criterion fraction (default 0.1)")
    parser.add_argument("--threshold", type=float, default=1.2, help="EV threshold (default 1.2)")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Minimum probability cutoff (e.g. 0.05)")
    parser.add_argument("--min_odds", type=float, default=0.0, help="Minimum odds cutoff (e.g. 2.0)")
    args = parser.parse_args()

    sim = KellySimulator(RANKER_MODEL_PATH, kelly_fraction=args.fraction, threshold=args.threshold, min_prob=args.min_prob, min_odds=args.min_odds)
    sim.run_simulation(args.start_date, args.end_date)
