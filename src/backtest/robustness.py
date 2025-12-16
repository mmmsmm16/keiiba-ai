import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import json
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.validation import ExpandingWindowValidator
from model.lgbm import KeibaLGBM
from betting.purchase_model import PurchaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustnessValidator(ExpandingWindowValidator):
    def __init__(self, data_path, dataset_info_path, model_type='lgbm', model_params=None, drop_groups=None):
        super().__init__(data_path, dataset_info_path, model_type, model_params)
        self.drop_groups = drop_groups
        self.slippage_factors = [1.0, 0.95, 0.90, 0.85]
        self.robustness_results = []

    def load_data(self):
        super().load_data()
        if self.drop_groups:
            self._drop_features()

    def _drop_features(self):
        # Reuse logic from AblationRunner (simplified)
        current_features = self.feature_cols
        drop_list = []
        for feat in current_features:
            for group in self.drop_groups:
                is_hit = False
                if group == 'embedding' and '_emb_' in feat: is_hit = True
                if group == 'bloodline' and ('sire' in feat or 'blood' in feat or 'bms' in feat): is_hit = True
                if group == 'human' and ('jockey' in feat or 'trainer' in feat): is_hit = True
                if group == 'speed' and ('speed' in feat or 'time' in feat or '3f' in feat): is_hit = True
                if group == 'realtime' and 'trend' in feat: is_hit = True
                
                if is_hit:
                    drop_list.append(feat)
                    break
        
        self.feature_cols = [f for f in current_features if f not in drop_list]
        logger.info(f"Robustness: Dropped {len(drop_list)} features (Group: {self.drop_groups}). Remaining: {len(self.feature_cols)}")

    def run_cv(self):
        # Override to add Slippage Loop inside or summarize differently
        # Actually, let's run standard CV but calculate ROI for multiple slippages inside _calculate_roi
        # But _calculate_roi is called per fold.
        # We can modify _calculate_roi to return metrics for all slippages.
        super().run_cv()

    def _calculate_roi(self, race_ids, probs, odds, y_true):
        # Override to calc multiple slippage scenarios
        
        base_df = pd.DataFrame({
            'race_id': race_ids.values,
            'prob': probs,
            'odds': odds.values,
            'hit': y_true.values
        })
        
        pm = PurchaseModel()
        metrics = {}
        
        for slippage in self.slippage_factors:
            logger.info(f"  Calculating metrics for Slippage x{slippage}...")
            df = base_df.copy()
            
            # Apply Slippage to Odds
            # Model of Slippage: Effective Odds = Odds * Factor
            # Note: Do not change 'raw_prob' calculation if it represents "Market View"?
            # P_market should utilize the ORIGINAL odds (Market View). 
            # Slippage usually affects EXPECTED RETURN (your execution price).
            # So:
            # 1. Calc P_market using ORIGINAL Odds (Market consensus doesn't change because you slipped)
            # 2. Calc EV using SLIPPED Odds (Your return changes)
            
            # 1. Market Prob (Original Odds)
            df = pm.calculate_market_probability(df) 
            
            # 2. EV (Slipped Odds)
            df['odds_slipped'] = df['odds'] * slippage
            df = pm.calculate_expected_value(df, odds_col='odds_slipped') # EV based on execution price
            
            # Strategies
            # A. Naive (Top 1)
            picks_naive = df.loc[df.groupby('race_id')['prob'].idxmax()]
            bet_sum = len(picks_naive) * 100
            # Return is based on Slipped Odds
            return_sum = (picks_naive[picks_naive['hit'] == 1]['odds_slipped']).sum() * 100
            metrics[f'roi_naive_s{slippage}'] = return_sum / bet_sum * 100 if bet_sum > 0 else 0
            
            # B. Kelly
            # Decision based on Slipped EV? Yes, you should bet based on what you get.
            df_kelly = pm.apply_betting_strategy(df, strategy_name='kelly', bankroll=10000, fraction=0.1, odds_col='odds_slipped')
            bet_sum_kelly = df_kelly['bet_amount'].sum()
            return_sum_kelly = (df_kelly[df_kelly['hit'] == 1]['bet_amount'] * df_kelly[df_kelly['hit'] == 1]['odds_slipped']).sum()
            metrics[f'roi_kelly_s{slippage}'] = return_sum_kelly / bet_sum_kelly * 100 if bet_sum_kelly > 0 else 0
            
            # C. EV Flat
            df_flat = pm.apply_betting_strategy(df, strategy_name='flat', threshold=0.0, bet_amount=100) # uses 'ev' col which is already slipped
            bet_sum_flat = df_flat['bet_amount'].sum()
            return_sum_flat = (df_flat[df_flat['hit'] == 1]['bet_amount'] * df_flat[df_flat['hit'] == 1]['odds_slipped']).sum()
            metrics[f'roi_ev_flat_s{slippage}'] = return_sum_flat / bet_sum_flat * 100 if bet_sum_flat > 0 else 0

        # Maintain standard keys for logger compatibility (use 1.0)
        metrics['roi'] = metrics.get('roi_naive_s1.0', 0)
        metrics['roi_naive'] = metrics.get('roi_naive_s1.0', 0)
        metrics['roi_kelly'] = metrics.get('roi_kelly_s1.0', 0)
        metrics['roi_ev_flat'] = metrics.get('roi_ev_flat_s1.0', 0)
        
        return metrics

    def _print_summary(self):
        logger.info("="*60)
        logger.info("Robustness Analysis Summary (Mean over Folds)")
        logger.info("="*60)
        
        # Aggregate across folds
        df_res = pd.DataFrame([r['metrics'] for r in self.results])
        
        # Calculate Mean for each Slippage
        summary = []
        for s in self.slippage_factors:
            row = {'Slippage': s}
            row['Naive'] = df_res[f'roi_naive_s{s}'].mean()
            row['Kelly'] = df_res[f'roi_kelly_s{s}'].mean()
            row['EV_Flat'] = df_res[f'roi_ev_flat_s{s}'].mean()
            summary.append(row)
            
        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string(index=False, float_format="%.2f%%"))
        print("-" * 60)
        
        # Original Summary
        logger.info("Detailed Fold Results (Slippage 1.0)")
        super()._print_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lgbm')
    parser.add_argument('--suffix', type=str, default='_v11')
    parser.add_argument('--drop', nargs='+', help='Groups to drop')
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, f'../../data/processed/preprocessed_data{args.suffix}.parquet')
    dataset_info_path = os.path.join(base_dir, f'../../data/processed/lgbm_datasets{args.suffix}.pkl')
    # Params
    params_path = os.path.join(base_dir, f'../../models/params/{args.model}_v1_best_params.json')
    params = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: params = json.load(f)

    validator = RobustnessValidator(
        data_path, 
        dataset_info_path, 
        model_type=args.model, 
        model_params=params, 
        drop_groups=args.drop
    )
    validator.load_data()
    validator.run_cv()

if __name__ == "__main__":
    main()
