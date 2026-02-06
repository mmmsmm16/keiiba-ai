
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import sys
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_PATH = "models/experiments/optuna_best_full/model.pkl"

def load_data():
    logger.info("Loading test data (2024)...")
    df = pd.read_parquet(DATA_PATH)
    
    # Merge valid odds from T1
    logger.info("Merging valid odds from data/temp_t1/T1_features_2024_2025.parquet...")
    try:
         t1_data = pd.read_parquet("data/temp_t1/T1_features_2024_2025.parquet")
         t1_data['race_id'] = t1_data['race_id'].astype(str)
         if 'odds_final' in t1_data.columns:
             t1_data = t1_data.rename(columns={'odds_final': 'odds_raw'})
         elif 'odds' in t1_data.columns:
             t1_data = t1_data.rename(columns={'odds': 'odds_raw'})
         
         df = df.merge(t1_data[['race_id', 'horse_number', 'odds_raw']], 
                       on=['race_id', 'horse_number'], how='left')
         
         if 'odds_final' not in df.columns:
             df['odds_final'] = df['odds_raw']
         else:
             df['odds_final'] = df['odds_final'].replace(0, np.nan)
             df['odds_final'] = df['odds_final'].fillna(df['odds_raw'])
             
    except Exception as e:
         logger.warning(f"Failed to merge T1 results: {e}")

    df['date'] = pd.to_datetime(df['date'])
    df_test = df[df['date'].dt.year == 2024].copy()
    
    # Filter JRA (Venue 01-10)
    if 'venue' in df_test.columns:
        df_test['venue'] = df_test['venue'].astype(str).str.zfill(2)
        jra_mask = df_test['venue'].isin([str(i).zfill(2) for i in range(1, 11)])
        df_test = df_test[jra_mask]
    else:
        df_test['venue_extracted'] = df_test['race_id'].astype(str).str[4:6]
        jra_mask = df_test['venue_extracted'].isin([str(i).zfill(2) for i in range(1, 11)])
        df_test = df_test[jra_mask]
        
    if 'odds_final' not in df_test.columns:
         if 'odds' in df_test.columns: df_test['odds_final'] = df_test['odds']
         else: raise ValueError("No odds column found")
         
    return df_test

def main():
    df = load_data()
    logger.info(f"Test Data: {len(df)} rows, {df['race_id'].nunique()} races")
    
    # Load Model
    logger.info(f"Loading Win Model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    feature_names = model.feature_name()
    X_test = df[feature_names].copy()
    
    # Preprocess exactly as in training (run_train_optuna_best.py)
    # Convert object/category to codes
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category' or X_test[col].dtype == 'object':
             X_test[col] = X_test[col].astype('category').cat.codes
        else:
             X_test[col] = X_test[col].fillna(-999)

    logger.info("Predicting probabilities...")
    preds = model.predict(X_test)
    df['prob'] = preds
    
    # Calculate EV
    df['ev'] = df['prob'] * df['odds_final']
    
    # Rank predictions within race
    df['race_id'] = df['race_id'].astype(str)
    df['pred_rank'] = df.groupby('race_id')['prob'].rank(ascending=False, method='first')
    
    # Valid Payout logic (Rank 1 & Valid Odds)
    df['payout'] = df.apply(
        lambda x: x['odds_final'] * 100 if x['rank'] == 1 and pd.notnull(x['odds_final']) else 0, 
        axis=1
    )
    
    # Extract Rank 1 and Rank 2
    r1 = df[df['pred_rank'] == 1].set_index('race_id')[['ev', 'payout', 'rank', 'odds_final']]
    r2 = df[df['pred_rank'] == 2].set_index('race_id')[['ev', 'payout', 'rank', 'odds_final']]
    
    # Rename cols
    r1 = r1.rename(columns={'ev': 'ev1', 'payout': 'payout1', 'rank': 'rank1', 'odds_final': 'odds1'})
    r2 = r2.rename(columns={'ev': 'ev2', 'payout': 'payout2', 'rank': 'rank2', 'odds_final': 'odds2'})
    
    # Join
    sim_df = r1.join(r2, how='inner')
    logger.info(f"Races for simulation: {len(sim_df)}")
    
    # --- Composite Strategy Simulation ---
    # Strategy:
    # 1. Bet Rank 1 if ev1 >= 1.5
    # 2. ELSE (if ev1 < 1.5), Bet Rank 2 if ev2 >= Threshold
    
    # Define Base Bets (Rank 1, EV >= 1.5)
    base_bets_mask = sim_df['ev1'] >= 1.5
    
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]
    results = []
    
    logger.info("Running Composite Grid Search...")
    
    for th in thresholds:
        # Bets on Rank 2: where Rank 1 failed base condition AND Rank 2 meets threshold
        # Condition: (NOT base_bets_mask) AND (ev2 >= th)
        r2_bets_mask = (~base_bets_mask) & (sim_df['ev2'] >= th)
        
        # Total Bets
        n_bets_r1 = base_bets_mask.sum()
        n_bets_r2 = r2_bets_mask.sum()
        total_bets = n_bets_r1 + n_bets_r2
        
        # Returns
        ret_r1 = sim_df.loc[base_bets_mask, 'payout1'].sum()
        ret_r2 = sim_df.loc[r2_bets_mask, 'payout2'].sum()
        total_return = ret_r1 + ret_r2
        
        cost = total_bets * 100
        roi = (total_return / cost * 100) if cost > 0 else 0
        profit = total_return - cost
        
        # Hit Rates
        # Hit if bet on R1 and R1 wins (rank1==1) OR bet on R2 and R2 wins (rank2==1)
        # Note: payout column is already 0 if rank!=1, so using payout > 0 counts hits roughly, 
        # but technically we should check rank for accuracy stats.
        hits_r1 = (sim_df.loc[base_bets_mask, 'rank1'] == 1).sum()
        hits_r2 = (sim_df.loc[r2_bets_mask, 'rank2'] == 1).sum()
        total_hits = hits_r1 + hits_r2
        hit_rate = (total_hits / total_bets * 100) if total_bets > 0 else 0
        
        results.append({
            "R2_EV_Th": th,
            "TotalBets": total_bets,
            "R1_Bets": n_bets_r1,
            "R2_Bets": n_bets_r2,
            "HitRate": hit_rate,
            "ROI": roi,
            "Profit": profit,
            "Hits": total_hits
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(" 🔄 Composite EV Strategy (Rank 1 EV >= 1.5 OR Rank 2 EV >= Threshold)")
    print("="*80)
    print(res_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*80)
    
    best_roi = res_df.loc[res_df['TotalBets'] > 100].sort_values('ROI', ascending=False).iloc[0]
    best_profit = res_df.sort_values('Profit', ascending=False).iloc[0]
    
    print(f"\n🏆 Best ROI: R2 EV >= {best_roi['R2_EV_Th']} -> ROI {best_roi['ROI']:.2f}% (Profit: {best_roi['Profit']:.0f})")
    print(f"💰 Best Profit: R2 EV >= {best_profit['R2_EV_Th']} -> Profit {best_profit['Profit']:.0f} (ROI {best_profit['ROI']:.2f}%)")

if __name__ == "__main__":
    main()
