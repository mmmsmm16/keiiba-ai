
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
MODEL_PATH = "models/experiments/exp_lambdarank/model.pkl"

def load_data():
    logger.info("Loading test data (2024)...")
    df = pd.read_parquet(DATA_PATH)
    
    # Merge valid odds from T1 if needed, same as before
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

def softmax(series):
    # Exp safely (subtract max for stability)
    e_x = np.exp(series - series.max())
    return e_x / e_x.sum()

def main():
    df = load_data()
    logger.info(f"Test Data: {len(df)} rows, {df['race_id'].nunique()} races")
    
    # Load Model
    logger.info(f"Loading LambdaRank Model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    feature_names = model.feature_name()
    X_test = df[feature_names].copy()
    
    # Preprocess (cat.codes) as confirmed in previous steps
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category' or X_test[col].dtype == 'object':
             X_test[col] = X_test[col].astype('category').cat.codes
        else:
             X_test[col] = X_test[col].fillna(-999)

    logger.info("Predicting scores...")
    preds = model.predict(X_test)
    df['score'] = preds
    
    # Convert Score to Probability via Softmax per race
    logger.info("Calculating Softmax Probabilities...")
    df['prob'] = df.groupby('race_id')['score'].transform(softmax)
    
    # Calculate EV
    df['ev'] = df['prob'] * df['odds_final']
    
    # Select Top 1 (Sort by score/prob descending)
    df['pred_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first')
    top1 = df[df['pred_rank'] == 1].copy()
    
    # Grid Search EV
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]
    results = []
    
    logger.info("Running EV Grid Search (LambdaRank)...")
    for th in thresholds:
        bets = top1[top1['ev'] >= th].copy()
        
        n_bets = len(bets)
        n_hits = len(bets[bets['rank'] == 1])
        
        cost = n_bets * 100
        bets['payout'] = bets.apply(
            lambda x: x['odds_final'] * 100 if x['rank'] == 1 and pd.notnull(x['odds_final']) else 0, 
            axis=1
        )
        return_amount = bets['payout'].sum()
        
        roi = (return_amount / cost) * 100 if cost > 0 else 0
        hit_rate = (n_hits / n_bets) * 100 if n_bets > 0 else 0
        profit = return_amount - cost
        
        results.append({
            "EV_Threshold": th,
            "Bets": n_bets,
            "Hits": n_hits,
            "HitRate": hit_rate,
            "ROI": roi,
            "Profit": profit,
            "Return": return_amount
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(" 📊 EV Grid Search Results (LambdaRank - Softmax Prob)")
    print("="*80)
    print(res_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*80)
    
    best_roi = res_df.loc[res_df['Bets'] > 100].sort_values('ROI', ascending=False).iloc[0]
    best_profit = res_df.sort_values('Profit', ascending=False).iloc[0]
    
    print(f"\n🏆 Best ROI (min 100 bets): Threshold {best_roi['EV_Threshold']} -> ROI {best_roi['ROI']:.2f}% (Profit: {best_roi['Profit']:.0f})")
    print(f"💰 Best Profit: Threshold {best_profit['EV_Threshold']} -> Profit {best_profit['Profit']:.0f} (ROI {best_profit['ROI']:.2f}%)")

if __name__ == "__main__":
    main()
