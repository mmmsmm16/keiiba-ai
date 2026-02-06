
"""
Compare Models ROI (Win Only)
=============================
Simulates betting on the predicted 1st place horse (単勝べた買い) for each model.
Models:
1. Win Model (optuna_best_full)
2. Top2 Model (optuna_top2)
3. Top3 Model (optuna_top3)
4. LambdaRank Model (exp_lambdarank)

Population: 2024 Test Data
Bet: 100 yen on the horse with the highest score in each race.
"""
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

MODEL_PATHS = {
    "Win Model": "models/experiments/optuna_best_full/model.pkl",
    "Top2 Model": "models/experiments/optuna_top2/model.pkl",
    "Top3 Model": "models/experiments/optuna_top3/model.pkl",
    "LambdaRank": "models/experiments/exp_lambdarank/model.pkl"
}

def load_data():
    logger.info("Loading test data (2024)...")
    df = pd.read_parquet(DATA_PATH)
    # If rank/odds missing or have zeros, try to merge from data/temp_t1/T1_features_2024_2025.parquet
    logger.info("Merging valid odds from data/temp_t1/T1_features_2024_2025.parquet...")
    try:
         t1_data = pd.read_parquet("data/temp_t1/T1_features_2024_2025.parquet")
         t1_data['race_id'] = t1_data['race_id'].astype(str)
         
         # T1 usually has 'odds_final'
         if 'odds_final' in t1_data.columns:
             t1_data = t1_data.rename(columns={'odds_final': 'odds_raw'})
         elif 'odds' in t1_data.columns:
             t1_data = t1_data.rename(columns={'odds': 'odds_raw'})
         
         # Merge
         df = df.merge(t1_data[['race_id', 'horse_number', 'odds_raw']], 
                       on=['race_id', 'horse_number'], how='left')
         
         # Overwrite odds_final with valid odds_raw
         if 'odds_final' not in df.columns:
             df['odds_final'] = df['odds_raw']
         else:
             # Fill 0 or NaN with raw values
             df['odds_final'] = df['odds_final'].replace(0, np.nan)
             df['odds_final'] = df['odds_final'].fillna(df['odds_raw'])
             
    except Exception as e:
         logger.warning(f"Failed to merge T1 results: {e}")

    df_test = df[df['date'].dt.year == 2024].copy()
    
    # Filter for JRA Races Only (Venue 01-10)
    # Venue code is usually in 'venue' column or 4-6 chars of race_id
    # Check if 'venue' column exists and is reliable
    if 'venue' in df_test.columns:
        # Venue codes in JRA are 01-10 (sometimes strings, sometimes ints)
        # Ensure string '01', '10' etc
        df_test['venue'] = df_test['venue'].astype(str).str.zfill(2)
        jra_mask = df_test['venue'].isin([str(i).zfill(2) for i in range(1, 11)])
        n_nar = (~jra_mask).sum()
        if n_nar > 0:
            logger.info(f"Filtering non-JRA races: Removing {n_nar} NAR/Overseas records.")
            df_test = df_test[jra_mask]
    else:
        # Fallback to race_id cleaning
        # race_id: YYYY(4) + Venue(2)
        df_test['venue_extracted'] = df_test['race_id'].astype(str).str[4:6]
        jra_mask = df_test['venue_extracted'].isin([str(i).zfill(2) for i in range(1, 11)])
        n_nar = (~jra_mask).sum()
        if n_nar > 0:
            logger.info(f"Filtering non-JRA races (by ID): Removing {n_nar} NAR/Overseas records.")
            df_test = df_test[jra_mask]

    # Ensure rank and odds are present
    if 'rank' not in df_test.columns or 'odds_final' not in df_test.columns:
        if 'odds' in df_test.columns and 'odds_final' not in df_test.columns:
             df_test['odds_final'] = df_test['odds']
             
    if 'rank' not in df_test.columns or 'odds_final' not in df_test.columns:
        raise ValueError("Test data missing 'rank' or 'odds_final' columns.")
        
    # Check for remaining zeros
    n_zeros = (df_test['odds_final'] == 0).sum()
    if n_zeros > 0:
        logger.warning(f"⚠️ Warning: {n_zeros} records still have 0.0 odds in 2024 JRA data!")
    
    # Needs race_id for grouping
    df_test['race_id'] = df_test['race_id'].astype(str)
    
    logger.info(f"Test Data: {len(df_test)} rows, {df_test['race_id'].nunique()} races")
    return df_test

def simulate_roi(df, model_name, model_path):
    logger.info(f"Evaluating {model_name}...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None
        
    # Prepare features
    # (Assuming models use typical feature set - excluding meta/leakage)
    # Most models save feature_name() or we can infer.
    # LightGBM models usually store feature names.
    
    try:
        feature_names = model.feature_name()
    except:
        # If model is sklearn wrapper pipeline or something else
        try:
            feature_names = model.booster_.feature_name()
        except:
             logger.error(f"Could not retrieve feature names for {model_name}")
             return None
             
    X_test = df[feature_names].copy()
    
    # Handle categoricals
    for c in X_test.columns:
        if X_test[c].dtype.name == 'category' or X_test[c].dtype == 'object':
             X_test[c] = X_test[c].astype('category').cat.codes
        # Simple fillna for safety (models should handle or match training)
        if hasattr(model, "predict_proba") or hasattr(model, "predict"):
             # For some models, NaN might be an issue if not handled same way
             # But our training scripts usually handle it. 
             # Let's hope lgb handles it naturally or the loaded model has info.
             pass
    
    # Predict
    if hasattr(model, "predict_proba"):
        # Binary Classification -> Prob of class 1
        preds = model.predict_proba(X_test)[:, 1]
    else:
        # Ranking or Regression -> Raw score
        preds = model.predict(X_test)
        
    df_pred = df[['race_id', 'horse_number', 'rank', 'odds_final']].copy()
    df_pred['score'] = preds
    
    # Select Top 1 per race
    # Sort by race_id, score descending
    df_pred = df_pred.sort_values(['race_id', 'score'], ascending=[True, False])
    
    # Group by race and take first
    top1 = df_pred.groupby('race_id').head(1).copy()
    
    # Calculate Metrics
    n_bets = len(top1)
    n_hits = len(top1[top1['rank'] == 1])
    total_cost = n_bets * 100
    
    # Payout: 100 * odds if rank 1, else 0
    top1['payout'] = top1.apply(lambda x: x['odds_final'] * 100 if x['rank'] == 1 else 0, axis=1)
    total_return = top1['payout'].sum()
    
    # --- DIAGNOSTICS ---
    hits = top1[top1['rank'] == 1]
    logger.info(f"Diagnostics for {model_name}:")
    logger.info(f"  Hit Odds Mean: {hits['odds_final'].mean():.2f}")
    logger.info(f"  Hit Odds Median: {hits['odds_final'].median():.2f}")
    logger.info(f"  Hit Odds Min/Max: {hits['odds_final'].min():.2f} / {hits['odds_final'].max():.2f}")
    logger.info(f"  Top 10 Payouts: {hits['payout'].nlargest(5).tolist()}")
    logger.info(f"  Zero Payouts (Hit but 0 odds?): {(hits['odds_final'] == 0).sum()}")
    logger.info(f"  NaN Odds (Hit but NaN odds?): {hits['odds_final'].isna().sum()}")
    # -------------------
    
    roi = (total_return / total_cost) * 100 if total_cost > 0 else 0
    hit_rate = (n_hits / n_bets) * 100 if n_bets > 0 else 0
    
    return {
        "Model": model_name,
        "Bets": n_bets,
        "Hits": n_hits,
        "HitRate": f"{hit_rate:.2f}%",
        "Cost": total_cost,
        "Return": total_return,
        "ROI": f"{roi:.2f}%",
        "Profit": total_return - total_cost
    }

def main():
    logger.info("="*60)
    logger.info("🏇 MODEL COMPARISON: WIN ONLY ROI (2024)")
    logger.info("="*60)
    
    df = load_data()
    
    results = []
    for name, path in MODEL_PATHS.items():
        res = simulate_roi(df, name, path)
        if res:
            results.append(res)
            
    # Display Results
    if results:
        res_df = pd.DataFrame(results)
        # Reorder columns
        cols = ["Model", "ROI", "HitRate", "Profit", "Return", "Cost", "Bets", "Hits"]
        res_df = res_df[cols]
        
        print("\n" + "="*80)
        print(res_df.to_string(index=False))
        print("="*80)
        
        # Best model?
        best = res_df.sort_values("Profit", ascending=False).iloc[0]
        print(f"\n🏆 Best Model (Profit): {best['Model']} (ROI: {best['ROI']})")

if __name__ == "__main__":
    main()
