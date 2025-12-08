
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from scipy.special import softmax
from scipy.stats import entropy
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.betting_strategy import BettingSimulator
from model.evaluate import load_payout_data
# from model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_race_features(df):
    """
    レース単位の特徴量を計算
    """
    race_feats = []
    
    logger.info(f"CalcFeatures DF Cols: {df.columns.tolist()}")
    logger.info(f"CalcFeatures Input Size: {len(df)}")
    for race_id, group in df.groupby(df['race_id']):
        probs = group['prob'].values
        odds = group['odds'].fillna(0).values
        
        # 1. Entropy (Confusion)
        ent = entropy(probs)
        
        # 2. Odds Volatility (Standard Deviation)
        odds_std = np.std(odds)
        
        # 3. Model Confidence (Max Prob)
        max_prob = np.max(probs)
        
        # 4. Confidence Gap (1st - 2nd)
        sorted_probs = sorted(probs, reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
        
        # 5. Number of horses
        n_horses = len(group)
        
        race_feats.append({
            'race_id': race_id,
            'entropy': ent,
            'odds_std': odds_std,
            'max_prob': max_prob,
            'confidence_gap': gap,
            'n_horses': n_horses
        })
        
    res_df = pd.DataFrame(race_feats)
    if res_df.empty:
        # Ensure columns exist to prevent merge error
        return pd.DataFrame(columns=['race_id', 'entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses'])
    return res_df

def main():
    parser = argparse.ArgumentParser(description='Train Betting Decision Model')
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.parquet')
    parser.add_argument('--model_dir', type=str, default='models')
    args = parser.parse_args()

    # 1. Load Data
    logger.info("Loading data...")
    df = pd.read_parquet(args.input)
    # Ensure race_id is a column
    if 'race_id' not in df.columns:
        df = df.reset_index()
    
    # 2. Load LightGBM Model (Faster scoring)
    logger.info("Loading LightGBM Model for scoring...")
    from model.lgbm import KeibaLGBM
    model = KeibaLGBM()
    model_path = os.path.join(args.model_dir, 'lgbm.pkl')
    
    if not os.path.exists(model_path):
        logger.error(f"LGBM model not found at {model_path}")
        return
        
    model.load_model(model_path)
    
    # Feature Extraction logic
    # Use 2020-2023 for Training, 2024 for Valid
    train_df = df[df['year'].isin([2020, 2021, 2022, 2023])].copy()
    valid_df = df[df['year'] == 2024].copy()
    
    logger.info(f"DEBUG: Train DF Cols Initial: {train_df.columns.tolist()}") 
    
    # Predict Scores
    feature_cols = None
    if hasattr(model.model, 'feature_name'):
         feature_cols = model.model.feature_name()

    logger.info(f"Train DF Columns: {train_df.columns.tolist()}")
    if 'race_id' not in train_df.columns:
        logger.error("race_id missing from train_df!")

    if feature_cols:
        missing = set(feature_cols) - set(train_df.columns)
        for c in missing: train_df[c] = 0
        for c in missing: valid_df[c] = 0
        # Ensure we don't overwrite train_df, just pick features
        X_train = train_df[feature_cols]
        X_valid = valid_df[feature_cols]
    else:
        # Fallback
        exclude = ['race_id', 'horse_number', 'horse_id', 'jockey_id', 'trainer_id', 'rank', 'time', 'odds', 'popularity', 'year']
        X_train = train_df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude if c in train_df.columns])
        X_valid = valid_df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude if c in valid_df.columns])

    logger.info("Generating scores (LGBM)...")
    # Predict
    train_scores = model.predict(X_train)
    valid_scores = model.predict(X_valid)
    
    # Reconstruct Clean DataFrames for Context Calculation
    # Step 1: Select rows (all columns)
    clean_train = df.loc[train_df.index].copy()
    clean_valid = df.loc[valid_df.index].copy()
    
    # Step 2: Ensure race_id is column
    if 'race_id' not in clean_train.columns:
        clean_train = clean_train.reset_index()
    if 'race_id' not in clean_valid.columns:
        clean_valid = clean_valid.reset_index()

    # Step 3: Select columns + Score
    cols_to_keep = ['race_id', 'odds', 'horse_number']
    clean_train = clean_train[cols_to_keep].copy()
    clean_valid = clean_valid[cols_to_keep].copy()
    
    clean_train['score'] = train_scores
    clean_valid['score'] = valid_scores
    
    logger.info(f"CleanTrain Cols: {clean_train.columns.tolist()}")
    
    # Probabilities
    # Use explicit Series for groupby to avoid lookup errors
    if 'race_id' in clean_train.columns:
        clean_train['prob'] = clean_train.groupby(clean_train['race_id'])['score'].transform(lambda x: softmax(x))
        clean_valid['prob'] = clean_valid.groupby(clean_valid['race_id'])['score'].transform(lambda x: softmax(x))
    else:
        logger.error("CRITICAL: race_id still missing in clean_train!")

    # 3. Create Race Features
    logger.info("Calculating Race Context Features...")
    # Use clean DFs
    train_race_feats = calculate_race_features(clean_train)
    valid_race_feats = calculate_race_features(clean_valid)
    
    # 4. Define Target (Did "Sanrenpuku Formation" hit?)
    logger.info("Defining Targets...")
    # Load Payouts
    year_list = df['year'].unique()
    logger.info(f"Loading Payouts for years: {year_list}")
    payout_dfs = []
    for y in year_list:
        p_df = load_payout_data(year=y)
        if not p_df.empty:
            payout_dfs.append(p_df)
            
    if not payout_dfs:
        logger.error("No payout data found!")
        return

    payout_df = pd.concat(payout_dfs)
    
    # Helper to check hit
    # Logic: For each race, check if [Axis=1st by Score, Opp=2-6th by Score] hits Sanrenpuku.
    
    targets = []
    
    # Merge Features and Target
    # Prepare Targets using simulation on Clean Data
    logger.info("Defining Targets using Clean Data...")
    
    all_df = pd.concat([clean_train, clean_valid])
    sim = BettingSimulator(all_df, payout_df)
    
    # Run simple simulation per race to get hit status
    strategies = []
    # Using the standard strategy as Target Reference
    axis_rank = 1
    opp_ranks = [2,3,4,5,6]
    
    logger.info(f"All DF Size: {len(all_df)}")
    if not all_df.empty:
        logger.info(f"All DF Head: {all_df.head()}")
        logger.info(f"All DF RaceID Dtype: {all_df['race_id'].dtype}")
        logger.info(f"All DF Unique Races: {all_df['race_id'].nunique()}")

    for race_id, group in all_df.groupby(all_df['race_id']):
        if race_id not in sim.payout_map:
            continue
            
        sorted_horses = group.sort_values('score', ascending=False)
        if len(sorted_horses) < 6: continue
        
        axis = int(sorted_horses.iloc[axis_rank-1]['horse_number'])
        opps = [int(h) for h in sorted_horses.iloc[1:6]['horse_number']] 
        
        results = sim.payout_map[race_id].get('sanrenpuku', {})
        is_hit = 0
        return_amount = 0
        
        for combo_str, payout in results.items():
            # Parse combo keys (assuming 6 digits for sanrenpuku "010203")
            # If length is insufficient, skip or handle (e.g. integer conversion stripped leading zero)
            # But BettingSimulator assumes :02 padding, so keys should be padded if _build_payout_map is consistent.
            # However, _build_payout_map uses str(row[...]). If DB returns int, it might lack padding.
            # Let's simple check membership.
            
            # Robust parsing?
            # Try to convert to padded string if needed?
            s = str(combo_str).zfill(6)
            try:
                horses = [int(s[i:i+2]) for i in range(0, 6, 2)]
            except:
                continue

            if axis in horses:
                others = [h for h in horses if h != axis]
                if all(o in opps for o in others):
                    is_hit = 1
                    return_amount += int(payout)
        
        cost = 10 * 100 
        is_profitable = 1 if return_amount > cost else 0
        
        strategies.append({
            'race_id': race_id,
            'is_hit': is_hit,
            'is_profitable': is_profitable,
            'return': return_amount
        })
        
    target_df = pd.DataFrame(strategies)
    if target_df.empty:
         target_df = pd.DataFrame(columns=['race_id', 'is_hit', 'is_profitable', 'return'])
    
    # Merge Features and Target
    logger.info(f"Train Feats Cols: {train_race_feats.columns.tolist()}, Shape: {train_race_feats.shape}")
    logger.info(f"Target DF Cols: {target_df.columns.tolist()}, Shape: {target_df.shape}")
    
    train_data = train_race_feats.merge(target_df, on='race_id')
    valid_data = valid_race_feats.merge(target_df, on='race_id')
    
    # 5. Train Logic
    logger.info("Training Betting Model...")
    
    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
    target = 'is_profitable' # Trying to predict broken races where we win big? Or just hit?
    # Better: Predict 'is_hit' (Stability) or 'is_profitable' (ROI).
    # Let's try 'is_profitable'.
    
    lgb_train = lgb.Dataset(train_data[features], train_data[target])
    lgb_valid = lgb.Dataset(valid_data[features], valid_data[target], reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    bst = lgb.train(params, lgb_train, valid_sets=[lgb_valid], num_boost_round=100,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)])
    
    # Evaluate
    preds = bst.predict(valid_data[features])
    valid_data['pred_conf'] = preds
    auc = roc_auc_score(valid_data[target], preds)
    logger.info(f"Test AUC: {auc:.4f}")
    
    # Save Feature Importance
    importance = pd.DataFrame({'feature': features, 'gain': bst.feature_importance(importance_type='gain')})
    logger.info(f"Feature Importance:\n{importance.sort_values('gain', ascending=False)}")

    # Save Model
    out_path = os.path.join(args.model_dir, 'betting_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(bst, f)
    logger.info(f"Model saved to {out_path}")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
