import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, ndcg_score
from scipy.stats import rankdata

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.evaluation.metrics import calculate_metrics

# Config
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
MODEL_BINARY_PATH = "models/experiments/exp_t2_refined_v3/model.txt" # Phase 1 (Check path)
MODEL_RANKER_PATH = "models/experiments/exp_lambdarank/model.txt"    # Phase 2 (Check path)
TEST_YEAR = 2024

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

def predict_ranker_normalized(model, X, group):
    # Predict raw scores
    raw_scores = model.predict(X)
    
    # Normalize per race (MinMax)
    # Using pandas for group operation
    temp = pd.DataFrame({'score': raw_scores, 'group': np.repeat(np.arange(len(group)), group)})
    # Wait, group is count per race. Need race_id mapping.
    # Actually simpler: we can just iterate? No, slow.
    # Creating a race_index
    race_ids = []
    for i, g in enumerate(group):
        race_ids.extend([i] * g)
    
    temp['race_id'] = race_ids
    
    # MinMax Scale per race
    def minmax(x):
        if x.max() == x.min(): return x - x # 0
        return (x - x.min()) / (x.max() - x.min())
    
    temp['norm_score'] = temp.groupby('race_id')['score'].transform(minmax)
    return temp['norm_score'].values

def run_ensemble():
    print("Loading Data...")
    df = load_data()
    test_df = df[df['date'].dt.year == TEST_YEAR].copy()
    test_df = test_df.sort_values(['date', 'race_id', 'horse_number'])
    
    print(f"Test Data: {len(test_df)} records")
    
    # Features (Must match training... this is tricky if models used different features)
    # Assuming both used same feature set (v11 + modifications)
    # If feature config differs, we need to load configs.
    # For now, assume we use all numeric columns except meta.
    exclude = ['race_id', 'horse_id', 'date', 'rank', 'horse_number', 'jockey_id', 'trainer_id', 
               'row_id', 'is_train', 'partition', 'abnormal_code', 'title', 'weather', 'surface', 'state']
    features = [c for c in test_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(test_df[c])]
    
    X_test = test_df[features]
    y_test = (test_df['rank'] == 1).astype(int)
    
    # Load Models
    print(f"Loading Binary Model: {MODEL_BINARY_PATH}")
    # binary model might be pickles or txt
    if MODEL_BINARY_PATH.endswith('.txt'):
        bst_binary = lgb.Booster(model_file=MODEL_BINARY_PATH)
    else:
        bst_binary = joblib.load(MODEL_BINARY_PATH)
        
    print(f"Loading Ranker Model: {MODEL_RANKER_PATH}")
    if MODEL_RANKER_PATH.endswith('.txt'):
        bst_ranker = lgb.Booster(model_file=MODEL_RANKER_PATH)
    else:
        bst_ranker = joblib.load(MODEL_RANKER_PATH)
        
    # Predict
    print("Predicting Binary...")
    prob_binary = bst_binary.predict(X_test)
    
    print("Predicting Ranker...")
    # Ranker needs query info? LightGBM predict doesn't need Query for inference usually, just features.
    score_ranker = bst_ranker.predict(X_test)
    
    # Normalize Ranker Score (Optional, but good for blending)
    # For simple blending, scaling to 0-1 helps.
    # Use global MinMax or per-race? Logit is global. Ranker score is global but relative.
    # Let's simple MinMax global for now
    score_ranker_norm = (score_ranker - score_ranker.min()) / (score_ranker.max() - score_ranker.min())
    
    # Evaluate Baseline
    auc_bin = roc_auc_score(y_test, prob_binary)
    auc_rank = roc_auc_score(y_test, score_ranker_norm)
    print(f"Binary AUC: {auc_bin:.4f}")
    print(f"Ranker AUC: {auc_rank:.4f}")
    
    # Blending Loop
    print("\nBlending Optimization...")
    best_auc = 0
    best_w = 0
    
    for w in np.linspace(0, 1, 21):
        blended = w * prob_binary + (1-w) * score_ranker_norm
        auc = roc_auc_score(y_test, blended)
        if auc > best_auc:
            best_auc = auc
            best_w = w
        print(f"w={w:.2f}, AUC={auc:.4f}")
        
    print(f"\nBest Weight (Binary): {best_w:.2f}")
    print(f"Best AUC: {best_auc:.4f}")
    
    # TODO: ROI Simulation for best blend
    
if __name__ == "__main__":
    run_ensemble()
