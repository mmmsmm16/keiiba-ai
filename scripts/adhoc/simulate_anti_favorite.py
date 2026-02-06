"""
Simulate "Anti-Favorite" Strategy
=================================
Tests ROI of betting on the Model's Top 1 prediction, 
BUT skipping if that horse is the 1st favorite (popularity == 1).
"""
import pandas as pd
import numpy as np
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df_test = df[df['year'] == 2024].copy()
df_test = df_test[(df_test['odds'] > 0) & (df_test['odds'].notna())].reset_index(drop=True)

# 1. LambdaRank (Best Model)
print("\n=== 1. LambdaRank (Best Model) ===")
lgbm_model_path = 'models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl'
lgbm_features_path = 'models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv'

if os.path.exists(lgbm_model_path):
    model = joblib.load(lgbm_model_path)
    features = pd.read_csv(lgbm_features_path)['0'].tolist()
    
    # Predict
    y_pred = model.predict(df_test[features].values)
    df_test['lgbm_score'] = y_pred
    df_test['lgbm_rank'] = df_test.groupby('race_id')['lgbm_score'].rank(ascending=False)
    
    # Base Top 1
    top1 = df_test[df_test['lgbm_rank'] == 1]
    wins = top1[top1['rank'] == 1]
    roi_base = wins['odds'].sum() / len(top1) * 100
    print(f"Base Top 1 ROI: {roi_base:.2f}% (Bets: {len(top1)})")
    
    # Anti-Favorite (Skip if popularity == 1)
    # Check if 'popularity' exists, if not use odds rank
    if 'popularity' in df_test.columns:
        top1_filtered = top1[top1['popularity'] != 1]
    else:
        print("Warning: 'popularity' column not found. Using odds rank.")
        df_test['odds_rank'] = df_test.groupby('race_id')['odds'].rank(ascending=True)
        top1_filtered = df_test[(df_test['lgbm_rank'] == 1) & (df_test['odds_rank'] != 1)]
        
    wins_filtered = top1_filtered[top1_filtered['rank'] == 1]
    roi_filtered = wins_filtered['odds'].sum() / len(top1_filtered) * 100
    print(f"Anti-Favorite Top 1 ROI: {roi_filtered:.2f}% (Bets: {len(top1_filtered)})")
    print(f"Diff: {roi_filtered - roi_base:.2f} pts")

# 2. Weighted TabNet (ROI 81.6%)
print("\n=== 2. Weighted TabNet (ROI 81.6%) ===")
tabnet_dir = 'models/experiments/exp_tabnet_v12_weighted'
tabnet_zip = os.path.join(tabnet_dir, 'tabnet_model_weighted.zip')

if os.path.exists(tabnet_zip):
    # Load artifacts
    feature_order = joblib.load(os.path.join(tabnet_dir, 'feature_order.pkl'))
    label_encoders = joblib.load(os.path.join(tabnet_dir, 'label_encoders.pkl'))
    fill_values = joblib.load(os.path.join(tabnet_dir, 'fill_values.pkl'))
    
    # Preprocess TabNet Test Data
    df_tab = df_test.copy()
    
    # Categorical
    cat_cols = [c for c in feature_order if c in label_encoders]
    for col in cat_cols:
        le = label_encoders[col]
        known = set(le.classes_)
        df_tab[col] = df_tab[col].astype(str).apply(lambda x: le.transform([x])[0] if x in known else 0)

    # Numerical
    num_cols = [c for c in feature_order if c not in cat_cols]
    for col in num_cols:
        val = fill_values[col]
        if pd.isna(val): val = 0
        df_tab[col] = df_tab[col].fillna(val)

    X_test_tab = df_tab[feature_order].values
    
    # Predict
    clf = TabNetClassifier()
    clf.load_model(tabnet_zip)
    preds = clf.predict_proba(X_test_tab)[:, 1]
    df_test['tab_prob'] = preds
    
    # Normalize
    def normalize_probs(group):
        p = group['tab_prob']
        return p / p.sum()
    df_test['tab_prob_norm'] = df_test.groupby('race_id', group_keys=False).apply(normalize_probs)
    df_test['tab_rank'] = df_test.groupby('race_id')['tab_prob_norm'].rank(ascending=False)
    
    # Base Top 1
    top1 = df_test[df_test['tab_rank'] == 1]
    wins = top1[top1['rank'] == 1]
    roi_base = wins['odds'].sum() / len(top1) * 100
    print(f"Base Top 1 ROI: {roi_base:.2f}% (Bets: {len(top1)})")
    
    # Anti-Favorite
    if 'popularity' in df_test.columns:
        top1_filtered = top1[top1['popularity'] != 1]
    else:
        top1_filtered = df_test[(df_test['tab_rank'] == 1) & (df_test['odds_rank'] != 1)]
        
    wins_filtered = top1_filtered[top1_filtered['rank'] == 1]
    roi_filtered = wins_filtered['odds'].sum() / len(top1_filtered) * 100
    print(f"Anti-Favorite Top 1 ROI: {roi_filtered:.2f}% (Bets: {len(top1_filtered)})")
    print(f"Diff: {roi_filtered - roi_base:.2f} pts")
    
else:
    print("Weighted TabNet model not found.")
