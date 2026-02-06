"""
Evaluate Final TabNet Model (Optuna Version)
============================================
"""
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

EXP_NAME = 'exp_tabnet_v12_optuna'
OUTPUT_DIR = f'models/experiments/{EXP_NAME}'

print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)
df['target'] = (df['rank'] == 1).astype(int)

# Load metadata
feature_order = joblib.load(os.path.join(OUTPUT_DIR, 'feature_order.pkl'))
label_encoders = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
fill_values = joblib.load(os.path.join(OUTPUT_DIR, 'fill_values.pkl'))
best_params = joblib.load(os.path.join(OUTPUT_DIR, 'best_params.pkl'))

print(f"Best Params: {best_params}")

# Preprocess Test Data
print("Preprocessing Test Data...")
df_test = df[df['year'] == 2024].copy()

# Categorical
cat_cols = [c for c in feature_order if c in label_encoders]
for col in cat_cols:
    df_test[col] = df_test[col].astype(str).map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0) # Handle unseen? TabNet handles unseen via embedding if mapped to unknown index? 
    # Actually LabelEncoder transform assumes known. 
    # For safety, we should reuse the same transformation logic.
    # In the training script, we just did le.fit_transform(df[col]) on the WHOLE df.
    # So the whole df mapping is preserved in label_encoders if I saved them correctly.
    # Wait, in training script I did: 
    # for i, col in enumerate(cat_cols):
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col])
    #     label_encoders[col] = le
    # This transformed the dataframe IN PLACE.
    # But later I split into train/valid/test.
    # But now I'm reloading raw parquet.
    # So I need to apply the label encoders.
    
    le = label_encoders[col]
    # Handle unseen values by mapping to a standard value or just mode?
    # TabNet expects integers.
    # Best way: map known, else 0?
    # Simple approach:
    known = set(le.classes_)
    df_test[col] = df_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in known else 0) # 0 is usually 'safe' if 0 is a valid class.

# Numerical
num_cols = [c for c in feature_order if c not in cat_cols]
for col in num_cols:
    val = fill_values[col]
    if pd.isna(val): val = 0
    df_test[col] = df_test[col].fillna(val)

X_test = df_test[feature_order].values
y_test = df_test['target'].values

# Load Model
print("Loading Model...")
clf = TabNetClassifier()
clf.load_model(os.path.join(OUTPUT_DIR, 'tabnet_model_optuna.zip'))

# Predict
print("Predicting...")
raw_preds = clf.predict_proba(X_test)[:, 1]
df_test['raw_prob'] = raw_preds

# Normalize
def normalize_probs(group):
    p = group['raw_prob']
    return p / p.sum()

df_test['pred_prob'] = df_test.groupby('race_id', group_keys=False).apply(normalize_probs)

# Eval
auc = roc_auc_score(y_test, df_test['pred_prob'])
print(f"Test AUC (Normalized): {auc:.4f}")

# ROI
print("\nROI Check (Test 2024)")
df_test['ev'] = df_test['pred_prob'] * df_test['odds']
df_test['pred_rank'] = df_test.groupby('race_id')['pred_prob'].rank(ascending=False)

top1 = df_test[df_test['pred_rank'] == 1]
wins = top1[top1['target'] == 1]
roi = wins['odds'].sum() / len(top1) * 100
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {len(wins)/len(top1)*100:.1f}%)")

print("\nEV Filter (All Horses):")
for ev_thr in [0.8, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5]:
    bets = df_test[df_test['ev'] >= ev_thr]
    if len(bets) < 10: continue
    w = bets[bets['target'] == 1]
    roi = w['odds'].sum() / len(bets) * 100
    print(f"EV >= {ev_thr}: Bets={len(bets)}, ROI={roi:.1f}%")
