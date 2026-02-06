"""
SHAP Analysis for Gap Prediction Model
======================================
Analyze which features contribute most to predicting high "Gap" (Top 5 & High Odds).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set japanese font if possible, though in docker might be tough.
# We will output text summary primarily.

def main():
    MODEL_DIR = 'models/experiments/exp_gap_prediction_reg'
    
    print("Loading model and data...")
    model = joblib.load(f'{MODEL_DIR}/model.pkl')
    features = pd.read_csv(f'{MODEL_DIR}/features.csv')['0'].tolist()
    
    # Load sample data (Test set: 2024)
    df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Filter valid odds
    df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)
    
    # Inject minimal necessary features for context (not used in model but for filtering)
    df_test = df[df['year'] == 2024].copy()
    
    # Use only features used in model
    # Need to handle missing columns if any
    for c in features:
        if c not in df_test.columns:
            df_test[c] = 0
            
    X_test = df_test[features]
    
    # Use a subset for SHAP to save time (e.g. 2000 samples)
    # Ideally we want to look at cases where model predicted HIGH GAP.
    # So let's predict first.
    print("Predicting...")
    X_test_np = X_test.values.astype(np.float64)
    scores = model.predict(X_test_np)
    df_test['pred_score'] = scores
    
    # Filter for Top Predicted Horses (Gap Rank 1)
    df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)
    top_picks = df_test[df_test['pred_rank'] <= 3] # Top 3 candidates
    
    # Sample 1000 from Top Picks to analyze "Why did you pick these?"
    if len(top_picks) > 1000:
        shap_data = top_picks.sample(1000, random_state=42)
    else:
        shap_data = top_picks
        
    X_shap = shap_data[features]
    
    print(f"Calculating SHAP values for {len(X_shap)} samples (Top High-Gap Candidates)...")
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # LightGBM output can be list if multiclass, but this is lambdarank (regression-like).
    # Check shape
    if isinstance(shap_values, list):
        shap_values = shap_values[0] # Shouldn't happen for lambdarank usually, but strictly check
        
    # Summary
    print("\n=== Top 20 Features driving High Gap Predictions ===")
    
    # Mean absolute SHAP value
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20))
    
    # Save text report
    with open(f'{MODEL_DIR}/shap_summary.txt', 'w', encoding='utf-8') as f:
        f.write(feature_importance.to_string())
        
    print(f"\nSaved summary to {MODEL_DIR}/shap_summary.txt")
    
    # Analysis of Direction
    # Check correlation between Feature Value and SHAP Value for top features
    print("\n=== Directionality Analysis (Top 5) ===")
    for idx, row in feature_importance.head(5).iterrows():
        feat = row['feature']
        vals = X_shap[feat].values
        shaps = shap_values[:, features.index(feat)]
        corr = np.corrcoef(vals.astype(float), shaps)[0, 1]
        direction = "Positive (+)" if corr > 0 else "Negative (-)"
        print(f"{feat}: Correlation {corr:.2f} ({direction})")
        
if __name__ == "__main__":
    main()
