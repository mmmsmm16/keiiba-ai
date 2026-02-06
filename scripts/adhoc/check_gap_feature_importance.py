"""
Gap Model Feature Importance Analysis
=====================================
Check which features the Gap model is relying on most heavily.
Flag suspicious features that may be result-based (leakage).
"""
import pandas as pd
import numpy as np
import joblib

# Load the Gap model
model = joblib.load('models/experiments/exp_gap_prediction_reg/model.pkl')

# Get feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

# Create DataFrame
df_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Top 30 features
print("=" * 60)
print("Top 30 Most Important Features in Gap Model")
print("=" * 60)
print(df_imp.head(30).to_string())

# Flag suspicious keywords
suspicious_keywords = [
    'time', 'corner', 'pace', 'agari', 'pci', 'pos', 
    'margin', 'late', 'makuri', 'charge', 'result'
]

print("\n" + "=" * 60)
print("Suspicious Features (May Be Result-Based)")
print("=" * 60)
suspicious = df_imp[df_imp['feature'].apply(
    lambda x: any(k in x.lower() for k in suspicious_keywords)
)]
print(suspicious.to_string())

# Total importance of suspicious features
total_imp = df_imp['importance'].sum()
suspicious_imp = suspicious['importance'].sum()
print(f"\n⚠️ Suspicious features account for {suspicious_imp/total_imp*100:.1f}% of total importance")
