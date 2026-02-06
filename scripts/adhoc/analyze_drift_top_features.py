
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import matplotlib.pyplot as plt

def main():
    print("Loading model...")
    MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return
        
    model = joblib.load(MODEL_PATH)
    
    # Get Importance
    if hasattr(model, 'booster_'):
        booster = model.booster_
    else:
        booster = model
        
    importance = booster.feature_importance(importance_type='gain')
    feature_names = booster.feature_name()
    
    df_imp = pd.DataFrame({'feature': feature_names, 'gain': importance})
    df_imp = df_imp.sort_values('gain', ascending=False)
    
    top_20 = df_imp.head(20)['feature'].tolist()
    print("\n--- Top 20 Features (Gain) ---")
    print(df_imp.head(20))
    
    # Load Data
    print("\nLoading Feature Data...")
    # Use the cache file we established is good
    cache_file = "data/features/temp_merge_current.parquet"
    if not os.path.exists(cache_file):
        print("Cache file not found")
        return
        
    df = pd.read_parquet(cache_file)
    df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    df_24 = df[df['year'] == 2024]
    df_25 = df[df['year'] == 2025]
    
    print(f"\nData Counts: 2024={len(df_24)}, 2025={len(df_25)}")
    
    print("\n--- Top Feature Comparison (Mean / Median) ---")
    print(f"{'Feature':<30} | {'2024 Mean':<10} | {'2025 Mean':<10} | {'Diff %':<8} | {'2024 Med':<10} | {'2025 Med':<10}")
    print("-" * 100)
    
    for f in top_20:
        if f not in df_24.columns:
            print(f"{f:<30} | NOT FOUND")
            continue
            
        # Check type
        is_numeric = pd.api.types.is_numeric_dtype(df_24[f])
        if not is_numeric:
            try:
                # Try convert
                df_24[f] = pd.to_numeric(df_24[f])
                df_25[f] = pd.to_numeric(df_25[f])
                is_numeric = True
            except:
                pass
                
        if is_numeric:
            m24 = df_24[f].mean()
            m25 = df_25[f].mean()
            med24 = df_24[f].median()
            med25 = df_25[f].median()
            
            diff = 0
            if m24 != 0 and not np.isnan(m24):
                 diff = (m25 - m24) / abs(m24) * 100
                 
            print(f"{f:<30} | {m24:<10.4f} | {m25:<10.4f} | {diff:<8.1f} | {med24:<10.4f} | {med25:<10.4f}")
        else:
             # Categorical stats (Unique count / Top value)
             mq24 = df_24[f].mode()[0] if not df_24[f].empty else "Nan"
             mq25 = df_25[f].mode()[0] if not df_25[f].empty else "Nan"
             print(f"{f:<30} | [CAT] Mode:{str(mq24)[:5]} | Mode:{str(mq25)[:5]} | ---      | ---        | ---")

if __name__ == "__main__":
    main()
