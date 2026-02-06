"""
Train and save calibrator for production use
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
CALIBRATOR_PATH = "models/experiments/exp_t2_refined_v3/calibrator.pkl"

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    targets['race_id'] = targets['race_id'].astype(str)
    df = df.merge(targets[['race_id', 'horse_number', 'rank']], on=['race_id', 'horse_number'], how='left')
    df['date'] = pd.to_datetime(df['date'])
    
    # Use 2023 as calibration set
    df_calib = df[df['date'].dt.year == 2023].copy()
    print(f"Calibration set (2023): {len(df_calib)} records")
    
    # Load model
    model = joblib.load(MODEL_PATH)
    feature_names = model.feature_name()
    
    # Prepare features
    X = df_calib[feature_names].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
    
    # Predict
    print("Predicting...")
    preds = model.predict(X.values.astype(np.float64))
    df_calib['pred'] = preds
    df_calib['pred_norm'] = df_calib.groupby('race_id')['pred'].transform(lambda x: x / x.sum())
    df_calib['is_win'] = (df_calib['rank'] == 1).astype(int)
    
    # Train calibrator
    print("Training calibrator...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(df_calib['pred_norm'].values, df_calib['is_win'].values)
    
    # Save
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"Saved calibrator to: {CALIBRATOR_PATH}")
    
    # Verify
    df_calib['pred_calib'] = calibrator.predict(df_calib['pred_norm'].values)
    
    print("\nCalibration verification:")
    df_calib['pred_bin'] = pd.cut(df_calib['pred_calib'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
    check = df_calib.groupby('pred_bin', observed=True).agg(
        count=('is_win', 'count'),
        actual=('is_win', 'mean'),
        pred=('pred_calib', 'mean')
    )
    for idx, row in check.iterrows():
        print(f"  {idx}: Pred={row['pred']*100:.1f}%, Actual={row['actual']*100:.1f}%")

if __name__ == "__main__":
    main()
