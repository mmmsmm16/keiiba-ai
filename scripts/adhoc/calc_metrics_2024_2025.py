
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import sys
import os

def main():
    print("Loading predictions...")
    path_24 = "data/temp_t2/T2_predictions_2024_2025.parquet"
    path_25 = "data/temp_t2/T2_predictions_2025_only.parquet"
    
    if not os.path.exists(path_24):
        print(f"Error: {path_24} not found.")
        return
    if not os.path.exists(path_25):
        print(f"Error: {path_25} not found.")
        return
        
    df_24_full = pd.read_parquet(path_24)
    df_24_full['date'] = pd.to_datetime(df_24_full['date'])
    df_24 = df_24_full[df_24_full['date'].dt.year == 2024].copy()
    
    if 'is_win' not in df_24.columns:
        if 'rank' in df_24.columns:
            df_24['is_win'] = (df_24['rank'] == 1).astype(int)
        else:
            print("Error: 2024 data has no rank or is_win")
            print(f"Columns: {df_24.columns.tolist()}")
            return
    
    df_25 = pd.read_parquet(path_25)
    
    # Ensure targets
    # 2024 usually has is_win.
    # 2025 might need it created from rank.
    if 'is_win' not in df_25.columns:
        if 'rank' in df_25.columns:
            df_25['is_win'] = (df_25['rank'] == 1).astype(int)
        else:
            print("Error: 2025 data has no rank or is_win")
            return

    # Calculate Metrics
    def calc_metrics(df, name):
        y_true = df['is_win']
        y_pred = df['pred_prob']
        
        auc = roc_auc_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred, labels=[0, 1])
        
        mean_prob = y_pred.mean()
        std_prob = y_pred.std()
        q95 = y_pred.quantile(0.95)
        q99 = y_pred.quantile(0.99)
        max_prob = y_pred.max()
        
        print(f"--- {name} Metrics ---")
        print(f"Rows: {len(df)}")
        print(f"AUC:      {auc:.4f}")
        print(f"LogLoss:  {ll:.4f}")
        print(f"MeanProb: {mean_prob:.4f}")
        print(f"StdProb:  {std_prob:.4f}")
        print(f"MaxProb:  {max_prob:.4f}")
        print(f"95% Prob: {q95:.4f}")
        print(f"99% Prob: {q99:.4f}")
        return auc, ll, mean_prob, q95

    metrics_24 = calc_metrics(df_24, "2024")
    metrics_25 = calc_metrics(df_25, "2025")
    
    # Compare
    print("\n--- Comparison (2025 vs 2024) ---")
    print(f"AUC Diff:      {metrics_25[0] - metrics_24[0]:.4f}")
    print(f"LogLoss Diff:  {metrics_25[1] - metrics_24[1]:.4f}")
    print(f"MeanProb Diff: {metrics_25[2] - metrics_24[2]:.4f}")
    print(f"95% Prob Diff: {metrics_25[3] - metrics_24[3]:.4f}")

if __name__ == "__main__":
    main()
