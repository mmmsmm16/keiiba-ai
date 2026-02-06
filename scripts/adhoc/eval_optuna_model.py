"""Compare Optuna HPO model vs baselines"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ndcg_score

def ndcg_at_k(y_true, y_pred, groups, k=3):
    scores = []
    idx = 0
    for g in groups:
        y_t = y_true[idx:idx+g]
        y_p = y_pred[idx:idx+g]
        if len(y_t) >= k and y_t.sum() > 0:
            s = ndcg_score([y_t], [y_p], k=k)
            scores.append(s)
        idx += g
    return np.mean(scores) if scores else 0.0

# Load data
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['is_win'] = (df['rank'] == 1).astype(int)
df['relevance'] = 0
df.loc[df['rank'] == 1, 'relevance'] = 3
df.loc[df['rank'] == 2, 'relevance'] = 2
df.loc[df['rank'] == 3, 'relevance'] = 1

test_df = df[df['year'] == 2024].copy()
groups = test_df.groupby('race_id').size().values

print("=== v12 Optuna HPO Model ===")
model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
X_test = test_df[features].values
y_pred = model.predict(X_test)

ndcg = ndcg_at_k(test_df['relevance'].values, y_pred, groups, k=3)

test_df['pred'] = y_pred
test_df['pred_rank'] = test_df.groupby('race_id')['pred'].rank(ascending=False)
top1 = test_df[test_df['pred_rank'] == 1]
total_bet = len(top1) * 100
returns = top1[top1['is_win'] == 1]['odds'].sum() * 100
roi = returns / total_bet * 100

print(f"NDCG@3: {ndcg:.4f}")
print(f"Win ROI: {roi:.2f}%")
print(f"Total Bets: {len(top1)}, Wins: {len(top1[top1['is_win'] == 1])}")
