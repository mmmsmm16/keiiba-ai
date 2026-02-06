"""Confirm AUC without leakage on full data"""
import pandas as pd
import lightgbm as lgb
import time

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def test_full_data():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    df_train = df[(df['year'] >= 2019) & (df['year'] <= 2022)]
    df_val = df[df['year'] == 2023]
    
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 
                 'is_win', 'is_top2', 'is_top3', 'year', 'rank_str', 'passing_rank']
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in id_cols]
    
    X_train = df_train[feature_cols].copy()
    y_train = (df_train['rank'] == 1).astype(int)
    X_val = df_val[feature_cols].copy()
    y_val = (df_val['rank'] == 1).astype(int)
    
    print(f"X_train shape: {X_train.shape}")
    
    for col in X_train.columns:
        if X_train[col].dtype.name == 'category' or X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category').cat.codes
            X_val[col] = X_val[col].astype('category').cat.codes
        else:
            X_train[col] = X_train[col].fillna(-999)
            X_val[col] = X_val[col].fillna(-999)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,  # Faster learning for test
        'num_leaves': 31,
        'verbosity': 1,
        'seed': 42,
        'n_jobs': -1
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    print("Starting training (30 rounds)...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=30,
        valid_sets=[lgb_val],
        callbacks=[lgb.log_evaluation(period=5)]
    )

if __name__ == "__main__":
    test_full_data()
