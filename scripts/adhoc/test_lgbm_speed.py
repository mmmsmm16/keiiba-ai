"""Minimal LightGBM test to check training speed"""
import pandas as pd
import lightgbm as lgb
import time
import os

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def test_speed():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Use smaller subset for diagnostic (50k samples)
    df_train = df[(df['year'] >= 2019) & (df['year'] <= 2022)].sample(n=50000, random_state=42)
    df_val = df[df['year'] == 2023].sample(n=10000, random_state=42)
    
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 
                 'is_win', 'is_top2', 'is_top3', 'year', 'rank_str', 'passing_rank']
    # Blanket exclusion of suspicious names
    suspicious_names = ['rank', 'win', 'top', 'finish', 'result']
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in id_cols and not any(s in c.lower() for s in suspicious_names)]
    print(f"Features after blanket exclusion: {len(feature_cols)}")
    
    X_train = df_train[feature_cols].copy()
    y_train = (df_train['rank'] == 1).astype(int)
    X_val = df_val[feature_cols].copy()
    y_val = (df_val['rank'] == 1).astype(int)
    
    print(f"X_train shape: {X_train.shape}")
    
    # Data Preprocessing
    print("Preprocessing data...")
    proc_start = time.time()
    
    # Identify categorical/object columns
    for col in X_train.columns:
        if X_train[col].dtype.name == 'category' or X_train[col].dtype == 'object':
            # Convert to category and then codes
            X_train[col] = X_train[col].astype('category').cat.codes
            X_val[col] = X_val[col].astype('category').cat.codes
        else:
            # Numeric
            X_train[col] = X_train[col].fillna(-999)
            X_val[col] = X_val[col].fillna(-999)
            
    proc_end = time.time()
    print(f"Preprocessing completed in {proc_end - proc_start:.2f} seconds")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': 1,
        'seed': 42,
        'n_jobs': -1  # Use all cores
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    print("Starting training (50 rounds)...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=[lgb_val],
        callbacks=[lgb.log_evaluation(period=10)]
    )
    
    # Feature Importance
    print("\nTop 20 Important Features:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(20).iterrows():
        print(f"  {row['feature']:<30}: {row['importance']:.2f}")


if __name__ == "__main__":
    test_speed()
