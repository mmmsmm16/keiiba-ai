import os
import pickle
import pandas as pd
import lightgbm as lgb
from datetime import datetime

# Config
DATA_PATH = "data/nar/lgbm_datasets_south_kanto.pkl"
MODEL_DIR = "models/production/nar"
MODEL_NAME = "v1_south_kanto.txt"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print(f"Loading datasets from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        datasets = pickle.load(f)
    
    print(f"Dataset keys: {datasets.keys()}")
    
    # Correct extraction logic for nested dict structure
    train_set = datasets.get('train')
    valid_set = datasets.get('valid')
    test_set  = datasets.get('test')
    
    if train_set is None or 'X' not in train_set:
        print("Error: Invalid dataset structure.")
        return

    X_train = train_set['X']
    y_train = train_set['y']
    q_train = train_set['group']
    
    # Use valid set if available, else test set? Usually valid is for early stopping.
    if valid_set is not None and not valid_set['X'].empty:
        X_valid = valid_set['X']
        y_valid = valid_set['y']
        q_valid = valid_set['group']
        print("Using validation set for early stopping.")
    elif test_set is not None and not test_set['X'].empty:
        X_valid = test_set['X']
        y_valid = test_set['y']
        q_valid = test_set['group']
        print("Using test set for early stopping (since valid is empty).")
    else:
        X_valid = None
        y_valid = None
        q_valid = None
        print("No validation set available.")

    print(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape if X_valid is not None else 'None'}")
    
    # Create LGBM Dataset
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, group=q_valid, reference=lgb_train) if X_valid is not None else None
    # JRA v7 Best Params (Replicating exactly)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'learning_rate': 0.15761696141200815,
        'num_leaves': 76,
        'min_data_in_leaf': 53,
        'feature_fraction': 0.8020834992152445,
        'bagging_fraction': 0.5619080838587781,
        'bagging_freq': 7,
        'lambda_l1': 1.5705346164280545e-05,
        'lambda_l2': 0.050907081289160994,
        'random_state': 42,
        'verbose': -1
    }
    
    print("Starting training...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid] if lgb_valid else [lgb_train],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Save
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save_model(save_path)
    print(f"Model saved to {save_path}")
    
    # Feature Importance
    print("\nFeature Importance (Top 20):")
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    print(importance)

if __name__ == "__main__":
    train()
