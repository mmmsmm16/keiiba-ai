import pickle
import lightgbm as lgb
import os
import json

model_path = "models/experiments/v12_win_lgbm/model.pkl"
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Feature Names:", model.feature_name())
    
    # Dump model to get info about categorical features
    model_json = model.dump_model()
    # Typically, categorical features are listed in 'categorical_feature' if it was a Dataset
    # but for a trained Booster, we can check the 'feature_infos' or just 'categorical_feature' key
    print("Categorical Feature Indices:", model_json.get('categorical_feature', []))
    
    cat_indices = model_json.get('categorical_feature', [])
    cat_names = [model.feature_name()[i] for i in cat_indices]
    print("Categorical Feature Names:", cat_names)
    
else:
    print(f"Model not found at {model_path}")
