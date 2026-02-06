
import sys
import os
import joblib
import lightgbm as lgb
import pandas as pd

MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    model = joblib.load(MODEL_PATH)
    print(f"Type: {type(model)}")
    
    booster = None
    if hasattr(model, 'booster_'):
        booster = model.booster_
        print("Extracted booster_")
    elif isinstance(model, lgb.Booster):
        booster = model
        print("Is Booster")
        
    if booster:
        print(f"Feature names: {len(booster.feature_name())}")
        # print(booster.feature_name())
        
        # Categorical features
        # In Booster, categorical_feature is stored in dump_model()
        dump = booster.dump_model()
        if 'pandas_categorical' in dump:
             print("Pandas Categorical Info Found")
             # print(dump['pandas_categorical'])
             # It usually contains a list of lists (categories)
             pass
        else:
             print("No pandas_categorical info in dump")
             
        # Check if we can see which indices are categorical
        # Usually checking feature_infos or similar?
        pass
    else:
        print("Could not extract booster")

if __name__ == "__main__":
    main()
