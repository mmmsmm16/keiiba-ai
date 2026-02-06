import joblib
import os
import lightgbm as lgb
import sys

path = "models/experiments/exp_t2_refined_v3/model.pkl"
if not os.path.exists(path):
    print(f"Model not found at {path}")
    sys.exit(1)

model = joblib.load(path)
print(f"Type: {type(model)}")

if hasattr(model, 'booster_'):
    booster = model.booster_
    print("Using model.booster_")
else:
    booster = model
    print("Using model as booster")

print("Feature names len:", len(booster.feature_name()))
params = booster.params
print("Params keys:", params.keys())
cat_indices = params.get('categorical_column', [])
print(f"Cat indices count: {len(cat_indices)}")
print(f"Cat indices raw: {cat_indices}")

if cat_indices:
    names = booster.feature_name()
    try:
        cat_names = [names[i] for i in cat_indices]
        print("Cat names from params:", cat_names)
    except IndexError:
        print("IndexError mapping cat indices to names")
else:
    print("No cat indices in params.")
