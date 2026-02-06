from catboost import CatBoostClassifier
import os

model_path = "models/experiments/v12_win_cat/model.cbm"
if os.path.exists(model_path):
    model = CatBoostClassifier()
    model.load_model(model_path)
    print("Model Feature Names:", model.feature_names_)
    print("Cat Feature Indices:", model.get_cat_feature_indices())
    cat_names = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
    print("Cat Feature Names:", cat_names)
else:
    print(f"Model not found at {model_path}")
