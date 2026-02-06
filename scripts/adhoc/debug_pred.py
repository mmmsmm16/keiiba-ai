"""
Debug script to test model prediction with preprocessed data
Fixed: Convert all features to numeric to avoid categorical mismatch
"""
import pandas as pd
import numpy as np
import joblib

print("=" * 60)
print("Debug: Testing Model Prediction (Fixed)")
print("=" * 60)

# Load model
MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"Model type: {type(model)}")

# Check categorical features in model
model_dump = model.dump_model()
model_cat_features = model_dump.get('categorical_feature', [])
print(f"Model categorical features: {model_cat_features}")

# Load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
print(f"Total rows: {len(df)}")

# Use 2024 data for test
df_test = df[df['year'] == 2024].copy()
print(f"2024 rows: {len(df_test)}")

# Leakage columns
LEAKAGE = ['pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank', 'last_3f',
           'raw_time', 'time_diff', 'margin', 'time', 'popularity', 'odds',
           'relative_popularity_rank', 'slow_start_recovery', 'track_bias_disadvantage',
           'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate']
META = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 'is_win', 
        'is_top2', 'is_top3', 'year', 'rank_str']
IDS = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']

exclude = set(META + LEAKAGE + IDS)
feature_cols = [c for c in df.columns if c not in exclude]
print(f"Feature columns: {len(feature_cols)}")

# Prepare X
X = df_test[feature_cols].copy()

# Convert ALL columns to numeric to avoid categorical mismatch
for c in X.columns:
    if X[c].dtype.name == 'category':
        X[c] = X[c].cat.codes
    elif X[c].dtype == 'object':
        X[c] = pd.Categorical(X[c]).codes
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(-999)

print(f"X shape: {X.shape}")
print(f"X dtypes unique: {X.dtypes.unique()}")

# Get expected features from model
expected_features = model.feature_name()
print(f"Model expects {len(expected_features)} features")

# Check alignment
current_cols = set(X.columns)
expected_set = set(expected_features)
missing = [c for c in expected_features if c not in current_cols]
extra = [c for c in current_cols if c not in expected_set]
print(f"Missing: {len(missing)}, Extra: {len(extra)}")
if missing:
    print(f"  Missing cols (first 10): {missing[:10]}")

# Align X to model
for c in missing:
    X[c] = -999.0
extra_to_drop = [c for c in extra if c in X.columns]
if extra_to_drop:
    X = X.drop(columns=extra_to_drop)
X = X.reindex(columns=expected_features)

# Convert to numpy array to completely avoid categorical issues
X_np = X.values.astype(np.float64)
print(f"After alignment X_np shape: {X_np.shape}")

# Predict with numpy array
print("Predicting...")
raw_preds = model.predict(X_np)
preds = 1 / (1 + np.exp(-raw_preds))  # Sigmoid

print(f"Raw preds sample: {raw_preds[:5]}")
print(f"Probs sample: {preds[:5]}")
print(f"Pred range: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")

# Verify predictions are different
unique_preds = len(np.unique(preds.round(4)))
print(f"Unique predictions: {unique_preds}")

if unique_preds < 10:
    print("WARNING: Very few unique predictions - likely an issue!")
else:
    print("OK: Predictions are varied")

# Show some examples
df_test = df_test.head(20).copy()
df_test['pred_prob'] = preds[:20]
print("\nSample predictions:")
print(df_test[['race_id', 'horse_number', 'pred_prob']].head(20).to_string())
