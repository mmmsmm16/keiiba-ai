"""
TabNet CUDA test using ACTUAL v9 dataset from the pipeline.
This will test if the issue is in the data itself.
"""
import torch
import numpy as np
import pickle
from pytorch_tabnet.tab_model import TabNetRegressor
import gc

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Load ACTUAL v9 dataset
dataset_path = "experiments/v12_tabnet_revival/data/lgbm_datasets.pkl"
print(f"Loading dataset from: {dataset_path}")

with open(dataset_path, 'rb') as f:
    datasets = pickle.load(f)

X_train = datasets['train']['X']
y_train = datasets['train']['y']
X_valid = datasets['valid']['X']
y_valid = datasets['valid']['y']

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
print(f"dtypes: X={X_train.dtypes.unique().tolist()}, y type={type(y_train)}")

# Check for non-numeric columns
non_numeric = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
if non_numeric:
    print(f"Non-numeric columns found: {non_numeric}")
    for col in non_numeric:
        print(f"  Converting {col}...")
        combined = X_train[col].astype('category')
        X_train[col] = combined.cat.codes
        X_valid[col] = X_valid[col].astype('category').cat.codes

# Convert to numpy
X_train_np = X_train.values.astype(np.float32)
X_valid_np = X_valid.values.astype(np.float32)

# Convert y
if hasattr(y_train, 'values'):
    y_train = y_train.values
y_train_np = y_train.reshape(-1, 1).astype(np.float32)

if hasattr(y_valid, 'values'):
    y_valid = y_valid.values
y_valid_np = y_valid.reshape(-1, 1).astype(np.float32)

print(f"Converted: X_train={X_train_np.shape}, dtype={X_train_np.dtype}")
print(f"y_train: {y_train_np.shape}, dtype={y_train_np.dtype}")

# Check for any remaining issues
print(f"NaN count X_train: {np.isnan(X_train_np).sum()}")
print(f"Inf count X_train: {np.isinf(X_train_np).sum()}")

# Replace any NaN/inf
X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_valid_np = np.nan_to_num(X_valid_np, nan=0.0, posinf=0.0, neginf=0.0)

torch.cuda.empty_cache()
gc.collect()

print("Creating TabNetRegressor...")
model = TabNetRegressor(
    device_name='cuda',
    verbose=1,
    n_d=32,
    n_a=32,
    n_steps=5
)

print("Starting fit (3 epochs only)...")
try:
    model.fit(
        X_train=X_train_np,
        y_train=y_train_np,
        eval_set=[(X_valid_np, y_valid_np)],
        eval_name=['valid'],
        max_epochs=3,
        patience=2,
        batch_size=512
    )
    print("SUCCESS: TabNet training completed!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
