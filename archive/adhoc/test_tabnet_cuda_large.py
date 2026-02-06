"""
TabNet CUDA test with realistic data dimensions matching v9 dataset.
"""
import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import gc

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Dimensions matching v9 dataset
n_samples_train = 2_000_000  # ~2M rows like real training data
n_samples_valid = 100_000
n_features = 380  # ~380 features like v9

print(f"Creating test data: {n_samples_train} train, {n_samples_valid} valid, {n_features} features")

# Use float32 explicitly
X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
y_train = np.random.randn(n_samples_train, 1).astype(np.float32)

X_valid = np.random.randn(n_samples_valid, n_features).astype(np.float32)
y_valid = np.random.randn(n_samples_valid, 1).astype(np.float32)

print(f"Data created. X_train shape: {X_train.shape}, dtype: {X_train.dtype}")

# Clear GPU cache
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

print("Starting fit...")
try:
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
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
