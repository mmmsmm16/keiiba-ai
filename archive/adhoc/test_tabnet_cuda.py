"""
Minimal TabNet CUDA test to isolate CUBLAS_STATUS_NOT_INITIALIZED error.
"""
import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Simple test data
n_samples = 1000
n_features = 50

X_train = np.random.randn(n_samples, n_features).astype(np.float32)
y_train = np.random.randn(n_samples, 1).astype(np.float32)

X_valid = np.random.randn(100, n_features).astype(np.float32)
y_valid = np.random.randn(100, 1).astype(np.float32)

print("Creating TabNetRegressor...")
model = TabNetRegressor(
    device_name='cuda',
    verbose=1
)

print("Starting fit...")
try:
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        max_epochs=5,
        patience=3,
        batch_size=256
    )
    print("SUCCESS: TabNet training completed!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    
    # Try to get more info
    import traceback
    traceback.print_exc()
