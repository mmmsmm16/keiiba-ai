import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Synthetic data
X = np.random.randn(2000, 32)
y = np.random.randn(2000, 1)

print("Initializing TabNet...")
clf = TabNetRegressor(
    n_d=32, n_a=32, n_steps=5,
    optimizer_params=dict(lr=2e-2),
    verbose=1,
    device_name='cuda'
)

print("Starting fit...")
clf.fit(
    X_train=X, y_train=y,
    eval_set=[(X, y)],
    eval_name=['train'],
    eval_metric=['rmse'],
    max_epochs=5,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)
print("Fit complete.")
