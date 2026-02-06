import lightgbm
import numpy as np
X = np.random.rand(10, 2)
y = np.array([0]*10)
ds = lightgbm.Dataset(X, label=y)
print(f"LightGBM version: {lightgbm.__version__}")
try:
    params = {
        'device': 'cuda',
        'verbose': -1,
        'gpu_use_dp': False
    }
    lightgbm.train(params, ds, 1)
    print("CUDA SUCCESS")
except Exception as e:
    print(f"CUDA FAILED: {e}")
