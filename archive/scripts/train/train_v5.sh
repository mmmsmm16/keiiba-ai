#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8


echo "Starting Model v5 (JRA Specialist) Training Pipeline..."

echo "[1/4] Training LightGBM v5..."
python src/model/train.py --model lgbm --version v5 --dataset_suffix _jra_v5

echo "[2/4] Training CatBoost v5..."
python src/model/train.py --model catboost --version v5 --dataset_suffix _jra_v5

echo "[3/4] Training TabNet v5..."
# Note: batch_size might need tuning for valid set size, defaulting to None (auto)
python src/model/train.py --model tabnet --version v5 --dataset_suffix _jra_v5

echo "[4/4] Training Ensemble v5..."
python src/model/train.py --model ensemble --version v5 --dataset_suffix _jra_v5

echo "All models trained successfully!"
