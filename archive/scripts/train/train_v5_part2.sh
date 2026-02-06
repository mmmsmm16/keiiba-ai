#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8

echo "Starting Part 2: Ensemble (LGBM + CatBoost)..."

# echo "[3/4] Training TabNet v5..."
# python src/model/train.py --model tabnet --version v5 --dataset_suffix _jra_v5

echo "[4/4] Training Ensemble v5..."
python src/model/train.py --model ensemble --version v5 --dataset_suffix _jra_v5

echo "Training Completed. Running Evaluation..."
python src/scripts/adhoc/evaluate_v5_vs_v4.py

echo "All steps finished."
