#!/bin/bash
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8

echo "Running Model Comparison: v5 (JRA Only) vs v4 (Baseline)..."
python src/scripts/adhoc/evaluate_v5_vs_v4.py
