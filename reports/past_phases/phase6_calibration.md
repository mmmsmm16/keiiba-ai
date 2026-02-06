# Phase 6: Calibration Check Report

**Date**: 2025-12-15 23:40
**Period**: 2024-2024

## Calibration Metrics

| Source | ECE (Win) | LogLoss (Win) | Brier (Win) | Samples |
|--------|-----------|---------------|-------------|---------|
| **model** | 0.00565 | 0.25500 | 0.07015 | 73,711 |
| **market** | 0.00179 | 0.20360 | 0.05781 | 46,752 |

## Model vs Market Comparison

| Metric | Delta (Model - Market) | Model Better? |
|--------|------------------------|---------------|
| ECE | +0.00386 | ❌ |
| LogLoss | +0.05140 | ❌ |
| Brier | +0.01235 | |

> ⚠️ **Model does not beat market baseline**

## Interpretation

- **ECE (Expected Calibration Error)**: Lower is better. Measures how well predicted probabilities match actual frequencies.
- **LogLoss**: Lower is better. Measures probabilistic accuracy.
- **Brier Score**: Lower is better. Mean squared error of probability predictions.

