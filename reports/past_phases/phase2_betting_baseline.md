# Phase 2: Purchase Model & Betting Baseline Report

**Date**: 2025-12-15
**Model**: LightGBM (LambdaRank) + PurchaseModel
**Dataset**: `v11`
**Strategy MVP**: Single Win (単勝)

## 1. Executive Summary

Phase 2では、オッズ正規化と期待値計算を行う `PurchaseModel` を実装し、Phase 1のWalk-Forward評価に統合した。
Naïve戦略（確率最大馬の単勝買い）と比較して、Kelly基準およびEV（期待値）に基づく戦略は **約18% のROI改善** を示した。
目標の100%には未達だが、購入モデルの導入効果（資金配分の最適化効果）は明確に確認された。

| Strategy | Mean ROI (3 Folds) | Bets Design |
| :--- | :--- | :--- |
| **Naïve (Top 1)** | **46.21%** | 確率1位を固定額購入 |
| **Kelly (Frac=0.1)** | **64.49%** | 期待値とオッズに応じた資金配分 (上限5%) |
| **EV > 0 (Flat)** | **64.82%** | 期待値プラスの馬のみ固定額購入 |

## 2. Fold-wise Results (Detail)

| Fold | Eval Year | ROI (Naïve) | ROI (Kelly) | ROI (EV Flat) | Improvement (Kelly vs Naïve) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 | **2021** | 46.34% | 55.26% | 65.93% | +8.9% |
| Fold 2 | **2022** | 43.85% | 65.95% | 63.83% | +22.1% |
| Fold 3 | **2023** | 48.45% | 72.27% | 64.70% | +23.8% |

## 3. Observations

1.  **購入モデルの有効性**:
    *   単純に確率が高い馬を買うのではなく、期待値 (EV) がプラスである馬（過小評価されている馬）に投資することで、ROIが大幅に改善した。
    *   特にFold 3 (2023) では、Kelly戦略がNaïve戦略を20%以上上回るパフォーマンスを見せた。

2.  **ベースラインとしての位置付け**:
    *   **64.5%** という数値は、特徴量エンジニアリング（Phase 3）や、より高度な券種戦略（Phase 4）によって100%超えを目指すための「出発点」となる。
    *   Calibration（Phase 1）が正常に機能しているため、EV計算の信頼性が担保されており、購入戦略がワークしていると言える。

## 4. Next Steps

*   **Phase 3 (Ablation Study)**:
    *   この評価環境を用いて、特徴量の選別（Screening & Verification）を行い、予測モデル自体の精度（LogLoss）とROIの底上げを図る。
