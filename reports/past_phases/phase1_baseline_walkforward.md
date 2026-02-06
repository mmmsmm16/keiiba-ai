# Phase 1: Baseline Walk-Forward Evaluation Report

**Date**: 2025-12-15
**Model**: LightGBM (LambdaRank)
**Dataset**: `v11` (Feature Set: Extended v11)
**Protocol**: Walk-Forward Validation (Expanding Window, Method A)

## 1. Evaluation Summary

本レポートは、Walk-Forward法（2021-2023年の3期間）におけるベースラインモデルの性能を確立するものである。今後導入する改善策（Speed Index, 購入戦略等）は、このベースライン数値を上回ることをAcceptance Gateとする。

| Metric | Mean (3 Folds) | Std Dev |
| :--- | :--- | :--- |
| **LogLoss** | **0.25200** | 0.00205 |
| **Brier Score** | **0.06937** | 0.00062 |
| **AUC** | **0.74990** | 0.00620 |
| **ROI (Single Bet)** | **46.21%** | 2.30% |

※ ROIは「各レースの最高確率馬を1点買い（資金配分・オッズ考慮なし）」での理論値。100%未満かつ低い値であるのは、人気サイド（低オッズ）を買う傾向が強いため正常である。これをPhase 2以降の購入モデルで100%超えに引き上げる。

## 2. Fold-wise Results (Detail)

| Fold | Eval Year | Train Data (Model) | Calib Data | LogLoss | Brier | AUC | ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 | **2021** | 2014-2019 | 2020 | 0.24991 | 0.06881 | 0.75475 | 46.34% |
| Fold 2 | **2022** | 2014-2020 | 2021 | 0.25400 | 0.07003 | 0.74856 | 43.85% |
| Fold 3 | **2023** | 2014-2021 | 2022 | 0.25208 | 0.06926 | 0.74640 | 48.45% |

## 3. Analysis & Observations

1.  **安定性 (Stability)**:
    *   LogLossの標準偏差は **0.002** と非常に小さく、年次によるブレは少ない。
    *   2022年がやや悪化（0.2540）しているが、極端なドリフト（0.26台への悪化など）は見られない。

2.  **Calibration**:
    *   Isotonic Regression (Train末尾1年利用) が正常に機能し、Brier Scoreも安定している。

3.  **Baseline for Acceptance**:
    *   今後のPhase 3（アブレーション）では、**Mean LogLoss < 0.25200** を達成することを第一の関門とする。

## 4. Next Steps
*   **Phase 2**: この確率 ($P_{model}$) とオッズ ($P_{market}$) を入力とする「購入モデル」を構築し、ROIの改善（特に100%超え条件の発見）に着手する。
