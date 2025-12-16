# Phase 3: Ablation Study & Feature Selection Report

**Date**: 2025-12-15
**Baseline Dataset**: `v11` (178 features)
**Protocol**: Screening (Fixed Valid 2024) -> Verification (Walk-Forward 2021-2023)

## 1. Executive Summary

Ablation実験の結果、**Human Stats (Jockey/Trainerの勝率等) を削除することで、モデル性能が劇的に向上する**ことが確認された。
一方、Embedding特徴量はWalk-Forward評価においてLogLoss/AUCの改善に寄与しており、採用を継続する。

| Experiment | Mean LogLoss | Mean AUC | Mean ROI (Kelly) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (v11 All)** | 0.25200 | 0.74990 | 64.49% | Ref |
| **Drop 'Human'** | **0.24539** | **0.77039** | **69.18%** | **Adopt (Strong Positive)** |
| **Drop 'Embedding'** | 0.25304 | 0.74655 | 71.65%* | Reject (Accuracy Degraded) |

*Drop EmbeddingのROIが高いのは、予測精度低下により「荒れ」を偶発的に拾った（LogLoss悪化が証拠）可能性が高く、信頼できない。

## 2. Decision: Acceptance Gate

**Acceptance Gate Check (vs Baseline)**
1.  **LogLoss改善**: 0.25200 -> **0.24539** ($\Delta -0.0066$) -> **PASS**
2.  **安定性**: 3年間全てのFoldでLogLoss/AUCが改善 -> **PASS**
3.  **ROI**: 64% -> 69% (Kelly) -> **PASS**

**結論**: `human_stats` (52特徴量) を削除したセットを **v12 (Proposed)** として採用する。

## 3. Analysis

*   **Human Statsの毒性**:
    *   `jockey_id_win_rate` などが、強力すぎて過学習を起こしていた（Trainでは完璧だがValidで通用しない）、もしくはLeakageに近い挙動（Target Encodingの設計不備など）を含んでいた可能性がある。削除により汎化性能（AUC +0.02）が回復した。
*   **Embeddingsの有効性**:
    *   2024年Screeningでは削除が良化に見えたが、より長期のWalk-Forward (2021-2023) では削除によりLogLossが悪化した。これはEmbeddingsが長期的には重要な馬の個性を捉えていることを示唆するため、維持する。

## 4. Next Steps

*   **Phase 4 (Backtest & Robustness)**:
    *   この "Cleaned Model" (v12) を用いて、Slippage感度分析（オッズ低下時の耐性）および最終的な資金配分戦略のチューニングを行う。
