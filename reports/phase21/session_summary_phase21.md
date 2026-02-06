# Phase 21 Session Summary: Amplifying Market-Neutral Edges

本セッションでは、Phase 20で確立した「Market-Neutral Model (Residual Model)」の更なる精度向上を目指し、動的な重み最適化と直接的な誤差学習の実装・検証を行いました。

## 1. セッション開始時点の状態 (Initial State)

- **現状**: 
  - `residual` (Base) モデル（OddsモデルとNo-Oddsモデルの単純平均）が、ROI 81.7% を記録し、他の単独モデルを上回っていた。
  - しかし、単純平均（0.5:0.5）は最適解ではない可能性が高かった。
- **目標**: 
  - ROI 100%超えを目指し、以下の2つの新手法を実装・検証する。
    1.  **`residual_opt`**: 直近の成績に基づいて、最適な重み $w$ を動的に変化させる。
    2.  **`residual_direct`**: 市場予測の「間違い（残差）」を直接学習し、補正する。
- **コードベース**:
  - `run_jra_pipeline_backtest.py` にはまだ上記の新機能が含まれていなかった。
  - 特徴量のリーク対策（正規表現）が厳格すぎて、一部の有効なオッズ系派生特徴量が意図せず除外されていた可能性があった。

## 2. 実施した主な変更・アクション (Actions Taken)

### A. コード実装と修正
1.  **新モードの実装 (`run_jra_pipeline_backtest.py`)**:
    - `optimize_residual_weight`: 過去3ヶ月のデータを用いて、Profitを最大化する重み $w$ を探索するロジックを追加。Sigmoid関数による確率化も導入。
    - `get_oof_predictions`: `residual_direct` のために、Market ModelのOut-Of-Fold予測（リークなし予測）を生成する機能を追加。
    - 学習ループの改修: `residual_opt` と `residual_direct` の分岐ロジックを統合。
2.  **リーク防止ゲートの緩和**:
    - `check_leakage` 関数を修正し、`odds_t_` や `log_odds` などの派生特徴量は許可しつつ、`final`（確定オッズ）のみを厳密に遮断するように変更。
3.  **分析スクリプトの作成**:
    - `analyze_phase21.py` を新規作成し、複数モードのROI比較と「Alpha（市場予測に対する上乗せ）」を可視化。

### B. 検証とデバッグ
1.  **バックテストの実行**:
    - コンテナ環境にて 2025年1月〜 のWalk-Forward検証を実施。
    - `residual_opt` において「常に $w=0.00$（No-Odds 100%）が選択される」現象に直面。
    - **対応**: LambdaRankの生スコアをSigmoid関数で確率次元に変換する修正を実施。
2.  **中間評価**:
    - 2025年1月〜4月の暫定結果を確認。

## 3. 実験結果と現在の状態 (Current State)

### 検証結果
新手法はいずれも、ベースラインである「単純平均」を超えることができませんでした。

| モデル | ROI (2025 Jan-Apr) | 特徴・課題 |
|:---|---:|:---|
| **odds_only** | 77.54% | 市場依存（基準）。 |
| **residual (Base)** | **81.70%** | **Best**. 単純平均0.5。最もロバストで安定している。 |
| **residual_opt** | 70.60% | $w=0$ に偏重し、アンサンブル効果（分散低減）が消失。過学習気味。 |
| **residual_direct** | 75.14% | 悪くはないが、単純平均のバランスの良さには及ばず。 |

### 最終判断
- **「策士策に溺れる」**: 複雑な最適化や直接学習を行うよりも、単純なアンサンブル（Base）の方が、未知のデータ（2025年）に対して頑健であった。
- **方針転換**: Phase 21 の目標であった「モデル構造による改善」は一旦完了とし、**`residual` (Base)** を正式採用する。
- **次のステップ**: モデル自体をいじるのではなく、**「賭け方（Betting Strategy）」** の最適化によってROI向上を目指す（EV閾値、オッズ帯フィルターなど）。

## 4. 成果物
- `scripts/adhoc/run_jra_pipeline_backtest.py`: 新機能実装済み（将来の再検証可能）。
- `scripts/adhoc/analyze_phase21.py`: 比較検証用スクリプト。
- `walkthrough_phase21.md`: 実験レポート。
