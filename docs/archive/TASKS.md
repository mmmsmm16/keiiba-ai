# プロジェクトタスク一覧 (Project Tasks)

## Phase 0: 環境構築 & 設計 (Environment Setup)
- [x] プロジェクトディレクトリ構造の作成
- [x] `.gitignore` の作成
- [x] Docker環境定義 (`requirements.txt`, `Dockerfile`)
- [x] `docker-compose.yml` の作成
- [x] データベース定義 (`00_init.sql`) の作成 (Note: JRA-VAN migration will overwrite this)
- [ ] 環境起動と動作確認 (Jupyter Lab, DB接続)

## Phase 1: データ収集基盤 (Data Collection)
- [x] スクレイピングモジュール基盤実装 (非推奨)
- [x] HTMLパーサー実装 (非推奨)
- [x] DBローダー実装 (非推奨)
- [x] 過去データ取得ジョブの実行 (非推奨)
- [x] **JRA-VANデータ導入 (新戦略)**
    - [x] PC-KEIBA Database 導入サポート (マニュアル作成)
    - [x] DBスキーマの確認と対応 (JraVanDataLoader実装)

## Phase 2: 前処理 & 特徴量エンジニアリング (Preprocessing)
- [x] データローダー実装 (`JraVanDataLoader`)
- [x] データクレンジング (型変換, 異常値除去, 欠損値処理) - Added Age fix.
- [x] 基本特徴量生成 (日付, 天候, コース条件の数値化)
- [x] 馬の過去走特徴量生成 (ラグ特徴量)
- [x] 集計特徴量生成 (騎手・調教師・種牡馬の過去勝率など ※リーク厳禁)
- [x] カテゴリ変数処理 (Target Encoding / Embedding) - Implemented as CategoryAggregator.
- [x] 中間データ保存機構 (Parquet形式での保存・読み込み)
- [x] 学習用データセット作成 (時系列Split, Query ID生成)

## Phase 3: モデリング (Modeling MVP)
- [x] LightGBM (Ranking) モデル実装 (`src/model/lgbm.py`)
- [x] 学習実行 & 評価 (`src/model/train.py`)
- [x] 回収率シミュレーション実装 (`src/model/evaluate.py`)

## Phase 4: 高度化 (Advanced Modeling)
- [x] 高度な特徴量生成
    - [x] 展開予測特徴量 (逃げ馬比率、メンバー構成によるペース予測)
    - [x] 血統・コース適性の深化 (距離別・競馬場別の種牡馬成績)
    - [x] 騎手×コース / 調教師×コース 特徴量 (v3 実装済)
- [x] アンサンブル学習
    - [x] CatBoost モデルの実装 (`src/model/catboost_model.py`)
    - [x] Stacking / Blending の実装 (`src/model/ensemble.py`)
- [x] 自動運用パイプライン構築 (MLOps)
    - [x] 定期実行スクリプトの整備 (`src/scripts/run_auto_predict_loop.py`)
    - [x] 自動起動設定 (`start_auto_service.bat`)

## Phase 4.5: モデル完成と高度機能 (現在進行中)
- [x] `CatBoost v3` & `TabNet v3` の最適化と学習 (最適化完了/中断, 学習進行中)
- [/] `Ensemble v3` (メタモデル) の学習
- [x] 「騎手×コース」特徴量の設計と実装
- [x] 「展開/ペース」特徴量の設計と実装
- [x] **v4モデルのデータリーク修正 (v4.1)**
    - 未来情報(着順, 確定オッズ)の除去と再学習
    - LGBM v4.1 ROI 67% (v3比微増, リーク完全排除)

## Phase 5: リアルタイム予測 & 運用 (Real-time Pipeline)
- [x] 推論用データローダー実装 (`src/inference/loader.py`)
    - [x] JRA-VAN DBから開催予定データ(出馬表)を取得
- [x] 推論用前処理パイプライン実装 (`src/inference/preprocessor.py`)
    - [x] 過去データと結合しての特徴量生成 (MVP戦略)
    - [x] `drop_cols` にリーク特徴量を追加 (v4.1対応)
- [x] 予測実行スクリプト実装 (`src/inference/predict.py`)
    - [x] モデルロード (LGBM, CatBoost, TabNet, Ensemble)
    - [x] スコア算出とランク付け
- [x] 予測閲覧簡易ビューア (`src/inference/viewer.py`)
    - [x] CLIまたはシンプルなWebUIでの結果表示
- [x] 自動購入連携 (Optional)
- [x] **Bug Fix**: `auto_predict.py` の勝率計算ロジック修正 (Softmax -> Calibration)
- [x] **自動通知システム**:
    - [x] Discord通知実装
    - [x] ループ実行スクリプト (`src/scripts/run_auto_predict_loop.py`)
    - [x] Windowsスタートアップ自動化 (`start_auto_service.bat`)

## Phase 6: 分析・可視化 (Analysis & Visualization)
- [x] 実験結果出力の強化 (`src/model/evaluate.py`)
    - [x] 回収率シミュレーションの詳細データをファイル保存 (JSON/CSV)
- [x] **モデル性能リーダーボード作成** (`experiments/LEADERBOARD.md`)
- [ ] **馬券戦略の最適化** (Betting Strategy)
    - [ ] `evaluate.py` の修正 (Raw予測データの保存)
    - [ ] 最適化スクリプト作成 (`src/analysis/optimize_betting.py`)
    - [ ] 勝率×オッズ×期待値のグリッドサーチ
- [x] ダッシュボードアプリ実装 (`src/dashboard/app.py`)
    - [x] **Streamlit** 導入
    - [x] 実験履歴 (`experiments/history.csv`) の一覧表示とフィルタリング
    - [x] 学習曲線の可視化 (もしログがあれば)
    - [x] 特徴量重要度の可視化
    - [x] 回収率シミュレーション結果 (ROIカーブ) の可視化
    - [x] 分布の可視化 (ROIカーブ) の可視化
    - [x] リアルタイム予測実行タブの実装
    - [x] **v4.1モデル (リーク修正版) 対応**

## Phase 7: 推論高速化と高度なシミュレーション
- [x] 推論高速化 (`InferencePreprocessor`)
    - [x] インクリメンタル前処理の実装 (過去データの差分更新)
    - [x] ダッシュボードへのキャッシュ導入
- [x] 複合馬券シミュレーション (`evaluate.py`)
    - [x] `jvd_hr` (払戻データ) のロード機能
    - [x] 馬連・3連複・3連単のBOX買いシミュレーション
    - [x] ダッシュボードへの反映
- [x] 精度向上
    - [x] Optunaによるハイパーパラメータチューニング
    - [x] 特徴量エンジニアリングの追加実装 -> Phase 8 参照

## Phase 8: 特徴量エンジニアリングの深化
- [x] **カテゴリ×条件別成績の拡張** (`src/preprocessing/category_aggregators.py`)
    - [x] 騎手×コース、種牡馬×コース、調教師×コースの成績集計
    - [x] 種牡馬×距離区分、種牡馬×馬場状態の成績集計
- [x] **相互作用特徴量の追加**
    - [x] 騎手×調教師（ゴールデンコンビ）の成績集計
- [x] **トレンドと変化の検知** (`src/preprocessing/advanced_features.py`)
    - [x] 間隔 (Interval) と長期休養フラグ
    - [x] 馬体重増減 (Weight Change) と大幅増減フラグ
    - [x] 騎手の近走勢い (直近100走勝率)
- [x] **レースコンテキストの数値化**
    - [x] メンバー平均賞金 (Race Level)
    - [x] メンバー平均年齢 (Race Age)
    - [x] レース内平均逃げ率 (Pace Prediction)

## Phase 9: 馬券戦略の最適化 (Betting Strategy Optimization)
- [x] **複合オッズのリアルタイムロード** (`src/inference/loader.py`)
    - [x] `jvd_o3` (馬連), `jvd_o5` (3連複), `jvd_o6` (3連単) のロード実装 (Bug fixed)
- [x] **最適化シミュレーターの実装** (`src/model/betting_strategy.py`)
    - [x] 確率xオッズの閾値探索 (Grid Search: `optimize_betting.py`)
    - [x] 複合馬券シミュレーション機能の強化
- [x] **ダッシュボードへの反映** (`src/dashboard/app.py`)
    - [x] リアルタイム期待値に基づく買い目提案
    - [x] ダッシュボードUI修正 (Session State対応)

## Phase 10: 買い時判定AI (Betting Decision ML)
- [x] **買い目判断用データセット作成**
    - [x] `src/inference/preprocessor.py`: エントロピー、オッズ分散、Confidence Gap等の特徴量生成
- [x] **判定モデル学習** (`src/model/train_betting_model.py`)
    - [x] ターゲット: `is_profitable` (回収率>100%)
    - [x] モデル: LightGBM (Binary Classification)
- [x] **ダッシュボード統合**

## Phase 11: 買い目最適化 (Betting Optimization - Priority S) [x] Verified
- [x] **最適化ロジック実装** (`src/model/betting_strategy.py`)
    - [x] 全買い目候補の生成 (Candidate Generation)
    - [x] 期待値 (EV) 計算ロジックの実装
    - [x] Knapsack/Greedy アルゴリズムによる選定ロジック
- [x] **評価・検証** (`src/model/evaluate_betting_roi.py`)
    - [x] 新戦略でのバックテスト実行
    - [x] 固定フォーメーションとの性能比較
- [x] **ダッシュボード統合** (`src/dashboard/app.py`)


## Phase 12: Deep Learning Features (Deep History Encoder - Priority A) [x] Verified
- [x] **Embedding学習** (`src/model/train_embedding.py`)
    - [x] `horse_id`, `jockey_id`, `trainer_id` のEmbedding学習 (PyTorch)
    - [x] Embedding Mapの保存
- [x] **特徴量生成** (`src/preprocessing/embedding_features.py`)
    - [x] EmbeddingをPre-trained特徴量として統合
- [x] **モデル再学習**
    - [x] 特徴量追加後のデータでv4モデル学習

## Phase 13: 確率の較正 (Calibration - Priority A) [x] Verified
- [x] **Calibration実装** (`src/inference/preprocessor.py`)
    - [x] `Isotonic Regression` または `Platt Scaling` の適用
    - [x] 補正後確率 (`calibrated_prob`) の生成
- [x] **信頼性検証** (`src/model/evaluate.py`)
    - [x] Reliability Diagram (信頼性曲線) のプロット (train_calibration.pyで確認)
    - [x] Expected Calibration Error (ECE) の算出
    - [x] Betting Model再学習 (Two-Stage Strategy: ROI 150% Achieved)


## Phase 14: 資金配分 (Dynamic Staking - Priority B) [x] Verified
- [x] **Kelly基準の実装** (`src/model/betting_strategy.py`)
    - [x] `Fractional Kelly Criterion` ロジックの追加
    - [x] 破産リスクを考慮した資金管理ルールの策定 (Max DD 54% observed)
- [x] **長期シミュレーション**
    - [x] 資金推移 (Bankroll Growth) のバックテスト (Umaren Strategy: +52% Profit, ROI 152%)


## Phase 15: モデル高度化 (Advanced Modeling - Priority C)
- [ ] **オッズ加重損失関数** (`src/model/lgbm.py`, `catboost_model.py`)
    - [ ] Custom Loss Function の実装 (Odds-Weighted LogLoss)
- [ ] **再学習と評価**
    - [ ] 新損失関数でのモデル再学習
    - [ ] 荒れるレースでの回収率比較




## Phase 16: 運用パイプラインの自動化 (Systematization & Automation - Priority A) [x] Verified
- [x] **統合パイプラインスクリプト** (`src/scripts/run_weekly.py`)
    - [x] データ更新 (Data Ingestion)
    - [x] 前処理 & 特徴量更新 (Preprocessing)
    - [x] モデル再学習 (Retraining / Update Model)
    - [x] 推論実行 (Inference v4)
    - [x] 買い目フィルタリング (Betting Strategy)
    - [x] レポート出力 (HTML)
- [x] **出力フォーマットの整備**
    - [x] スマホで見やすいHTMLレポート作成 (`src/reporting/html_generator.py`)
- [ ] ログ監視
    - [ ] 実行ログの保存とエラー通知

## Phase 16.5: Daily Backtest & Verification (Verification Tooling) [x] Verified
- [x] **Backtest Tool Impl** (`src/scripts/run_daily_backtest.py`)
    - [x] Simulate past inference and betting (`predict.py`)
    - [x] Evaluation against actual results (`evaluate_bets`)
    - [x] Payout data parsing (`BettingOptimizer` integration)
    - [x] HTML Report Generation (`reports/backtest/`)
- [x] **Verification on Past Data**
    - [x] Tested on 2024-11-24 (Japan Cup Day)
    - [x] Confirmed pipeline consistency (Loader -> Features -> Predict -> Strategy -> P&L)

## Phase 17: リスク管理UI (Risk Management Dashboard) [x] Implemented
- [x] **ダッシュボード拡張**
    - [x] 資金シミュレーション結果の統合表示 (`src/dashboard/pages/4_Risk_Management.py`)
    - [x] 「推奨賭け金」計算ツールの実装 (Your Bankroll -> Bet Size, Kelly Criterion)

## Phase 18: リアルタイム・オッズ連携 (Real-time Integration)
- [ ] **直前オッズ取得**
    - [ ] リアルタイムオッズ取得スクリプト
- [ ] **直前GO/NO-GO判定**
    - [ ] 期待値再計算ロジック

## Phase 19: リファクタリング・整理 (Refactoring & Cleanup)
- [x] **ルートディレクトリの整理**
    - [x] 不要なスクリプトを `src/scripts/adhoc` 等へ移動
    - [x] ログファイルの整理
- [x] **`src/scripts` の整理**
    - [x] `adhoc` (一時的なチェック用), `examples` ディレクトリの作成
    - [x] スクリプトの分類・移動
- [x] **モジュールの整理**
    - [x] `src/preprocessing` 内の実行スクリプトの移動/削除
    - [x] 重複ファイルの削除 (`regenerate_datasets.py`)
- [x] **インポートパスの修正**
    - [x] 移動したスクリプトの `sys.path` 修正
