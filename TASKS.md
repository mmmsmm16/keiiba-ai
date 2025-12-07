# Project Strongest Tasks

## Phase 0: 環境構築 & 設計 (Environment Setup)
- [x] プロジェクトディレクトリ構造の作成
- [x] `.gitignore` の作成
- [x] Docker環境定義 (`requirements.txt`, `Dockerfile`)
- [x] `docker-compose.yml` の作成
- [x] データベース定義 (`00_init.sql`) の作成 (Note: JRA-VAN migration will overwrite this)
- [ ] 環境起動と動作確認 (Jupyter Lab, DB接続)

## Phase 1: データ収集基盤 (Data Collection)
- [x] スクレイピングモジュール基盤実装 (Depreciated)
- [x] HTMLパーサー実装 (Depreciated)
- [x] DBローダー実装 (Depreciated)
- [x] 過去データ取得ジョブの実行 (Depreciated)
- [x] **JRA-VANデータ導入 (New Strategy)**
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
- [x] 高度な特徴量生成 (Advanced Feature Engineering)
    - [x] 展開予測特徴量 (逃げ馬比率、メンバー構成によるペース予測)
    - [ ] 血統・コース適性の深化 (距離別・競馬場別の種牡馬成績)
- [x] アンサンブル学習 (Ensemble)
    - [x] CatBoost モデルの実装 (`src/model/catboost_model.py`)
    - [x] Stacking / Blending の実装 (`src/model/ensemble.py`)
- [ ] 自動運用パイプライン構築 (MLOps)
    - [ ] 定期実行スクリプトの整備

## Phase 5: リアルタイム予測 & 運用 (Real-time Pipeline)
- [x] 推論用データローダー実装 (`src/inference/loader.py`)
    - [x] JRA-VAN DBから開催予定データ(出馬表)を取得
- [x] 推論用前処理パイプライン実装 (`src/inference/preprocessor.py`)
    - [x] 過去データと結合しての特徴量生成 (MVP戦略)
- [x] 予測実行スクリプト実装 (`src/inference/predict.py`)
    - [x] モデルロード (LGBM, CatBoost, TabNet, Ensemble)
    - [x] スコア算出とランク付け
- [x] 予測閲覧簡易ビューア (`src/inference/viewer.py`)
    - [x] CLIまたはシンプルなWebUIでの結果表示
- [ ] 自動購入連携 (Optional)

## Phase 6: 分析・可視化 (Analysis & Visualization)
- [x] 実験結果出力の強化 (`src/model/evaluate.py`)
    - [x] 回収率シミュレーションの詳細データをファイル保存 (JSON/CSV)
- [x] ダッシュボードアプリ実装 (`src/dashboard/app.py`)
    - [x] **Streamlit** 導入
    - [x] 実験履歴 (`experiments/history.csv`) の一覧表示とフィルタリング
    - [x] 学習曲線の可視化 (もしログがあれば)
    - [x] 特徴量重要度の可視化
    - [x] 回収率シミュレーション結果 (ROIカーブ) の可視化
    - [x] リアルタイム予測実行タブの実装

## Phase 7: 推論高速化と高度なシミュレーション (Optimization & Advanced Simulation)
- [x] 推論高速化 (`InferencePreprocessor`)
    - [x] インクリメンタル前処理の実装 (過去データの差分更新)
    - [x] ダッシュボードへのキャッシュ導入
- [x] 複合馬券シミュレーション (`evaluate.py`)
    - [x] `jvd_hr` (払戻データ) のロード機能
    - [x] 馬連・3連複・3連単のBOX買いシミュレーション
    - [x] ダッシュボードへの反映
- [x] 精度向上 (Accuracy Improvement)
    - [x] Optunaによるハイパーパラメータチューニング
    - [x] 特徴量エンジニアリングの追加実装 -> See Phase 8

## Phase 8: 特徴量エンジニアリングの深化 (Deep Feature Engineering)
- [x] **カテゴリ×条件別成績の拡張** (`src/preprocessing/category_aggregators.py`)
    - [x] 騎手×コース、種牡馬×コース、調教師×コースの成績集計
    - [x] 種牡馬×距離区分、種牡馬×馬場状態の成績集計
- [x] **相互作用特徴量の追加**
    - [x] 騎手×調教師（ゴールデンコンビ）の成績集計
- [x] **トレンドと変化の検知** (`src/preprocessing/advanced_features.py`)
    - [x] 間隔 (Interval) と長期休養フラグ
    - [x] 馬体重増減 (Weight Change) と大幅増減フラグ
    - [x] 騎手の近走勢い (Rolling 100 races win rate)
- [x] **レースコンテキストの数値化**
    - [x] メンバー平均賞金 (Race Level)
    - [x] メンバー平均年齢 (Race Age)
    - [x] レース内平均逃げ率 (Pace Prediction)
