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
- [ ] **JRA-VANデータ導入 (New Strategy)**
    - [ ] PC-KEIBA Database 導入サポート (マニュアル作成)
    - [ ] DBスキーマの確認と対応

## Phase 2: 前処理 & 特徴量エンジニアリング (Preprocessing)
- [x] データローダー実装 (`RawDataLoader`) -> *Needs Update for JRA-VAN Schema*
- [x] データクレンジング (型変換, 異常値除去)
- [x] 基本特徴量生成 (日付, 天候, コース条件)
- [x] 馬の過去走特徴量生成 (ラグ特徴量)
- [x] 集計特徴量生成 (騎手・調教師・種牡馬)
- [x] カテゴリ変数処理
- [x] 中間データ保存機構
- [x] 学習用データセット作成

## Phase 3: モデリング (Modeling MVP)
- [x] LightGBM (Ranking) モデル実装
- [x] 学習実行 & 評価
- [x] 回収率シミュレーション実装

## Phase 4: 高度化 (Advanced Modeling)
- [x] 高度な特徴量生成 (展開予測)
- [x] アンサンブル学習 (CatBoost + Stacking)
- [ ] ハイパーパラメータチューニング
- [x] 高度な特徴量生成 (Advanced Feature Engineering)
    - [x] 展開予測特徴量 (逃げ馬比率、メンバー構成によるペース予測)
    - [ ] 血統・コース適性の深化 (距離別・競馬場別の種牡馬成績)
- [x] アンサンブル学習 (Ensemble)
    - [x] CatBoost モデルの実装 (`src/model/catboost_model.py`)
    - [x] Stacking / Blending の実装 (`src/model/ensemble.py`)
- [ ] 自動運用パイプライン構築 (MLOps)
    - [ ] 定期実行スクリプトの整備
- [x] データベース定義 (`00_init.sql`) の作成 (Note: JRA-VAN migration will overwrite this)
- [ ] 環境起動と動作確認 (Jupyter Lab, DB接続)

## Phase 5: リアルタイム予測 & 運用 (Real-time Pipeline)
- [ ] リアルタイムデータ取得構成 (PC-KEIBA連携)
- [ ] 推論用パイプライン構築 (Preprocessing -> Inference)
- [ ] 自動購入連携 (Optional)
