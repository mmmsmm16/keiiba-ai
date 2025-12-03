# Project Strongest Tasks

## Phase 0: 環境構築 & 設計 (Environment Setup)
- [x] プロジェクトディレクトリ構造の作成
- [x] `.gitignore` の作成
- [x] Docker環境定義 (`requirements.txt`, `Dockerfile`)
- [x] `docker-compose.yml` の作成
- [x] データベース定義 (`00_init.sql`) の作成
- [ ] 環境起動と動作確認 (Jupyter Lab, DB接続)

## Phase 1: データ収集基盤 (Data Pipeline)
- [ ] スクレイピングモジュール基盤実装 (netkeiba.com)
- [ ] HTMLパーサー実装 (HTML -> DataFrame)
- [ ] DBローダー実装 (DataFrame -> PostgreSQL)
- [ ] 過去データ取得ジョブの実行 (過去10年分)

## Phase 2: 前処理 & 特徴量エンジニアリング
- [ ] データクレンジング処理
- [ ] ラグ特徴量生成 (過去走集計)
- [ ] カテゴリ変数処理 (Encoding)
- [ ] データセット作成 (Train/Valid/Test split)

## Phase 3: モデリング (Modeling MVP)
- [ ] LightGBM (Ranking) モデル実装
- [ ] 学習実行 & 評価 (Accuracy, NDCG)
- [ ] 回収率シミュレーション実装

## Phase 4: 高度化 (Advanced)
- [ ] Deep Learning モデル検討
- [ ] アンサンブル実装
- [ ] 自動運用パイプライン構築
