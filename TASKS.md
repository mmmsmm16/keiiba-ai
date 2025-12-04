# Project Strongest Tasks

## Phase 0: 環境構築 & 設計 (Environment Setup)
- [x] プロジェクトディレクトリ構造の作成
- [x] `.gitignore` の作成
- [x] Docker環境定義 (`requirements.txt`, `Dockerfile`)
- [x] `docker-compose.yml` の作成
- [x] データベース定義 (`00_init.sql`) の作成
- [ ] 環境起動と動作確認 (Jupyter Lab, DB接続)

## Phase 1: データ収集基盤 (Data Pipeline)
- [x] スクレイピングモジュール基盤実装 (netkeiba.com)
- [x] HTMLパーサー実装 (HTML -> DataFrame)
- [x] DBローダー実装 (DataFrame -> PostgreSQL)
- [x] 過去データ取得ジョブの実行 (過去10年分) - Script `src/scraping/bulk_loader.py` prepared.

## Phase 2: 前処理 & 特徴量エンジニアリング (Preprocessing)
- [ ] データローダー実装 (SQL -> DataFrame, テーブル結合)
- [ ] データクレンジング (型変換, 異常値除去, 欠損値処理)
- [ ] 基本特徴量生成 (日付, 天候, コース条件の数値化)
- [ ] 馬の過去走特徴量生成 (着順, タイム差, 上がり3Fの移動平均など)
- [ ] 集計特徴量生成 (騎手・調教師・種牡馬の過去勝率など ※リーク厳禁)
- [ ] カテゴリ変数処理 (Target Encoding / Embedding)
- [ ] 中間データ保存機構 (Parquet形式での保存・読み込み)
- [ ] 学習用データセット作成 (時系列Split, Query ID生成)

## Phase 3: モデリング (Modeling MVP)
- [ ] LightGBM (Ranking) モデル実装
- [ ] 学習実行 & 評価 (Accuracy, NDCG)
- [ ] 回収率シミュレーション実装

## Phase 4: 高度化 (Advanced)
- [ ] Deep Learning モデル検討
- [ ] アンサンブル実装
- [ ] 自動運用パイプライン構築
