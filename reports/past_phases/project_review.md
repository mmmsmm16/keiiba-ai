# Project Strongest コードレビュー報告書

**日付:** 2025年5月 (推定)
**レビュワー:** Senior Data Scientist (AI Agent)

## 1. 概要 (Executive Summary)

プロジェクトはPhase 16-17まで進行しており、アーキテクチャはモジュール化され、Docker活用による再現性の担保も意図されています。しかし、**モデルの信頼性を根底から損なう「データリーク（Leakage）」**が主要な特徴量エンジニアリング内に発見されました。また、NumPy 2.x 環境への移行に伴う潜在的な非互換性リスクが存在します。

これらは緊急の修正が必要です。

## 2. 重要課題 (Critical Issues)

### 🚨 2.1. `CategoryAggregator` におけるデータリーク (High Priority)

**場所:** `src/preprocessing/category_aggregators.py`
**現象:**
カテゴリ変数の集計において、`sort_values(['date', 'race_id'])` の後に `groupby(col).shift(1).expanding()` を適用しています。
同一レースに同じ「調教師(trainer_id)」や「種牡馬(sire_id)」の管理馬が複数頭出走する場合、データフレーム上でそれらは隣接します。
`shift(1)` は直前の行を参照するため、**同レース内の先に処理された馬の結果（勝ち負け）が、後に処理される同条件の馬の特徴量に含まれてしまいます。**

**影響:**
学習時にモデルが「同レースの別馬の結果」という未来情報を参照してしまい、検証スコアが不当に高くなります（過学習）。本番予測時やリークのないテストデータでは性能が大幅に低下します。

**修正案:**
集計前に `race_id` 単位で集約を行うか、同一開催日(date)の統計を除外するロジックに変更する必要があります。

### ⚠️ 2.2. NumPy 2.x 互換性と Pickle のリスク

**場所:** `docker/python/Dockerfile`, `src/model/train.py`
**現象:**
Dockerfileのベースイメージ (`nvcr.io/nvidia/pytorch:24.06-py3`) は NumPy 2.x を含んでいる可能性が高いですが、`requirements.txt` でバージョンの固定が行われていません。
Pythonの `pickle` は NumPy のバージョン間で互換性がない場合があり（特に 1.x で作成したデータを 2.x でロードする場合など）、`data/processed/lgbm_datasets.pkl` の読み込みでエラーが発生する、あるいは予期せぬ挙動を示すリスクがあります。

**修正案:**
1. `requirements.txt` で `numpy<2.0.0` を指定して 1.x 系に固定する（推奨）。
2. または、NumPy 2.x に完全移行する場合は、既存の `.pkl` ファイルを全て破棄し、データセット生成 (`run_preprocessing.py`) を再実行することを `USER_MANUAL` に明記する。

### ❌ 2.3. テストコードの欠落

**場所:** `src/preprocessing/test_category_aggregators.py`
**現象:**
メモリ（記憶）上は存在するとされていたテストファイルが見当たりません。特にリークが発生しやすい集計ロジックのテストがないことは、品質管理上重大なリスクです。

## 3. その他の発見事項 (Other Findings)

*   **EnsembleModelのバージョン管理:** `train.py` 内のコメントにもある通り、`EnsembleModel` のロード処理がバージョン指定 (`v5` など) に完全に対応しきれていない可能性があります。
*   **Docker構成:** `torch==2.7.0` という記述が見られますが、これが正しいNightlyビルドを指しているか確認が必要です（通常、安定版の未来バージョンを指定するとインストールエラーになる可能性があります）。
*   **スクリプトの散在:** `src/scripts/` に `adhoc` や `debug_` などの一時ファイルが多く残っています。これらは混乱を招くため、削除または `archive/` への移動を推奨します。

## 4. 推奨アクションプラン (Recommendations)

1.  **リーク修正 (最優先):** `CategoryAggregator` のロジックを修正し、同一レース内の情報を参照しないようにする。
2.  **テスト作成:** `test_category_aggregators.py` を作成し、リークが起きていないか検証するテストケースを追加する。
3.  **環境固定:** `requirements.txt` に `numpy` のバージョンを明記し、データセットの再生成を行う。
4.  **リファクタリング:** 古いデバッグスクリプトの整理。

以上
