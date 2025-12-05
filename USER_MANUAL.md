# プロジェクト 最強 - ユーザーマニュアル

このドキュメントでは、**JRA-VANデータ (PC-KEIBA Database)** を使用した「Project Strongest」の環境構築、データ準備、学習、評価の手順を説明します。

## 1. 概要

本プロジェクトは、JRA-VANの公式データを利用して競馬予測モデルを構築します。
データ取得には Windows用ソフト「PC-KEIBA Database」を使用し、Docker上のPostgreSQLデータベースにデータをインポートします。
その後、Python環境でデータの前処理、特徴量エンジニアリング、LightGBM/CatBoostによるアンサンブル学習を行います。

---

## 2. 環境構築

### 前提条件
*   Windows PC (JRA-VANデータ取得用)
*   Docker Desktop がインストールされていること
*   JRA-VAN Data Lab. の契約があること
*   PC-KEIBA Database がインストールされていること

### 手順

1.  **Dockerコンテナの起動**
    プロジェクトのルートディレクトリで以下のコマンドを実行します。
    ```bash
    docker-compose up -d --build
    ```
    これにより、Python環境とPostgreSQLデータベース (`pckeiba`) が起動します。

2.  **データベース接続設定 (PC-KEIBA側)**
    PC-KEIBA Database を起動し、環境設定から以下のようにデータベース接続を設定してください。

    *   **接続先:** `localhost`
    *   **ポート:** `5433` (Dockerの外部公開ポート)
    *   **データベース名:** `pckeiba`
    *   **ユーザー:** `postgres`
    *   **パスワード:** `password`
    *   **文字コード:** `UTF-8`

3.  **データのインポート**
    PC-KEIBA Database の機能を使用して、JRA-VANからデータを取得し、データベースに登録してください。
    *   過去データ（セットアップ）と、最新データの登録を行ってください。
    *   必要なテーブル: `jvd_race_shosai`, `jvd_seiseki`, `jvd_uma_master`, `jvd_haraimodoshi` など

---

## 3. データ処理と学習の実行

すべての操作はDockerコンテナ内、または `docker-compose exec` コマンド経由で行います。

### ステップ 1: 前処理 (Preprocessing)

データベースからデータを読み込み、クリーニング、特徴量生成、データ分割を行い、Parquetファイルとして保存します。

```bash
docker-compose exec app python src/preprocessing/run_preprocessing.py
```
*   **出力:** `data/processed/preprocessed_data.parquet`

### ステップ 2: モデル学習 (Training)

作成されたデータセットを使用して、アンサンブルモデル (LightGBM + CatBoost) を学習します。

```bash
docker-compose exec app python src/model/train.py
```
*   **出力:** `models/ensemble_model.pkl` (学習済みモデル)

### ステップ 3: 評価 (Evaluation)

テストデータ (2024年) を使用して、モデルの精度と回収率シミュレーションを行います。

```bash
docker-compose exec app python src/model/evaluate.py
```
*   **出力:** コンソールにRMSE、MAP@10、および回収率シミュレーションの結果が表示されます。

---

## 4. プロジェクト構成

*   `src/preprocessing/`: データ前処理用コード
    *   `loader.py`: JRA-VANデータ (`jvd_*` テーブル) の読み込み
    *   `cleanser.py`: データクリーニング
    *   `features.py`: 特徴量エンジニアリング
    *   `run_preprocessing.py`: 前処理パイプラインの実行スクリプト
*   `src/model/`: モデル定義と学習・評価
    *   `lgbm.py`: LightGBMモデル定義
    *   `catboost_model.py`: CatBoostモデル定義
    *   `ensemble.py`: アンサンブルモデル定義
    *   `train.py`: 学習実行スクリプト
    *   `evaluate.py`: 評価実行スクリプト
*   `docker-compose.yml`: Docker構成定義
*   `data/`: データ保存ディレクトリ (gitignore)
*   `models/`: 学習済みモデル保存ディレクトリ (gitignore)

## 5. トラブルシューティング

*   **DB接続エラー:** PC-KEIBAから接続できない場合は、ポートが `5433` であること、ファイアウォール設定などを確認してください。
*   **データ不足:** 学習時にエラーが出る場合、PC-KEIBA側で十分な期間（例: 2015年以降）のデータが取り込まれているか確認してください。
