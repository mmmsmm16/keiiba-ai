# プロジェクト 最強 - ユーザーマニュアル

このドキュメントでは、**JRA-VANデータ (PC-KEIBA Database)** を使用した「Project Strongest」の環境構築、データ準備、学習、評価の手順を説明します。

## 1. 概要

本プロジェクトは、JRA-VANの公式データを利用して競馬予測モデルを構築します。
データ取得には Windows用ソフト「PC-KEIBA Database」を使用し、Docker上のPostgreSQLデータベースにデータをインポートします。
その後、Python環境でデータの前処理、特徴量エンジニアリング、Deep Learning (TabNet) を含むアンサンブル学習を行います。

---

## 2. 環境構築

### 前提条件
*   Windows PC (JRA-VANデータ取得用)
*   Docker Desktop がインストールされていること
*   JRA-VAN Data Lab. の契約があること
*   PC-KEIBA Database がインストールされていること
*   **[New] NVIDIA GPU (RTXシリーズ等)** が搭載されている場合、Deep Learningの高速化が可能です。

### GPU環境のセットアップ (Phase 4)
Deep Learning (TabNet) を有効活用するため、Docker環境はGPUパススルーに対応しています。
1.  ホストマシンに **NVIDIA Container Toolkit** をインストールしてください。
2.  `docker-compose.yml` にてGPUリソースが予約されていることを確認してください。

### 基本手順

1.  **Dockerコンテナの起動**
    プロジェクトのルートディレクトリで以下のコマンドを実行します。
    ```bash
    docker-compose up -d --build
    ```
    これにより、Python環境 (CUDA対応) とPostgreSQLデータベース (`pckeiba`) が起動します。

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

データベースからデータを読み込み、クリーニング、高度な特徴量生成（血統など）、データ分割を行い、Parquetファイルとして保存します。
Phase 4より、種牡馬・繁殖牝馬データの集計処理が追加されています。

```bash
docker-compose exec app python src/preprocessing/run_preprocessing.py
```
*   **出力:** `data/processed/preprocessed_data.parquet`

### ステップ 2: モデル学習と実験 (Training & Experiment)

アンサンブルモデル (LightGBM + CatBoost + TabNet) を学習します。
Phase 4より、**実験管理機能**が追加されました。

**基本コマンド:**
```bash
docker-compose exec app python src/model/train.py
```

**実験名やメモを指定して実行:**
```bash
docker-compose exec app python src/model/train.py --experiment_name "exp001_tabnet_tuning" --note "TabNetのパラメータ調整"
```

*   **出力:** `models/ensemble_model.pkl` (学習済みモデル)
*   **実験ログ:** `experiments/history.csv` (実験履歴一覧), `experiments/<exp_name>_detail.json` (詳細パラメータと結果)

### ステップ 3: 評価 (Evaluation)

テストデータ (2024年) を使用して、モデルの精度と回収率シミュレーションを行います。

```bash
docker-compose exec app python src/model/evaluate.py
```
*   **出力:** コンソールにRMSE、MAP@10、および回収率シミュレーションの結果が表示されます。

---

## 4. プロジェクト構成

*   `src/preprocessing/`: データ前処理用コード
    *   `bloodline_features.py`: **[New]** 血統特徴量 (Sire/Mare) の生成
    *   `run_preprocessing.py`: 前処理パイプラインの実行スクリプト
*   `src/model/`: モデル定義と学習・評価
    *   `tabnet_model.py`: **[New]** Deep Learning (TabNet) モデル定義
    *   `ensemble.py`: LightGBM, CatBoost, TabNet のアンサンブル
    *   `train.py`: 学習実行スクリプト (実験ログ保存機能付き)
*   `src/utils/`: ユーティリティ
    *   `experiment_logger.py`: **[New]** 実験ログ管理
*   `docker-compose.yml`: Docker構成定義 (GPU対応)
*   `experiments/`: **[New]** 実験結果ログディレクトリ

## 5. トラブルシューティング

*   **GPUが認識されない:** `nvidia-smi` コマンドがコンテナ内で実行できるか確認してください。実行できない場合、Docker DesktopのGPU設定やNVIDIA Driverを確認してください。
*   **TabNetの学習が遅い:** GPUが有効でない場合、CPUで学習が行われます。データ量が多い場合は時間がかかります。
*   **DB接続エラー:** PC-KEIBAから接続できない場合は、ポートが `5433` であること、ファイアウォール設定などを確認してください。
