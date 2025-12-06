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
    *   **パスワード:** `postgres`
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

メモリ競合を防ぎ、各モデルを個別に評価するため、学習プロセスは4段階に分かれています。以下の順序で実行してください。
Phase 4より、**実験管理機能**が追加されました。

**1. LightGBM の学習**
```bash
docker-compose exec app python src/model/train.py --model lgbm
```
*   **出力:** `models/lgbm.pkl`

**2. CatBoost の学習**
GPUリソースをTabNet用に確保するため、CPUモードで実行されます。
```bash
docker-compose exec app python src/model/train.py --model catboost
```
*   **出力:** `models/catboost.pkl`

**3. TabNet の学習 (Deep Learning)**
GPU (CUDA) を使用して学習します。
```bash
docker-compose exec app python src/model/train.py --model tabnet
```
*   **出力:** `models/tabnet.pkl`

**4. アンサンブル学習 (Meta-Model)**
上記3つのモデルをロードし、最終的な予測モデルを作成します。
```bash
docker-compose exec app python src/model/train.py --model ensemble
```
*   **出力:** `models/ensemble_model.pkl`
*   **実験ログ:** `experiments/history.csv`, `experiments/<exp_name>_detail.json`

**オプション (実験名指定など):**
各コマンドに `--experiment_name "名前"` や `--note "メモ"` を付与できます。
例:
```bash
docker-compose exec app python src/model/train.py --model tabnet --note "バッチサイズ調整"
```

### ステップ 3: 評価 (Evaluation)

テストデータ (2024年) を使用して、モデルの精度と回収率シミュレーションを行います。

```bash
docker-compose exec app python src/model/evaluate.py
```
*   **出力:** コンソールにRMSE、MAP@10、および回収率シミュレーションの結果が表示されます。

---

## 4. 推論の実行 (Inference)

Phase 5より、学習済みモデルを使用して実際に開催されるレースの予想を行うことができます。

### 準備

1.  **データの更新:**
    PC-KEIBA Database を使用して、最新の「出馬表データ (Entry)」を取得・登録してください。
    *   `JV-Link` -> `データ登録` -> `通常（セットアップ以外）` を実行
    *   必要なテーブル: `jvd_race_shosai`, `jvd_uma_race`

2.  **学習済みモデルの確認:**
    `models/` ディレクトリに学習済みモデル (`lgbm.pkl`, `catboost.pkl`, `tabnet.zip`, `ensemble_model.pkl`) が存在することを確認してください。

### 実行方法

指定した日付（YYYYMMDD）のレースを予想します。

```bash
docker-compose exec app python src/inference/predict.py --date 20241201
```

**オプション:**

*   **`--model`**: 予測に使用するモデルを指定します。（デフォルト: `ensemble`）
    *   `ensemble`: 3つのモデルの統合予測（推奨）
    *   `lgbm`: LightGBMのみ
    *   `catboost`: CatBoostのみ
    *   `tabnet`: TabNetのみ

    ```bash
    # 例: LightGBMのみで予測
    docker-compose exec app python src/inference/predict.py --date 20241201 --model lgbm
    ```

*   **`--race_ids`**: 特定のレースIDのみを予測する場合。
    ```bash
    docker-compose exec app python src/inference/predict.py --race_ids 202406010101
    ```

### 結果の確認

予測結果は `data/predictions/` ディレクトリに CSV ファイルとして保存されます。
簡易ビューアを使用して、ターミナルで結果を確認できます。

```bash
docker-compose exec app python src/inference/viewer.py data/predictions/20241201.csv
```

---

## 5. 分析・可視化 (Dashboard)

Phase 6より、実験結果や回収率シミュレーションを可視化するWebベースのダッシュボードが利用可能です。

### 利用方法

Streamlitを使用するため、ブラウザからアクセスします。

**1. ダッシュボードの起動**
```bash
docker-compose exec app streamlit run src/dashboard/app.py
```

**2. ブラウザでアクセス**
以下のURLを開いてください。
`http://localhost:8501`

**画面構成と機能:**

1.  **概要 (Overview)**
    *   **実験履歴 (Experiment History):** 過去に行った学習実験の一覧（タイムスタンプ、モデル名、RMSE、NDCGなど）を表示します。
    *   **精度メトリクスの推移:** 実験ごとの精度の変化を折れ線グラフで確認できます。ドロップダウンで「RMSE」「NDCG@k」「MAP@k」を切り替え可能です。

2.  **特徴量重要度 (Feature Importance)**
    *   TabNetやLightGBMなどのモデルが「どのデータを重視して予測したか」を表示します。
    *   モデルの挙動を理解し、不要な特徴量の削減や新しい特徴量のアイデア出しに活用してください。

3.  **回収率シミュレーション (ROI Simulation)**
    *   `src/model/evaluate.py` で計算されたシミュレーション結果を表示します。
    *   **戦略別サマリ:** 「最大期待値買い」「最大スコア買い」など、単純な戦略での回収率と的中率を表示します。
    *   **回収率カーブ (ROI Curve):** 「期待値がX以上の馬を全て買う」という閾値戦略における、**回収率**（青線）と**購入件数**（橙棒）の推移をグラフ化します。
    *   **活用法:** 青線が赤の点線（100%ライン）を超えている箇所を探し、その時の閾値を実際の運用ルールとして採用します。

---

## 6. プロジェクト構成

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

## 7. トラブルシューティング

*   **GPUが認識されない:** `nvidia-smi` コマンドがコンテナ内で実行できるか確認してください。実行できない場合、Docker DesktopのGPU設定やNVIDIA Driverを確認してください。
*   **TabNetの学習が遅い:** GPUが有効でない場合、CPUで学習が行われます。データ量が多い場合は時間がかかります。
*   **DB接続エラー:** PC-KEIBAから接続できない場合は、ポートが `5433` であること、ファイアウォール設定などを確認してください。
