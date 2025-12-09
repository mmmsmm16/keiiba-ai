# プロジェクト 最強 - ユーザーマニュアル

このドキュメントでは、**JRA-VANデータ (PC-KEIBA Database)** を使用した「Project Strongest」の環境構築、データ準備、学習、評価、そして運用（予測）の手順を説明します。
実装順ではなく、実際の**運用フロー（データ準備 → モデル構築 → 予測）**に沿って構成しています。

---

## 1. 概要

本プロジェクトは、JRA-VANの公式データを利用して競馬予測モデルを構築・運用するシステムです。
Windows用ソフト「PC-KEIBA Database」で取得したデータをDocker上のPostgreSQLに蓄積し、Python環境で機械学習（LightGBM, CatBoost, TabNet）およびアンサンブル学習を行います。

---

## 2. 環境構築

### 前提条件
*   Windows PC (JRA-VANデータ取得用)
*   Docker Desktop がインストールされていること
*   JRA-VAN Data Lab. の契約があること
*   PC-KEIBA Database がインストールされていること
*   **[推奨] NVIDIA GPU (RTXシリーズ等)**: Deep Learning高速化のため

### セットアップ手順

1.  **Dockerコンテナの起動**
    ```bash
    docker-compose up -d --build
    ```
    これにより、Python環境 (CUDA対応) とPostgreSQLデータベース (`pckeiba`) が起動します。
    *   **GPU設定:** `docker-compose.yml` でGPUパススルーが設定されています。`nvidia-smi` で確認可能です。

2.  **データベース接続設定 (PC-KEIBA側)**
    PC-KEIBA Database の「環境設定」で以下を設定し、接続テストを行ってください。
    *   **接続先:** `localhost` / **ポート:** `5433`
    *   **DB名:** `pckeiba` / **ユーザー:** `postgres` / **パス:** `postgres`
    *   **文字コード:** `UTF-8`

3.  **データのインポート**
    PC-KEIBA Database でJRA-VANからデータを取得し、DBに登録します。
    *   必要なテーブル: `jvd_race_shosai`, `jvd_seiseki`, `jvd_uma_master`, `jvd_haraimodoshi`, `jvd_uma_race` など

---

## 3. ステップ 1: データの準備 (前処理)

学習や予測を行う前に、データベースの生データを機械学習用フォーマットに変換します。

**データ更新時（毎週）の実行:**
```bash
docker-compose exec app python src/preprocessing/run_preprocessing.py
```
*   **出力:** `data/processed/preprocessed_data.parquet`
*   **処理内容:** データのクリーニング、血統データ集計、特徴量エンジニアリング、Parquet保存。

---

## 4. ステップ 2: モデルの構築 (最適化・学習・評価)

高品質な予測モデルを作成するためのフェーズです。「パラメータ最適化」→「学習」→「評価」の順に進めます。

### 4.1 ハイパーパラメータ自動最適化 (Optuna)

 **[推奨]** 定期的、または大きなデータ更新があった場合に実行します。
Optunaを利用して、各モデル（LightGBM / CatBoost / TabNet）の最適な設定値を探索します。

```bash
# 例: LightGBMを20回試行で最適化し、バージョン'v2'としてパラメータ保存
docker-compose exec app python src/model/optimize.py --model lgbm --trials 20 --version v2
```
*   `--model`: `lgbm`, `catboost`, `tabnet` から選択
*   `--trials`: 試行回数
*   `--version`: **[New]** 保存するパラメータのバージョンダグ (例: `v2`)。
    *   結果は `models/params/<model>_v2_best_params.json` に保存されます。

### 4.2 モデルの学習 (Training)

最適化されたパラメータ（またはデフォルト）を使用して、全期間のデータでモデルを学習させます。

**基本コマンド:**
```bash
docker-compose exec app python src/model/train.py --model <model_type> --version <tag>
```

**実行順序と例 (バージョン v2 を作成する場合):**

1.  **LightGBM (決定木勾配ブースティング)**
    ```bash
    docker-compose exec app python src/model/train.py --model lgbm --version v2
    ```
2.  **CatBoost (カテゴリカル特徴量に強い)**
    ```bash
    docker-compose exec app python src/model/train.py --model catboost --version v2
    ```
3.  **TabNet (Deep Learning)**
    ```bash
    docker-compose exec app python src/model/train.py --model tabnet --version v2
    ```
4.  **アンサンブル (最終モデル)**
    上記3つのモデルを統合します。
    ```bash
    docker-compose exec app python src/model/train.py --model ensemble --version v2
    ```
    *   **出力:** `models/ensemble_v2.pkl`

### 4.3 モデルの評価 (Evaluation)

テストデータ (2024年) を使用して、モデルの精度 (RMSE, AUC) や回収率シミュレーションを行います。

```bash
docker-compose exec app python src/model/evaluate.py --model ensemble --version v2
```
*   **[New]** `--version`: 評価対象のモデルバージョンを指定します。
*   **出力:** 
    *   ターミナル: 予測精度スコア、単純な回収率シミュレーション結果
    *   ファイル: `experiments/latest_simulation.json` (ダッシュボード用)

---

## 5. ステップ 3: 予測と分析
## 6. Dashboard 2.0 (Modern UI) の使い方
Streamlitダッシュボードが大幅にアップデートされ、分析機能が強化されました。

### 起動方法
```bash
streamlit run src/dashboard/app.py
```

### 主な機能
1.  **Home (Predictions)**
    *   **Hybrid View:** 左側のリストからレースを選択し、右側のカードで詳細分析を行います。
    *   **Deep Analytics:**
        *   **Radar Chart:** スピード・スタミナ・騎手などの6軸評価。
        *   **Position Map:** 展開予想図（逃げ・先行などの位置取り）。
        *   **Head-to-Head:** 気になる2頭の直接比較。
    *   **Value Detector:** AIの評価に対してオッズがおいしい馬を「Value」として強調表示します。

2.  **Schedule**
    *   今後のレース開催予定や、過去の開催日をカレンダー形式で確認できます。
    *   日付を選択すると、その日のレース一覧へジャンプします。

3.  **Settings**
    *   サイドバーの「Dark Mode」トグルで、ライトモード/ダークモードを切り替えられます。

---
、実際に開催されるレースの予想や分析を行います。

### 5.1 CLIでの予測実行 (Inference)

週末のレース予想など、特定の日付の予測データをCSV出力する場合に使用します。

**準備:**
PC-KEIBA Database で最新の「出馬表 (Entry)」データを登録しておく必要があります。

**実行コマンド:**
```bash
# 2024年12月1日のレースを v2 モデルで予測
docker-compose exec app python src/inference/predict.py --date 20241201 --version v2
```
*   `--version`: 使用するモデルバージョン
*   `--model`: 使用するアルゴリズム (デフォルト: `ensemble`)

**結果確認:**
```bash
docker-compose exec app python src/inference/viewer.py data/predictions/20241201_ensemble_v2.csv
```

### 5.2 分析ダッシュボード (Web UI)

ブラウザで実験結果の確認や、インタラクティブな予測・シミュレーションが可能です。

**起動:**
```bash
docker-compose exec app streamlit run src/dashboard/app.py
```
**アクセス:** `http://localhost:8501`

**主な機能:**
1.  **実験履歴:** 過去の実験結果と比較。
2.  **特徴量重要度:** AIがどのデータを重視しているか確認。
3.  **シミュレーション:** 「期待値1.0以上を買ったらどうなるか？」などの回収率カーブや、BOX買いシミュレーションを確認。
4.  **予測実行 (Predict タブ):**
    *   日付・場所を選んでリアルタイム予測。
    *   **モデル選択:** 使用するモデルと**バージョン (v1, v2...)** をプルダウンで選択可能。
    *   AIの予測スコア、勝率、オッズに基づいた「期待値」を表示し、狙い目の馬（期待値>100%）をハイライトします。

---

## 6. プロジェクト構成

*   `src/preprocessing/`: データ前処理 (`run_preprocessing.py`)
*   `src/model/`: 学習・最適化・評価
    *   `optimize.py`: Optuna最適化
    *   `train.py`: モデル学習
    *   `evaluate.py`: モデル評価
    *   `lgbm.py`, `catboost_model.py`, `tabnet_model.py`: 各モデル定義
*   `src/inference/`: 推論
    *   `predict.py`: CLI予測
*   `src/dashboard/`: Webダッシュボード (`app.py`)
*   `data/`: データディレクトリ
*   `models/`: 学習済みモデル・パラメータ保存先
*   `experiments/`: 実験ログ・シミュレーション結果

---

## 7. トラブルシューティング

*   **GPUが使われない:** `docker-compose exec app nvidia-smi` でGPUが見えるか確認してください。TabNet以外のモデル (LightGBM/CatBoost) はデフォルトでCPU学習設定になっている場合があります（コード内で設定変更可能）。
*   **データがないと言われる:** PC-KEIBAで「データ登録」が正しく完了しているか、日付指定が合っているか確認してください。
*   **Versionが見つからない:** `models/` ディレクトリに `lgbm_v2.pkl` などのファイルが実際に存在するか確認してください。

*   **Versionが見つからない:** `models/` ディレクトリに `lgbm_v2.pkl` などのファイルが実際に存在するか確認してください。

---

## 8. APIサーバー (FastAPI) の運用管理

Next.jsフロントエンドが利用するバックエンドAPI (uvicorn) は、通常Dockerコンテナ (`app` サービス) 内で稼働しています。
正常に応答しない場合や、再起動が必要な場合は以下の手順で操作してください。

### ステータス確認
コンテナ内で `uvicorn` プロセスが起動しているか確認します。
```bash
docker exec keiiba-ai-app-1 ps aux | grep uvicorn
```
*   正常時: プロセス情報が表示されます。
*   異常時: 何も表示されない、または `grep` 自体しか表示されません。

### サーバー停止
プロセスを強制終了します。
```bash
docker exec keiiba-ai-app-1 pkill -f uvicorn
```

### サーバー起動 (再起動)
バックグラウンドでサーバーを起動します。
```bash
docker exec -d keiiba-ai-app-1 uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
*   `--reload`: コード変更を検知して自動リロードする開発モード用のオプションです。

### コンテナごとの再起動
上記で改善しない場合、コンテナごと再起動します（Jupyterなども再起動されます）。
```bash
docker restart keiiba-ai-app-1
# 自動起動設定がない場合、再起動後に上記「サーバー起動」コマンドが必要になることがあります。
```

---

## 9. アクセス情報 (URL)

各サービスへのアクセスURLは以下の通りです。

### 🆕 新ウェブアプリ (Next.js)
**URL:** [http://localhost:3000](http://localhost:3000)
*   モダンなUIで高速に動作する新しいダッシュボードです。
*   **シミュレーター機能:** [http://localhost:3000/simulation](http://localhost:3000/simulation)

### 📊 旧ダッシュボード (Streamlit)
**URL:** [http://localhost:8501](http://localhost:8501)
*   探索的データ分析や、詳細なログ確認に使用します。

### ⚙️ バックエンド API (FastAPI)
**URL:** [http://localhost:8000](http://localhost:8000)
*   **APIドキュメント (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   APIの仕様確認や、テスト実行がブラウザ上で行えます。

