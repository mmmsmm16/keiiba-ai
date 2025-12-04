# Project Strongest ユーザーマニュアル

このドキュメントでは、競馬予測AI開発環境のセットアップからデータ収集、前処理、モデリングまでの手順を解説します。

## 1. 概要
本プロジェクトは以下のフェーズで構成されています。
*   **Phase 0:** 開発環境構築 (Docker + Python + PostgreSQL)
*   **Phase 1:** データ収集 (スクレイピング)
*   **Phase 2:** 前処理 & 特徴量エンジニアリング
*   **Phase 3:** モデリング & シミュレーション

## 2. 環境構築 (Setup)
まず、Dockerコンテナを起動して開発環境を立ち上げます。

```bash
# プロジェクトルートで実行
docker-compose up -d
```

*   **Jupyter Lab:** `http://localhost:8888` (トークンなし)
*   **PostgreSQL:** `localhost:5432` (User: user, Pass: password, DB: keiba)

## 3. データ収集 (Data Collection)
`netkeiba.com` から過去のレースデータを取得します。

### 実行コマンド
コンテナ内で実行することをお勧めします。

```bash
# 1. コンテナに入る
docker-compose exec app bash

# 2. スクレイピング実行 (例: 2015年〜2024年)
python src/scraping/bulk_loader.py --year_start 2015 --year_end 2024
```

### Tips
*   **中断と再開:** 処理が中断した場合でも、同じコマンドを再実行すれば、**取得済みのレースは自動的にスキップ**されます。
*   **バックグラウンド実行:** 長時間かかるため、`nohup` や `docker-compose exec -d` を使うと便利です。
    ```bash
    # コンテナ外からバックグラウンド実行
    docker-compose exec -d app python src/scraping/bulk_loader.py --year_start 2015 --year_end 2024
    ```
*   **進捗確認:** ログファイル (`logs/scraping.log`) を `tail -f logs/scraping.log` で確認できます。

## 4. 前処理 & 特徴量生成 (Preprocessing)
収集したデータを学習用データセットに変換します。

### 実行コマンド
```bash
# コンテナ内で実行
python src/preprocessing/run_preprocessing.py
```

### 処理内容
このコマンド一発で以下の処理が全自動で行われます。
1.  **データロード:** DBからレース・馬・結果データを結合して読み込み。
2.  **クレンジング:** 欠損値補完、異常値除外。
3.  **特徴量生成:**
    *   基本特徴量 (日付、天候、コース条件など)
    *   **ラグ特徴量:** 前走の着順、近5走平均タイムなど (過去走集計)
    *   **集計特徴量:** 騎手・種牡馬ごとの通算勝率など (Expanding Window集計)
4.  **データセット分割:**
    *   Train (2015-2022), Valid (2023), Test (2024)
5.  **保存:**
    *   `data/processed/preprocessed_data.parquet` (全データ)
    *   `data/processed/lgbm_datasets.pkl` (学習用分割データ)

## 5. モデル学習 & シミュレーション (Modeling)
作成したデータセットを使ってモデルを学習し、シミュレーションを行います。

### モデル学習
```bash
# コンテナ内で実行
python src/model/train.py
```
*   **出力:**
    *   `models/lgbm_model.pkl`: 学習済みモデル
    *   `models/importance.png`: 特徴量重要度のプロット

### 評価 & シミュレーション
2024年のテストデータに対して予測を行い、単勝1点買い（スコア1位の馬を購入）の回収率を計算します。

```bash
# コンテナ内で実行
python src/model/evaluate.py
```
*   **出力例:**
    ```
    対象レース数: 3450
    的中率: 25.4%
    総投資額: 345000円
    総回収額: 280000円
    回収率: 81.2%
    ```
