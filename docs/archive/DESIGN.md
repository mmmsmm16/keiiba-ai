# 競馬予測AIプロジェクト "Project Strongest" 要件定義書 (v1.0)

## 1. プロジェクト概要
* **プロジェクト名:** Project Strongest (Keiba AI)
* **目的:** JRA（中央競馬）を対象に、市場オッズの精度を上回る予測モデルを構築し、長期的かつ安定的なプラス収支（回収率100%超）を実現する。
* **アプローチ:**
    * **Data Driven:** 過去10年以上のレース結果・馬柱・オッズデータをRDBに蓄積。
    * **Machine Learning:** LightGBM（Ranking学習）とDeep Learning（時系列解析）のアンサンブル。

---

## 2. ディレクトリ構成 (Directory Structure)
再現性と拡張性を担保するため、以下の構成を厳守する。

keiba-ai/
├── README.md               # 本ドキュメント
├── .env                    # 環境変数（DBパスワード等。Git管理外）
├── .gitignore              # Git管理除外設定
├── docker-compose.yml      # コンテナ構成定義
│
├── db/                     # データベース関連ディレクトリ
│   ├── data/               # [自動生成] DBデータ実体（Dockerマウント用）
│   └── init/               # DB初期化用SQL
│       └── 00_init.sql     # ★初回起動時にテーブルを作成するDDL
│
├── docker/                 # Docker環境構築用ファイル
│   └── python/
│       ├── Dockerfile      # Python環境定義
│       └── requirements.txt # Pythonライブラリ一覧
│
├── notebooks/              # 実験・分析用 (Jupyter Notebook)
│   ├── 01_data_exploration.ipynb
│   └── ...
│
├── src/                    # 本番用ソースコード
│   ├── __init__.py
│   ├── scraping/           # データ収集モジュール
│   │   ├── netkeiba.py     # netkeibaスクレイパー
│   │   └── loader.py       # DB格納処理
│   ├── preprocessing/      # 前処理モジュール
│   │   └── transformer.py  # 特徴量生成クラス
│   └── model/              # モデル定義・学習モジュール
│       ├── lgbm.py
│       └── trainer.py
│
└── data/                   # [一時利用] CSVファイル置き場（raw data等）

---

## 3. データベース設計 (Database Schema)
まずは「最小構成（MVP）」として以下の4テーブルを定義する。
**DBエンジン:** PostgreSQL 15+

### 3.1 `races` テーブル（レース情報）
レースの開催条件を管理するマスタ。
* **race_id (VARCHAR, PK):** レースID (例: '202305010111')
* **date (DATE):** 開催日
* **venue (VARCHAR):** 開催場所 (東京, 中山...)
* **race_number (INTEGER):** 第R
* **distance (INTEGER):** 距離 (m)
* **surface (VARCHAR):** 馬場 (芝, ダート)
* **weather (VARCHAR):** 天候
* **state (VARCHAR):** 馬場状態 (良, 重, 不良)
* **title (VARCHAR):** レース名

### 3.2 `results` テーブル（レース結果・出走馬情報）
学習データの核心となるトランザクションテーブル。
* **race_id (VARCHAR, FK):** レースID
* **horse_id (VARCHAR, FK):** 馬ID
* **jockey_id (VARCHAR):** 騎手ID
* **trainer_id (VARCHAR):** 調教師ID
* **frame_number (INTEGER):** 枠番
* **horse_number (INTEGER):** 馬番
* **rank (INTEGER):** 着順 (目的変数)
* **time (FLOAT):** 走破タイム (秒換算推奨)
* **passing_rank (VARCHAR):** 通過順 (例: '3-3-4')
* **last_3f (FLOAT):** 上がり3Fタイム
* **odds (FLOAT):** 確定単勝オッズ
* **popularity (INTEGER):** 人気順
* **weight (INTEGER):** 馬体重
* **weight_diff (INTEGER):** 体重増減
* *(PKは race_id + horse_id の複合キー)*

### 3.3 `horses` テーブル（馬基本情報）
不変な馬の属性情報。
* **horse_id (VARCHAR, PK):** 馬ID
* **name (VARCHAR):** 馬名
* **sex (VARCHAR):** 性別
* **birthday (DATE):** 生年月日
* **sire_id (VARCHAR):** 父ID
* **mare_id (VARCHAR):** 母ID

### 3.4 `payouts` テーブル（払い戻し情報）
シミュレーション時の収支計算用。
* **race_id (VARCHAR, FK):** レースID
* **ticket_type (VARCHAR):** 券種 (単勝, 複勝, ...)
* **winning_numbers (VARCHAR):** 当たり馬番 (例: '7', '7-10')
* **payout (INTEGER):** 払戻金

---

## 4. 開発ロードマップ (WBS)

### 【Phase 0】 環境構築 & 設計 (Current Status)
* [ ] フォルダ構成の作成
* [ ] `docker-compose.yml` の作成
* [ ] `Dockerfile` / `requirements.txt` の作成
* [ ] `00_init.sql` (DB作成クエリ) の作成
* [ ] 開発環境 (`Jupyter Lab` + `PostgreSQL`) の起動確認

### 【Phase 1】 データ収集基盤 (Data Pipeline)
* [ ] スクレイピングモジュールの実装 (HTML取得)
* [ ] パーサーの実装 (HTML -> DataFrame変換)
* [ ] DBローダーの実装 (DataFrame -> PostgreSQL)
* [ ] 全レース情報の取得ジョブ実行 (過去10年分)

### 【Phase 2】 前処理 & 特徴量エンジニアリング
* [ ] 生データのクレンジング
* [ ] **ラグ特徴量生成:** 過去走の着順平均、タイム差、上がり3F等を計算して結合
* [ ] **カテゴリ変数処理:** LabelEncoding / Embedding

### 【Phase 3】 モデリング (Modeling MVP)
* [ ] **モデル:** LightGBM (Objective: `lambdarank`)
* [ ] **学習:** 期間によるTrain/Valid/Test分割 (例: 2015-2020学習, 2021検証, 2022-2023テスト)
* [ ] **評価:** Accuracy, Recall, NDCG, 回収率シミュレーション

### 【Phase 4】 高度化 (Advanced)
* [ ] Transformer / LSTM による時系列モデルの実装
* [ ] アンサンブル (Stacking) の実装
* [ ] オッズの歪み検知ロジック

---

## 5. 技術スタック & ツール
* **Language:** Python 3.10+
* **Database:** PostgreSQL 15
* **Libraries:**
    * *Data:* Pandas, Polars, SQLAlchemy, Psycopg2
    * *ML:* LightGBM, PyTorch, Scikit-learn
    * *Scraping:* Requests, BeautifulSoup4, Selenium (必要な場合のみ)
    * *Dev:* Jupyter Lab, Black, Flake8

---

## 6. 注意事項 (Guidelines)
1.  **リーク厳禁:** 未来のデータ（確定オッズや馬体重など、馬券購入時点で知り得ない情報）を特徴量に入れてはならない。
2.  **アクセス負荷:** スクレイピング時は必ず `time.sleep(1)` 以上を挟み、相手サーバーに迷惑をかけないこと。
3.  **バージョン管理:** コードはGitで管理し、実験ごとのパラメータや結果を記録すること（MLflow等の導入も視野）。