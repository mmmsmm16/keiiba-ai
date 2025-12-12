# Project Strongest コードレビュー報告書 (第2回)

**日付:** 2025年5月
**レビュワー:** Senior Data Scientist (AI Agent)

## 1. 概要 (Executive Summary)

修正のご連絡をいただきましたが、リポジトリ内の主要ファイルを再確認したところ、**前回の指摘事項（データリーク、環境依存性、テスト欠落）がコードに反映されていない（修正されていない）状態**であると見受けられます。

おそらく修正ファイルの保存忘れ、コミット漏れ、あるいは修正箇所の認識齟齬が発生している可能性があります。
再度、以下の「修正が確認できなかった項目」をご確認ください。また、新たに発見した改善点も追記しました。

## 2. 未修正の重要課題 (Unresolved Critical Issues)

### 🚨 2.1. `CategoryAggregator` のデータリーク (Still Active)

*   **状態:** **未修正**
*   **確認ファイル:** `src/preprocessing/category_aggregators.py`
*   **現状:**
    `aggregate` メソッド内のロジックが依然として以下のままです。
    ```python
    df = df.sort_values(['date', 'race_id'])
    # ...
    history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count())
    ```
    これでは、同一レースに出走する同条件（同調教師、同種牡馬）の馬同士で、**先行する行の「当日の結果」が後続行の「過去成績」として混入（リーク）**します。
*   **必要なアクション:**
    `race_id` で集約（`groupby(['race_id', target_col])`）してから統計量をとるか、集計対象から「当日（当該race_id）」を明確に除外するロジックへの書き換えが必要です。

### ⚠️ 2.2. NumPy 2.x 互換性と依存関係 (Still Active)

*   **状態:** **未修正**
*   **確認ファイル:** `docker/python/requirements.txt`
*   **現状:**
    `numpy` のバージョン指定が追加されていません。DockerベースイメージがNumPy 2.x系である場合、既存の `pickle` ファイル（1.x系で作成）との互換性エラーが発生するリスクが高い状態が続いています。
*   **必要なアクション:**
    `requirements.txt` に `numpy<2.0.0` を追記するか、データセット再生成手順をマニュアルに明記してください。

### ❌ 2.3. テストの欠落 (Still Active)

*   **状態:** **未修正**
*   **確認ファイル:** `src/preprocessing/test_category_aggregators.py` (存在せず)
*   **現状:**
    テストファイルが作成されていません。

## 3. 新たな発見・改善点 (New Findings & Other Points)

「他の点」として、さらに詳細なコード確認を行いました。

### 📉 3.1. アンサンブル学習におけるリーク (Ensemble Leakage)

*   **場所:** `src/model/ensemble.py` (`train_meta_model`)
*   **問題:**
    Meta Model (LinearRegression) の学習に `valid_set` を使用しています。
    Base Model (LGBMなど) も学習時の Early Stopping 等で `valid_set` を参照しているため、Base Model は `valid_set` に対して過剰に適合（オーバーフィット）している可能性があります。
    その予測値を Meta Model の学習に使うと、Meta Model は「Base Model の過信」を学習してしまい、未知のテストデータでの性能が落ちる原因になります。
*   **推奨:**
    本来は **Out-of-Fold (OOF) 予測値**（学習データに対するクロスバリデーション予測値）を用いて Meta Model を学習するのが定石です。現状の実装（Validation Set利用）は簡易的ですが、精度を追求するなら修正推奨です。

### 💰 3.2. 馬券戦略のリスク管理 (Risk Management)

*   **場所:** `src/inference/optimal_strategy.py`
*   **問題:**
    3連単の買い目生成 (`_strategy_sanrentan_4`) などで、モデルのスコアや確率を信頼して高配当を狙うロジックになっています。
    現在のモデル（2.1のリークを含む状態）では、確率が不正確（自信過剰）に出力される恐れがあり、この状態で運用すると**多額の損失**を生むリスクがあります。
*   **推奨:**
    リーク修正が完了し、確率のキャリブレーション（補正）が正常に機能していることを確認するまで、実際の資金投入は控えるべきです。

### 📄 3.3. ドキュメントの不整合

*   `USER_MANUAL.md` では `preprocessed_data.parquet` が出力とされていますが、`train.py` は `lgbm_datasets.pkl` を読み込みます。データフローの記述を統一すべきです。

## 4. アクションプラン (Updated)

1.  **最優先:** `CategoryAggregator` のリーク修正をコードに反映し、コミットする。
2.  **環境:** `requirements.txt` に `numpy<2.0.0` を追加。
3.  **検証:** テストコードを作成し、修正が機能しているか確認。

以上、再確認のほどよろしくお願いいたします。
