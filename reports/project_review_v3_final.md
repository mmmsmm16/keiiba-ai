# Project Strongest コードレビュー報告書 (最終確認版)

**日付:** 2025年5月
**レビュワー:** Senior Data Scientist (AI Agent)

## 1. 調査回答: 「参照したファイルは古い/未使用ではないか？」について

ご指摘を受け、プロジェクト全体のファイル構成と実行パスを徹底的に再調査いたしました。

**結論:**
私がレビューした `src/preprocessing/category_aggregators.py` は、**現在稼働しているパイプライン (`src/preprocessing/run_preprocessing.py`) から直接インポートされ、使用されている「現役のコード」**であることを確認いたしました。

**確認根拠:**
1.  **エントリーポイント:** `src/preprocessing/run_preprocessing.py` (マニュアルで指定された実行スクリプト)
2.  **インポート文:** 12行目 `from preprocessing.category_aggregators import CategoryAggregator`
3.  **実行箇所:** 61-63行目 `cat_aggregator = CategoryAggregator(); df = cat_aggregator.aggregate(df)`
4.  **ファイル検索:** `CategoryAggregator` クラスを定義しているファイルは `src/preprocessing/category_aggregators.py` 以外に存在しません。

したがって、このファイルに残っているロジックの不備は、確実に現在のモデル学習プロセスに影響を与えています。

## 2. 依然として残る重要課題 (Critical Issues confirmed)

前回の報告と重複しますが、**ファイルが修正されていないため**、以下の問題が解決していません。

### 🚨 2.1. `CategoryAggregator` のデータリーク (確実)

**なぜリークするのか:**
```python
# src/preprocessing/category_aggregators.py
df = df.sort_values(['date', 'race_id']) # 1. 日付とレースIDでソート
# ...
grouped = df.groupby(col) # 2. 調教師IDなどでグループ化
# 3. 前行を参照して集計
history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count())
```

**具体的な発生シナリオ:**
あるレース (Race X) に、同じ調教師 (Trainer T) の管理馬が2頭 (馬A, 馬B) 出走したとします。
1.  `sort_values` により、馬Aと馬Bはデータフレーム上で**隣接**します（例: 行100=馬A, 行101=馬B）。
2.  `shift(1)` を行うと、行101 (馬B) の計算時に 行100 (馬A) のデータが参照されます。
3.  学習データ作成時、`df` には「レース結果 (`rank`)」が含まれています。
4.  したがって、**馬Bの特徴量作成時に、同レースの馬Aの結果（勝利したかどうか）が使われてしまいます。**

**修正案 (再掲):**
集計時に `race_id` をグループ化キーに含めるか、同日のデータを集計から除外する処理が必要です。

### ⚠️ 2.2. 環境依存性とテスト欠落

*   `docker/python/requirements.txt` に `numpy` のバージョン指定がなく、Pickleファイル破損のリスクがあります。
*   `src/preprocessing/test_category_aggregators.py` は依然として存在しません。

## 3. 追加発見事項 (Additional Findings)

*   **Ensembleのリーク:** `src/model/ensemble.py` にて、メタモデルの学習に `valid_set` を使用しているため、スタッキングとしての性能が過大評価される設計になっています（Out-of-Fold予測の使用を推奨）。

## 4. 最終推奨 (Final Recommendations)

ユーザー様の環境で修正を行ったつもりでも、Gitへのコミット漏れや、Dockerボリュームの同期ズレなどで、**私の見ている環境（リポジトリ）に反映されていない**可能性が高いです。

以下の手順で確認・修正をお願いいたします。

1.  `src/preprocessing/category_aggregators.py` を開き、上記のリーク箇所が修正されているか目視確認する。
2.  修正されていなければ、修正コードを適用する。
3.  `docker/python/requirements.txt` に `numpy<2.0.0` を追記する。

現状のコードのまま学習を進めると、**本番で全く勝てないモデル（検証スコアだけが高いモデル）**が出来上がってしまいます。

以上
