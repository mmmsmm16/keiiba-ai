# 実験管理運用ルール (Experiment Management Guidelines)

これまで「どの特徴量で、どのモデルを使い、どのような結果が出たか」の追跡が不十分だった反省を踏まえ、以下のルールを定めて運用します。

## 1. 実験の定義 (Experiment Definition)
全ての学習・検証プロセスは「実験 (Experiment)」単位で管理します。
アドホックなスクリプト修正でパラメータを変えて実行することは禁止し、必ず**設定ファイル**を作成してから実行します。

### 1.1 設定ファイル (Config)
実験ごとに **`config/experiments/exp_{id}_{description}.yaml`** を作成します。

**必須項目:**
*   **experiment_name**: ユニークな実験ID (例: `exp_v01_baseline`)
*   **feature_version**: 使用する特徴量セットのバージョン (例: `v1_basic`)
*   **model_params**: LightGBM等のハイパーパラメータ
*   **dataset**:
    *   `train_start_date`, `train_end_date`
    *   `valid_year`
*   **evaluation**: 評価指標 (AUC, NDCG, ROIなど)

### 1.2 特徴量セット管理 (Feature Blocks)
「モノリスな特徴量生成」を避け、**「Feature Block（特徴量の塊）」** 単位で管理・キャッシュします。

*   **構造**: `data/features/{block_name}.parquet`
*   **Config指定**:
    ```yaml
    features:
      - base_attributes      # 基本属性 (年齢, 性別, 斤量...)
      - history_stats_v1     # 過去走集計
      - jockey_stats_v1      # 騎手成績
      # - weather_condition  # コメントアウトで除外可能
    ```
*   **メリット**:
    *   試行錯誤の高速化（変更がないブロックはキャッシュ利用）。
    *   「どのブロックが効いたか」のA/Bテストが容易。

## 2. 品質保証 (QA)
### 2.1 自動リーク検知 (Leakage Check)
学習実行前 (`run_experiment.py`) に、以下のチェックを自動実行し、異常があれば即停止します。

1.  **Target相関チェック**: 相関係数 > 0.9 の特徴量は「答え」とみなしエラー。
2.  **Null分布チェック**: 確定オッズなど「事前にはNullであるべき」データの混入をチェック。

### 2.2 ベンチマーク基準 (Benchmarks)
リーダーボードには必ず以下の基準値を掲載し、比較対象とします。

1.  **Odds Rank**: 単勝人気順をそのままスコアとした場合 (The Crowd Baseline)。
2.  **v19 (Old Model)**: リビルド前の旧モデルの性能。


## 2. 評価指標 (Metrics)
全ての実験で以下の指標を**必ず**計測し、記録します。

1.  **機械学習指標**:
    *   **AUC / LogLoss**: 基本的な分類精度。
    *   **NDCG@K**: ランキング精度（上位に来るべき馬を上位にできているか）。
2.  **投資指標 (Backtest)**:
    *   **Flat ROI**: 期待値 > 1.0 の馬を均等買いした場合の回収率。
    *   **Kelly ROI**: Kelly基準で資金管理した場合の回収率（オプション）。
    *   **Win Rate**: 期待値 > 1.0 の馬の勝率。

## 3. 結果の記録 (Logging)
実験結果は自動的に以下の場所に記録されるようにします。

### 3.1 Leaderboard (リーダーボード)
**`reports/experiment_leaderboard.md`** (またはCSV) に、全実験のサマリを一覧化します。

| Exp ID | Features | CV Strategy | AUC | NDCG@5 | ROI (Flat) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| v01_base | Basic | TimeSplit | 0.720 | 0.450 | 78.5% | Rawデータのみ |
| v02_hist | +History | TimeSplit | 0.755 | 0.480 | 92.0% | 過去走集計追加 |

### 3.2 Artifacts (成果物)
各実験の出力は `models/experiments/{experiment_name}/` フォルダに隔離して保存します。
*   `config.yaml` (実行時の設定コピー)
*   `model.pkl` (学習済みモデル)
*   `metrics.json` (評価結果)
*   `importance.png` (特徴量重要度)
*   `simulation_log.csv` (バックテストの全履歴)

## 4. ワークフロー (Workflow)

1.  **Hypothesis**: 仮説を立てる（例：「騎手の過去3走成績を入れると精度が上がるはず」）。
2.  **Implementation**: 特徴量作成コードを実装し、バージョン (`vXX`) を割り当てる。
3.  **Config**: `config/experiments/exp_vXX_name.yaml` を作成。
4.  **Run**: `python src/run_experiment.py --config config/experiments/exp_vXX_name.yaml` を実行。
5.  **Review**: `experiment_leaderboard.md` が更新されるので、過去の実験と比較する。
6.  **Decision**: 採用かボツかを判断。

---
**次のアクション**:
このルールに従い、まずは `v01_baseline` の実験設定を作成しましょう。
