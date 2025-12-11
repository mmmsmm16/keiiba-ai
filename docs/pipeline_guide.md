# MLパイプライン 実行ガイド

本ドキュメントでは、`src/pipeline/` に実装された統合MLパイプラインの設定方法と実行手順について解説します。
このパイプラインは、データ準備、モデル学習、評価、そして戦略最適化までの全工程を単一のスクリプトで自動実行します。

## 概要

パイプラインは以下の4つのステップで構成されています：
1.  **Data Preparation**: 特徴量エンジニアリング済みのデータをロードし、学習用データセットを作成します。
2.  **Model Training**: 設定されたモデル（LightGBM, CatBoost, TabNet, Ensemble）を学習します。
3.  **Evaluation**: 検証データを用いて推論を行い、基本的な精度評価および回収率シミュレーションを行います。
4.  **Strategy Optimization**: グリッドサーチ等を用いて、最もROIが高くなる馬券戦略（閾値や買い目条件）を探索します。

## 実行方法

パイプラインは `src/pipeline/run_experiment.py` を通じて実行します。
全ての設定はYAML形式の設定ファイルで管理されます。

```bash
# Docker環境内で実行する場合
docker-compose exec app python src/pipeline/run_experiment.py --config config/experiments/base.yaml
```

## 設定ファイル (`config/experiments/`)

実験設定はYAMLファイルで記述します。`config/experiments/base.yaml` をコピーして新しい実験設定を作成することをお勧めします。

### 設定項目一覧

```yaml
experiment_name: "experiment_v1"  # 実験名 (出力ディレクトリ名に使用)
description: "ベースラインモデルの実験"

# 1. データ設定
data:
  train_years: [2020, 2021, 2022, 2023, 2024] # 学習対象年
  valid_year: 2025                            # 検証対象年 (この年のデータで評価・戦略探索を行います)
  features: "v5_default"                      # 使用する特徴量セット
  drop_features: ["horse_weight", "jockey_id"] # 実験から除外したい特徴量のリスト (Optional)
  use_cache: true                             # 既存のデータセットがあれば再利用するか
  jra_only: true                              # JRA開催のみに絞るか

# 2. モデル設定
model:
  type: "ensemble"  # モデルタイプ: "lgbm", "catboost", "tabnet", "ensemble"
  
  # 各モデルのハイパーパラメータ (省略時はデフォルト値が使用されます)
  lgbm_params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 1000
    
  catboost_params:
    iterations: 1000
    depth: 6
    
  tabnet_params:
    max_epochs: 50
    batch_size: 1024
    device_name: "cuda" # GPUを使用しない場合は "cpu"

# 3. 評価設定
evaluation:
  metric: "roi" # 評価指標
  strategies: ["umaren", "sanrentan"] # シミュレーション対象の券種

# 4. 戦略最適化設定
strategy:
  enabled: true       # 戦略最適化を実行するか
  min_roi: 100.0      # レポート抽出する最低ROI
  target_bet_types: ["tansho", "umaren", "sanrentan"] # 最適化対象の券種
```

## 出力ディレクトリ (`experiments/`)

実験を実行すると、`experiments/<experiment_name>/` 配下に以下のファイルが生成されます。

```
experiments/experiment_v1/
├── config_snapshot.yaml    # 実行時の設定ファイルのバックアップ
├── data/
│   └── lgbm_datasets.pkl   # 学習に使用したデータセット
├── models/
│   ├── lgbm.pkl            # 学習済みLightGBMモデル
│   ├── catboost.pkl        # 学習済みCatBoostモデル
│   ├── tabnet.zip          # 学習済みTabNetモデル
│   └── ensemble.pkl        # 学習済みアンサンブルモデル
├── reports/
│   ├── metrics.json            # 評価スコア (Accuracy, 基本ROI等)
│   ├── predictions.parquet     # 検証データの全推論結果 (Odds, Rank入り)
│   ├── optimization_report.json # 戦略最適化の結果レポート
│   └── lgbm_importance.png     # 特徴量重要度プロット(LGBMのみ)
└── logs/
    └── experiment.log      # 実行ログ
```

## 活用するためのヒント

1.  **モデルの比較**: `model.type` を変更した複数のconfigファイル (`exp_lgbm.yaml`, `exp_ensemble.yaml`) を用意し、それぞれ実行して `metrics.json` を比較します。
2.  **高速な反復**: コード修正後の動作確認には `test_fast.yaml` （学習回数を極端に減らした設定）を使用してください。
3.  **戦略の分析**: `reports/optimization_report.json` には、様々な条件下（例: 「単勝期待値1.2倍以上」「3連単1着流し相手5頭」など）でのROIが記録されています。これを元に、実際の運用ルールを決定します。

## 実験管理・可視化 (MLOps Dashboard)

実験結果をブラウザ上で比較・分析できるダッシュボード機能を提供しています。

### 起動方法
```bash
docker-compose exec app streamlit run src/dashboard/app.py
```
ブラウザでダッシュボードを開き、サイドバーメニューから **「🧪 実験管理 (MLOps)」** を選択してください。

### 機能
*   **実験一覧**: 過去の全実験のROI, Accuracy, 設定パラメータを一覧比較できます。
*   **比較グラフ**: 各モデルのパフォーマンス差を棒グラフで直感的に把握できます。
*   **詳細レポート**: Web UI上で直接 `config.yaml` や `metrics.json` の中身を確認できます。
