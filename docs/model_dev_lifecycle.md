# Model Improvement Lifecycle Guide

本ドキュメントは、モデルの改良から実運用への反映までの標準的なワークフローをまとめたものです。
今後の開発（モデル更新・新券種追加）はこの手順に沿って行うことで、効率的かつ安全に運用をアップデートできます。

---

## 全体フロー
1. **Model Dev**: 特徴量作成・学習・精度評価
2. **Simulation**: 回収率シミュレーション・戦略最適化
3. **Deployment**: 設定変更・本番反映

---

## Phase 1: Model Dev (学習 & 評価)

新しいアイデア（特徴量追加、アルゴリズム変更）をモデルに反映させるフェーズです。

### 1-1. 前処理 & 特徴量生成
データリークを防ぎつつ、最新のデータをParquet化します。
```bash
# 特徴量を追加・修正後、再生成
docker compose exec app python src/preprocessing/run_preprocessing.py
```
- 出力: `data/processed/preprocessed_data.parquet`

### 1-2. モデル学習
実験管理スクリプトを使用して学習を実行し、検証用メトリクス (AUC, LogLoss) を確認します。
```bash
# 例: 実験名 "v14_new_feature" で学習
docker compose exec app python src/run_experiment.py --experiment_name v14_new_feature
```
- **Checkpoint**: AUCが向上しているか？学習曲線は正常か？
- 出力: `models/v14_new_feature/*.txt` (LightGBMモデルファイル)

---

## Phase 2: Simulation (戦略策定)

「予測スコアが良いモデル」が「儲かるモデル」とは限りません。シミュレーションで投資戦略を最適化します。

### 2-1. 単勝・複勝・馬連の最適化
既存の `optimize_multiticket.py` や `optimize_umaren_frequency.py` を使い、EV閾値や購入点数を決定します。
```bash
# 頻度重視の馬連戦略を探す例
docker compose exec app python src/simulation/optimize_umaren_frequency.py --model_dir models/v14_new_feature
```

### 2-2. 複合券種 (Wide/3連系) の最適化
オッズ依存度の低い券種や、組み合わせ系のシミュレーションを行います。
```bash
# ワイド・3連複の組み合わせシミュレーション
docker compose exec app python src/simulation/optimize_combinations.py
```

### 2-3. ポートフォリオ設定
シミュレーション結果に基づき、**「採用する戦略」**と**「パラメータ (閾値・金額)」**を決定します。
- Win Core: `p1 >= X, EV >= Y`
- Umaren: `EV >= Z`
- Wide: `Condition A`

---

## Phase 3: Deployment (実運用への反映)

テスト済みのモデルと戦略を本番環境（常駐スケジューラ）に適用します。

### 3-1. 設定ファイルの更新
新しい設定ファイル (`config/runtime/phase_k_v1.yaml` など) を作成し、Phase 2 で決めた値を記述します。
```yaml
prediction:
  model_version: "v14_new_feature" # 新モデルディレクトリを指定
strategies:
  win_core:
    thresholds: {p1: 0.65, ev: 2.5} # 最適化した値
```

### 3-2. モデルラッパーの更新 (必要な場合)
特徴量の増減やライブラリの変更があった場合、`src/runtime/model_wrapper.py` を修正して新モデルに対応させます。
- 通常は `MODEL_DIR` のパス変更や、`run_experiment.py` と同じ特徴量リストの使用だけで済みます。

### 3-3. プロセスの再起動
常駐している `race_scheduler.py` を再起動して、新しい設定とモデルを読み込ませます。
```bash
# プロセス停止
docker compose exec app pkill -f race_scheduler.py

# 新Configで起動
docker compose exec -d app python -u src/runtime/race_scheduler.py --config config/runtime/phase_k_v1.yaml
```
※ `race_scheduler.py` が引数で config path を受け取るように改修推奨（現在はコード内で固定または環境変数）。

---

## 主要ファイル・ディレクトリ一覧

| カテゴリ | パス | 説明 |
| :--- | :--- | :--- |
| **学習** | `src/run_experiment.py` | 学習実行エントリーポイント |
| **前処理** | `src/preprocessing/` | 特徴量生成・ETLロジック |
| **シミュレーション** | `src/simulation/` | 回収率・戦略最適化スクリプト群 |
| **運用ロジック** | `src/runtime/strategy_engine.py` | 買い目決定・予算管理ロジック |
| **運用設定** | `config/runtime/` | 戦略パラメータ定義 (YAML) |
