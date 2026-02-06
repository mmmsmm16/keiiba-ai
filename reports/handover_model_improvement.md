
# AIモデル改善・引継ぎガイド (2026-01-20)

## 概要
本ドキュメントは、Keiiba-AIモデルの現状、本番運用戦略、および今後の改善計画についてまとめています。

## 1. モデルの現状 (Phase 3 完了時点)

2024年のデータを用いて、4つのモデルタイプの収益性（ROI）を評価しました。

| モデルタイプ | 実験名 | 指標 | キャリブレーション | ステータス | ROI ポテンシャル |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Win Model** (単勝) | `optuna_best_full` | LogLoss | `calibrator.pkl` | ⚠️ 限定的 | 115% @ EV>2.5 (購入数 極少) |
| **Top2 Model** (連対) | `...top2` | LogLoss | なし | ❌ 未検証 | `Rank <= 2` での検証が必要 |
| **Top3 Model** (複勝) | `exp_t2...top3` | LogLoss | `calibrator.pkl` | ❌ 失敗 | 複勝は控除率の壁により構造的に利益が出ない。 |
| **LambdaRank** (ランキング) | `exp_lambdarank` | NDCG | `calibrator_win.pkl` | ✅ **本番採用** | **107% @ EV>1.8** (十分な購入数) |

**重要な洞察:**
- **生スコアは確率ではない**: LightGBMやLambdaRankのスコアはそのままでは確率として扱えません。
- **キャリブレーション必須**: `IsotonicRegression` を通すことで、EV（期待値）の信頼性が劇的に向上しました。
- **単勝にはLambdaRank**: 逆説的ですが、単勝専用モデルよりも、順位学習（LambdaRank）を「単勝ターゲット」でキャリブレーションした方が、穴馬（Value Longshots）をうまく捉え高ROIを達成しました。

## 2. 本番運用構成

現在、**LambdaRank (Calibrated)** を使用した本番運用を行っています。

- **実行スクリプト**: `scripts/production_run_lambdarank.py`
    - **高速化対応済**: データベースから毎回特徴量を生成せず、`data/processed/preprocessed_data_v11.parquet` を直接ロードして高速に予測します（約15秒）。
- **スケジューラ**: `scripts/jit_scheduler.py`
    - 毎日自動的に上記スクリプトを呼び出し、Discordに通知します。
    - また、日次で `update_daily_features.py` を実行し、Parquetファイルを最新化しています。
- **モデル**: `models/experiments/exp_lambdarank/model.pkl`
- **キャリブレータ**: `models/experiments/exp_lambdarank/calibrator_win.pkl` (`rank == 1` で補正済)
- **戦略**: **単勝EV > 1.6** の馬を購入
    - **EV計算式**: `補正後確率 * 単勝オッズ`
    - **想定購入数**: 週4-5レース程度
    - **想定ROI**: 100-108%

**運用確認:**
- Dockerコンテナ `app` で `jit_scheduler.py` が起動していることを確認してください。
- ログファイル: `reports/jit_scheduler.log`

## 3. 今後の改善ロードマップ (Phase 4 以降)

更なる精度向上のために、以下のステップを推奨します。

### A. Top2モデル戦略 (未開拓)
- **アイデア**: 特化したTop2モデルを用いて、**馬単(Exacta)** や **馬連(Quinella)** で利益を出せないか？
- **アクション**:
    1. `Top2 Model` を `rank <= 2` または `rank == 2` をターゲットにキャリブレーションする。
    2. 馬単のEVシミュレーションを行う: `Prob(1着) * Prob(2着|1着) * 配当`。
    3. ※条件付き確率（1着が決まった後の2着確率）の算出が課題。

### B. 単勝アンサンブル
- **アイデア**: `Win Model` + `LambdaRank` + `Top3 Model` を組み合わせる。
- **状況**: 複勝ではTop3モデルが強すぎたが、単勝なら補完しあえる可能性がある。
- **アクション**:
    1. 3つのモデル全てを `rank == 1` でキャリブレーションする。
    2. `train_ensemble.py` を使い、`rank == 1` ターゲットで加重平均を最適化する。

### C. 特徴量エンジニアリング (Phase 4)
- **現状の弱点**: "データ品質" が精度向上のボトルネックになりつつある。
- **アクション**:
    1. **テキスト埋め込み**: コメントデータ (`jvd_bt.txt`) をLLMでベクトル化して特徴量に加える。
    2. **騎手/調教師ベクトル**: 騎手の特性（逃げ、差しなど）を埋め込みベクトルとして学習させる。

### D. 多クラス分類 (Multiclass)
- **アイデア**: `multiclass` 目的関数 (Class 0: 1着, Class 1: 2着, Class 3: 圏外...) を使う。
- **メリット**: 全順位の確率がSoftmaxで自然に出力されるため、キャリブレーションの手間が減る。
- **課題**: 学習が重く、メモリ消費が激しい。GPU環境の活用が必須。

## 4. 便利なコマンド・ツール

- **手動で予測を実行する**:
  ```bash
  python scripts/production_run_lambdarank.py --date YYYYMMDD
  ```
- **キャリブレーションを作成する**:
  ```bash
  python scripts/train_calibration.py --model_path <path> --target_col "rank == 1" --output_name calibrator_win.pkl
  ```
- **EVグリッドサーチ (シミュレーション)**:
  ```bash
  python scripts/adhoc/grid_search_ev_roi.py --model_path <path> --calib_name calibrator_win.pkl
  ```

---
**引継ぎ完了**
