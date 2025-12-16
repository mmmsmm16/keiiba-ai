# v13 競馬AI パイプライン: インプット/アウトプット整理

**最終更新**: 2025-12-16

## 全体フロー図

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              v13 競馬AI パイプライン                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

[1] 前処理        [2] 学習           [3] 推論           [4] バックテスト    [5] 評価
     ↓                 ↓                  ↓                   ↓               ↓
preprocessed    →   fold models    →   predictions    →   backtest    →   reports
     data              (.txt)          (.parquet)        results          (.md)
```

---

## ステップ別 詳細

### [1] 前処理 (Preprocessing)

| 項目 | 内容 |
|------|------|
| **スクリプト** | `src/preprocessing/run_preprocessing.py` |
| **インプット** | PostgreSQL DB (`jvd_se`, `jvd_ra`, `jvd_hr` 等) |
| **アウトプット** | `data/processed/preprocessed_data_v11.parquet` |
| **主な処理** | 特徴量生成、欠損値処理、カテゴリエンコード |
| **出力行数** | 約280万行 (2013-2025, JRA+NAR) |
| **出力列数** | 約190列 |

**主要出力カラム:**

| カラム | 説明 |
|--------|------|
| `race_id` | レースID (例: 202501010101) |
| `horse_id` | 馬ID |
| `horse_number` | 馬番 (チケット生成に使用) |
| `odds` | オッズ |
| `rank` | 着順 (正解ラベル) |
| 各種特徴量 | bloodline, jockey, trainer, course, distance 等 |

---

### [2] モデル学習 (Training: Walk-Forward)

| 項目 | 内容 |
|------|------|
| **スクリプト** | `src/phase6/train_market_residual_wf.py` |
| **インプット** | `data/processed/preprocessed_data_v11.parquet` (2022-2024) |
| **アウトプット** | 下記参照 |
| **学習方式** | Walk-Forward (年単位 fold) |
| **ターゲット** | `is_winner = (rank == 1)` |
| **アルゴリズム** | LightGBM (Market Residual) |

**アウトプット詳細:**

| ファイル | 説明 |
|----------|------|
| `models/v13_market_residual/v13_fold_2022.txt` | Fold 1 モデル (train: ~2021, val: 2022) |
| `models/v13_market_residual/v13_fold_2023.txt` | Fold 2 モデル (train: ~2022, val: 2023) |
| `models/v13_market_residual/v13_fold_2024.txt` | Fold 3 モデル (train: ~2023, val: 2024) |
| `data/predictions/v13_market_residual_oof.parquet` | OOF予測 (2022-2024) |
| `reports/phase6_market_residual_wf.md` | 学習レポート |

**OOF予測ファイル構造:**

| カラム | 説明 |
|--------|------|
| `race_id`, `horse_id` | キー |
| `date`, `year` | 日付情報 |
| `odds`, `p_market` | オッズ、市場確率 |
| `rank` | 着順 (正解) |
| `prob_residual_raw` | 生予測値 |
| `score_logit`, `delta_logit` | Logitスコア |
| `prob_residual_norm` | 正規化確率 |
| **`prob_residual_softmax`** | **最終予測確率 (使用する列)** |

---

### [3] 2025推論 (Inference for Holdout)

| 項目 | 内容 |
|------|------|
| **スクリプト** | `src/phase6/infer_v13_2025.py` |
| **インプット (特徴量)** | `data/processed/preprocessed_data_v11.parquet` (2025のみ) |
| **インプット (モデル)** | `models/v13_market_residual/v13_fold_{2022,2023,2024}.txt` |
| **アウトプット** | `data/predictions/v13_market_residual_2025_infer.parquet` |
| **処理** | 3つのfoldモデルのアンサンブル（平均） |

**推論出力ファイル構造:**

```
race_id, horse_id, date, year, odds, p_market
prob_residual_raw, score_logit, delta_logit
prob_residual_norm, prob_residual_softmax  ← バックテストで使用
```

---

### [4] バックテスト (Backtest)

| 項目 | 内容 |
|------|------|
| **スクリプト** | `src/backtest/multi_ticket_backtest_v2.py` |
| **インプット (ベース)** | `data/processed/preprocessed_data_v11.parquet` |
| **インプット (予測)** | `data/predictions/v13_market_residual_2025_infer.parquet` |
| **インプット (払戻)** | PostgreSQL (`jvd_hr` → payout_map) |
| **アウトプット** | `reports/phase8/phase7_backtest_v2_jra_only.md` |

**主な設定パラメータ:**

| パラメータ | 値 | 説明 |
|------------|-----|------|
| `--prob_col` | `prob_residual_softmax` | 順位付けに使用する確率列 |
| `--odds_source` | `final` | オッズ取得タイミング |
| `--slippage_factor` | `0.90` | スリッページ係数 |
| `--ticket` | `sanrenpuku`, `sanrentan`, `umaren` | 馬券種類 |
| `--topn` | 3, 4, 5, 6 | BOX対象頭数 |
| `--bankroll` | 100,000 | 初期資金 |
| `--max_bet_frac` | 0.05 | 最大賭け比率 |

---

### [5] Placebo評価 (Validation)

| 項目 | 内容 |
|------|------|
| **スクリプト** | `scripts/run_placebo_sweep.py` |
| **インプット** | (バックテストと同じ) |
| **アウトプット** | 下記参照 |
| **処理** | prob列をシャッフルして複数seed実行、統計比較 |

**アウトプット:**

| ファイル | 説明 |
|----------|------|
| `reports/phase8/phase8_placebo_sweep.md` | 統計比較レポート |
| `reports/phase8/phase8_placebo_sweep.csv` | seed別詳細データ |

---

## ファイル依存関係図

```
PostgreSQL DB
     │
     ▼
┌────────────────────────────────────────┐
│ src/preprocessing/run_preprocessing.py │
└────────────────────────────────────────┘
     │
     ▼
data/processed/preprocessed_data_v11.parquet
     │
     ├───────────────────────────────────────────┐
     ▼                                           ▼
┌─────────────────────────────────────┐   ┌──────────────────────────────┐
│ src/phase6/train_market_residual_wf.py │   │ src/phase6/infer_v13_2025.py │
│ (2022-2024 学習)                     │   │ (2025 推論)                  │
└─────────────────────────────────────┘   └──────────────────────────────┘
     │                                           │
     ▼                                           ▼
models/v13_market_residual/             data/predictions/
├── v13_fold_2022.txt                   ├── v13_market_residual_oof.parquet
├── v13_fold_2023.txt                   └── v13_market_residual_2025_infer.parquet
└── v13_fold_2024.txt
                    │                            │
                    └──────────────┬─────────────┘
                                   ▼
                    ┌────────────────────────────────────────┐
                    │ src/backtest/multi_ticket_backtest_v2.py │
                    │ + PostgreSQL payout data                │
                    └────────────────────────────────────────┘
                                   │
                                   ▼
                    reports/phase8/phase7_backtest_v2_jra_only.md
                    reports/phase8/phase8_placebo_sweep.md
```

---

## 主要カラム対応表

| ステージ | 重要カラム | 説明 |
|----------|------------|------|
| **前処理** | `horse_number` | 馬番 (チケット生成に使用) |
| | `odds` | オッズ (順位付けのベース) |
| | `rank` | 着順 (正解ラベル) |
| | 各種特徴量 | ~190列 |
| **学習/推論** | `p_market` | odds逆数の正規化 |
| | `baseline_logit` | logit(p_market) |
| | **`prob_residual_softmax`** | **最終予測確率** |
| **バックテスト** | `prob` | prob_residual_softmax をコピー |
| | `hit` | 的中フラグ (0/1) |
| | `payout` | 払戻金額 |

---

## 実行コマンド一覧

```bash
# [1] 前処理
docker compose exec app python src/preprocessing/run_preprocessing.py

# [2] 学習 (Walk-Forward)
docker compose exec app python src/phase6/train_market_residual_wf.py

# [3] 2025推論
docker compose exec app python src/phase6/infer_v13_2025.py

# [4] バックテスト
docker compose exec app python src/backtest/multi_ticket_backtest_v2.py \
    --year 2025 \
    --predictions_input data/predictions/v13_market_residual_2025_infer.parquet \
    --prob_col prob_residual_softmax \
    --odds_source final --allow_final_odds \
    --slippage_factor 0.90

# [5] Placebo評価
docker compose exec app python scripts/run_placebo_sweep.py \
    --year 2025 --ticket sanrenpuku --topn 4 --n_seeds 20
```

---

## Phase8 (2025 Holdout) 結果サマリー

| Strategy | ROI | Race Hit Rate | Ticket Hit Rate | Max DD |
|----------|-----|---------------|-----------------|--------|
| **sanrenpuku BOX4** | **612.5%** | 45.9% | 11.5% | 0.01% |
| sanrentan BOX4 | 622.9% | 45.9% | 1.9% | 0.01% |

**Placebo比較 (race_shuffle, 20 seeds):**
- Normal ROI: 612.5%
- Placebo Mean: 50.3% ± 23.4%
- **統計的有意性**: ✅ (Normal > Placebo P95)
