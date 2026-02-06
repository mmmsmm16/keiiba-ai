# 影響を受けたモデル一覧

**調査日:** 2025-12-12  
**対象:** データリーク (`CategoryAggregator`) の影響を受けたモデル

---

## 概要

`CategoryAggregator` は `src/preprocessing/run_preprocessing.py` で呼び出され、
生成されたデータセット (`lgbm_datasets*.pkl`) を使用してすべてのモデルが学習されています。

したがって、**すべての既存モデルがリークの影響を受けています。**

---

## 影響を受けたデータセット

| ファイル | サイズ | ステータス |
|---------|-------|----------|
| `data/processed/lgbm_datasets.pkl` | 2.35 GB | ❌ リーク含む |
| `data/processed/lgbm_datasets_jra_v5.pkl` | 1.53 GB | ❌ リーク含む |
| `data/processed/lgbm_datasets_v7.pkl` | 640 MB | ❌ リーク含む |
| `data/processed/lgbm_datasets_v8.pkl` | 1.29 GB | ❌ リーク含む |
| `data/processed/lgbm_datasets_v9.pkl` | 709 MB | ❌ リーク含む |
| `data/processed/preprocessed_data.parquet` | 1.13 GB | ❌ リーク含む |
| `data/processed/preprocessed_data_jra_v5.parquet` | 685 MB | ❌ リーク含む |
| `data/processed/preprocessed_data_v8.parquet` | 477 MB | ❌ リーク含む |
| `data/processed/base_features_all.parquet` | 957 MB | ⚠️ 要確認 |

---

## 影響を受けたモデル

### LightGBM モデル

| モデル | ファイル | 再学習必要 |
|-------|---------|----------|
| lgbm (v1) | `lgbm.pkl` | ✅ 必要 |
| lgbm_v2 | `lgbm_v2.pkl` | ✅ 必要 |
| lgbm_v3 | `lgbm_v3.pkl` | ✅ 必要 |
| lgbm_v4 | `lgbm_v4.pkl` | ✅ 必要 |
| lgbm_v4_1 | `lgbm_v4_1.pkl` | ✅ 必要 |
| lgbm_v4_2025 | `lgbm_v4_2025.pkl` | ✅ 必要 |
| lgbm_v4_emb | `lgbm_v4_emb.pkl` | ✅ 必要 |
| lgbm_v5 | `lgbm_v5.pkl` | ✅ 必要 |
| lgbm_v5_2025 | `lgbm_v5_2025.pkl` | ✅ 必要 |
| lgbm_v5_weighted | `lgbm_v5_weighted.pkl` | ✅ 必要 |
| lgbm_v6 | `lgbm_v6.pkl` | ✅ 必要 |
| lgbm_v7 | `lgbm_v7.pkl` | ✅ 必要 |

### CatBoost モデル

| モデル | ファイル | 再学習必要 |
|-------|---------|----------|
| catboost (v1) | `catboost.pkl` | ✅ 必要 |
| catboost_v3 | `catboost_v3.pkl` | ✅ 必要 |
| catboost_v4 | `catboost_v4.pkl` | ✅ 必要 |
| catboost_v4_1 | `catboost_v4_1.pkl` | ✅ 必要 |
| catboost_v4_2025 | `catboost_v4_2025.pkl` | ✅ 必要 |
| catboost_v5 | `catboost_v5.pkl` | ✅ 必要 |
| catboost_v5_2025 | `catboost_v5_2025.pkl` | ✅ 必要 |
| catboost_v7 | `catboost_v7.pkl` | ✅ 必要 |
| catboost_v8 | `catboost_v8.pkl` | ✅ 必要 |
| catboost_v9 | `catboost_v9.pkl` | ✅ 必要 |
| catboost_v9_emb | `catboost_v9_emb.pkl` | ✅ 必要 |

### TabNet モデル

| モデル | ファイル | 再学習必要 |
|-------|---------|----------|
| tabnet (v1) | `tabnet.zip` | ✅ 必要 |
| tabnet_v3 | `tabnet_v3.zip` | ✅ 必要 |
| tabnet_v4 | `tabnet_v4.zip` | ✅ 必要 |
| tabnet_v4_1 | `tabnet_v4_1.zip` | ✅ 必要 |

### Ensemble モデル

| モデル | ファイル | 再学習必要 |
|-------|---------|----------|
| ensemble (v1) | `ensemble_model.pkl` | ✅ 必要 |
| ensemble_v3 | `ensemble_v3.pkl` | ✅ 必要 |
| ensemble_v4_2025 | `ensemble_v4_2025.pkl` | ✅ 必要 |
| ensemble_v5 | `ensemble_v5.pkl` | ✅ 必要 |
| ensemble_v5_2025 | `ensemble_v5_2025.pkl` | ✅ 必要 |
| ensemble_v7 | `ensemble_v7.pkl` | ✅ 必要 |

### Betting モデル

| モデル | ファイル | 再学習必要 |
|-------|---------|----------|
| betting_model | `betting_model.pkl` | ✅ 必要 |
| betting_model_place | `betting_model_place.pkl` | ✅ 必要 |
| betting_model_win | `betting_model_win.pkl` | ✅ 必要 |
| betting_model_v5_weighted_* | 複数ファイル | ✅ 必要 |

---

## 再学習の優先順位

リーク修正後、以下の順序で再学習を推奨:

1. **最新かつ主要なモデル (High Priority)**
   - `lgbm_v7.pkl` - 現在の主要モデル
   - `catboost_v9.pkl` / `catboost_v9_emb.pkl`
   - `ensemble_v7.pkl`

2. **2025年版モデル (Medium Priority)**
   - `*_v5_2025.pkl` 系
   - `*_v4_2025.pkl` 系

3. **旧バージョン (Low Priority)**
   - v1-v6 のモデルは必要に応じて

---

## 保存すべき旧モデル (バックアップ)

修正前の性能比較のため、以下は `models/archive_pre_leakfix/` にバックアップ:

```
models/archive_pre_leakfix/
├── lgbm_v7.pkl
├── catboost_v9.pkl
├── ensemble_v7.pkl
└── README.md (バックアップ理由の記載)
```

---

## 再学習後の確認事項

- [ ] ROI が修正前より低下しているか確認 (リーク除去による期待される結果)
- [ ] ただし「本来の性能」なので、これが正しい値
- [ ] 新しいモデルで本番予測を開始
