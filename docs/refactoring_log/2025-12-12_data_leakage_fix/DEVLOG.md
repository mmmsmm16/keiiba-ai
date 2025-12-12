# 開発ログ: データリーク修正 (2025-12-12)

## 概要

第三者レビューで指摘されたデータリーク問題を修正し、モデルを再学習した。

---

## 対応した課題

1. **CategoryAggregator データリーク** (Critical)
2. **NumPy 2.x 互換性** (Critical)
3. **テストコード不足** (Important)

---

## 修正内容

### 1. CategoryAggregator リーク修正

**問題点:**
- `_aggregate_basic()` と `_aggregate_context()` で `shift(1)` を使用
- 同一レース内の同カテゴリ（調教師、種牡馬等）が隣接するため、同レース結果がリーク

**修正:**
- race_id単位で事前集約し、累積統計を計算してからマージする方式に変更
- `observed=True` を追加してメモリ爆発を防止

**ファイル:** `src/preprocessing/category_aggregators.py`

### 2. NumPy バージョン固定

```
numpy>=1.24.0,<2.0.0
```

**ファイル:** `docker/python/requirements.txt`

### 3. ユニットテスト作成

6つのテストケースを作成し全てパス。

**ファイル:** `src/preprocessing/test_category_aggregators.py`

---

## 新規データセット

| ファイル | サイズ | 説明 |
|---------|-------|------|
| `lgbm_datasets_v10_leakfix.pkl` | 2.15 GB | リーク修正後のデータセット |
| `preprocessed_data_v10_leakfix.parquet` | 789 MB | 中間データ |

---

## モデル学習結果 (v21)

| モデル | メトリクス |
|-------|----------|
| LightGBM v21 | NDCG@5: 0.597, RMSE: 3.05 |
| CatBoost v21 | NDCG: 0.744, RMSE: 1.33 |
| Ensemble v21 | RMSE: 0.86 |

### Ensemble Meta-Model Weights
- LightGBM: -0.024
- CatBoost: 0.280
- Bias: 0.513

---

## 評価結果 (v21, 2025年JRAレース)

| 戦略 | ROI | 的中率 | 備考 |
|-----|-----|-------|------|
| Max Score (単勝1点) | 60.61% | 35.71% | |
| Sanrentan Box5 | 90.76% | 32.12% | |
| Sanrenpuku Box5 | 81.94% | 32.12% | |
| Umaren Box5 | 82.42% | 53.82% | |
| Sanrenpuku Nagashi | 83.80% | 33.65% | |

> **注意:** リーク修正前のモデル(v12等)と比較してROIが低下しているが、これが「本来の性能」。
> 以前の高ROIはデータリークによる過大評価だった。

---

## スクリプト修正

### evaluate.py

`--dataset_suffix` パラメータを追加。

```bash
docker-compose exec app python src/model/evaluate.py \
  --model ensemble --version v21 --years 2025 \
  --dataset_suffix _v10_leakfix
```

---

## 新規ファイル

| パス | 説明 |
|-----|------|
| `src/preprocessing/test_category_aggregators.py` | ユニットテスト |
| `docs/refactoring_log/README.md` | リファクタリングログ |
| `docs/refactoring_log/2025-12-12_data_leakage_fix/CHANGELOG.md` | 変更履歴 |
| `docs/refactoring_log/2025-12-12_data_leakage_fix/affected_models.md` | 影響モデル一覧 |
| `config/experiments/exp_v20_leakfix.yaml` | v20実験設定(TabNet有) |
| `config/experiments/exp_v21_leakfix_no_tabnet.yaml` | v21実験設定(TabNet無) |

---

## 変更ファイル

| パス | 変更内容 |
|-----|---------|
| `src/preprocessing/category_aggregators.py` | リークロジック修正 |
| `docker/python/requirements.txt` | NumPyバージョン固定 |
| `src/model/evaluate.py` | dataset_suffix対応 |
| `src/model/ensemble.py` | インポートパス修正 |

---

## 今後の推奨作業

1. **既存モデルの再学習**: v7, v9, v12等もリーク修正データで再学習すべき
2. **TabNet込みv20学習**: 時間があればTabNet込みの学習も実施
3. **スクリプト整理**: `src/scripts/adhoc/` の76ファイルの整理
