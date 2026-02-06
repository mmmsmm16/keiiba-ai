---
description: T2 Refined v3 (No-IDs)モデルを用いた本番予測・JIT運用手順
---

# 競馬AI 本番運用ガイド (T2 v3 Refined)

## 概要
2026年1月リリースの **T2 Refined v3** モデルを使用した予測手順です。
従来のNo-IDsモデルに加え、レース条件・展開・トレンド等の新規特徴量を導入し、精度(AUC 0.7958)を向上させています。

## クイックスタート

### 当日のレース予測 (手動実行)
```bash
docker compose exec app python scripts/production_run_t2_v3.py
```

### 特定日付の予測
```bash
docker compose exec app python scripts/production_run_t2_v3.py --date 20260111
```

### Discord通知付きで実行
```bash
docker compose exec app python scripts/production_run_t2_v3.py --discord
```

---

## 自動実行 (JIT Scheduler) への適用

`scripts/jit_scheduler.py` を修正し、呼び出すスクリプトを v3 に変更してください。

```python
# scripts/jit_scheduler.py の修正箇所
CMD = ["python", "scripts/production_run_t2_v3.py", "--discord"]
```

### スケジューラの起動
```bash
docker compose exec -d app python scripts/jit_scheduler.py
```

---

## 使用モデル詳細

| 項目 | 値 |
|------|-----|
| モデルパス | `models/experiments/exp_t2_refined_v3/model.pkl` |
| 設定ファイル | `models/experiments/exp_t2_refined_v3/config.yaml` |
| 特徴量数 | 約153 |
| ID特徴量 | なし |
| Valid AUC | 0.7958 (2023) |

## トラブルシューティング

### 予測結果が出ない
- データがロードできているか確認 (`scripts/production_run_t2_v3.py` のログ)
- 直前オッズ (`jvd_o1`) がDBに入っているか確認

### エラー: `categorical_feature do not match`
- `config.yaml` の特徴量設定と、学習済みモデル(`model.pkl`)の整合性を確認してください。
- `scripts/production_run_t2_v3.py` は自動カテゴリ検出を実装済みです。

### オッズが正しく表示されない
- `jvd_o1` テーブルの `odds_tansho` カラムは、`馬番(2桁) + オッズ(4桁) + 人気(2桁)` の8バイト固定長フォーマットで格納されています。
- スクリプトはこのフォーマットに従ってパースし、欠損値(`----`等)は適切に処理します。
