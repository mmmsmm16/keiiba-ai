# 競馬AI ペーパートレード自動化マニュアル

## 概要

T-10m オッズを使用した三連複 BOX4 戦略の自動予測・Discord通知システム。

---

## ファイル構成

| ファイル | 説明 |
|---------|------|
| `src/scripts/auto_predict_v13.py` | メイン自動予測スクリプト |
| `src/scripts/run_auto_predict_v13_loop.py` | ループ実行ラッパー |
| `scripts/run_auto_predict.bat` | Windows タスクスケジューラ用 |
| `models/v13_market_residual/` | 予測モデル (3-fold) |
| `data/processed/preprocessed_data.parquet` | 特徴量データ |

---

## 運用フロー

> ⚠️ **重要**: 予測実行前に必ず前処理を実行してください

### Step 1: 前処理（parquet最新化）

```bash
docker compose exec app python src/preprocessing/run_preprocessing.py
```

- **所要時間**: 約25分
- **実行タイミング**: 週に1回、または新しいレースデータが入った後

### Step 2: 予測実行

```bash
# 手動テスト (dry-run)
docker compose exec app python src/scripts/auto_predict_v13.py --dry-run --date 2025-12-14

# 本番実行 (Discord通知あり)
docker compose exec app python src/scripts/auto_predict_v13.py

# ループ実行 (土日9-17時)
docker compose exec app python src/scripts/run_auto_predict_v13_loop.py
```

---

## 引数

| 引数 | 説明 |
|------|------|
| `--dry-run` | 通知せずにプレビュー |
| `--date YYYY-MM-DD` | 対象日付（省略時は当日） |

---

## 通知タイミング

- **リアルタイム**: 発走 **5-15分前** に自動検出
- **日付指定時**: 指定日の全レースを処理

---

## 戦略

| 項目 | 設定 |
|------|------|
| 券種 | 三連複 |
| 買い方 | BOX4 (4頭から3頭選び4点) |
| 1点単位 | ¥100 |
| オッズ | T-10m スナップショット |

---

## 予測の仕組み

1. **特徴量取得**: `preprocessed_data_v11.parquet` から該当レースを取得
2. **オッズ取得**: `apd_sokuho_o1` から発走10分前のオッズを取得
3. **モデル予測**: v13モデル (3-fold ensemble) で予測
4. **確率計算**: `expit(avg_pred) → softmax` で確率へ変換
5. **馬券生成**: 確率上位4頭で三連複BOX

> **Note**: parquet にデータがない場合は警告が表示されます。前処理を再実行してください。

---

## Discord設定

`.env` に以下を設定:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx
```

---

## タスクスケジューラ設定

1. タスクスケジューラを開く
2. 「基本タスクの作成」
3. トリガー: 毎週土日、9:00-17:00 (1分間隔)
4. 操作: `C:\Users\masat\MyLab\Dev\keiiba-ai\scripts\run_auto_predict.bat`

---

## トラブルシューティング

### parquet にデータがない

```
WARNING - parquetにデータなし: 5 races
→ 前処理を再実行してください
```

**解決方法**:
```bash
docker compose exec app python src/preprocessing/run_preprocessing.py
```

### オッズが取得できない

- PC-KEIBA が起動しているか確認
- `apd_sokuho_o1` テーブルにデータがあるか確認

### 通知が送信されない

- `.env` の `DISCORD_WEBHOOK_URL` を確認
- `--dry-run` を外して実行

---

## バックテスト参考値

| 戦略 | ROI | 的中率 |
|------|-----|--------|
| 三連複 BOX4 | 1111% | 27.9% |

(2025年 parquet バックテスト結果)
