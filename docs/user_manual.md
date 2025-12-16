# 競馬AI 運用マニュアル

**バージョン**: v13 (2025-12-16)
**モデル**: v13_market_residual
**戦略**: 三連複 BOX4

---

## 目次

1. [概要](#1-概要)
2. [初期設定](#2-初期設定)
3. [毎日の運用フロー](#3-毎日の運用フロー)
4. [コマンドリファレンス](#4-コマンドリファレンス)
5. [通知の見方](#5-通知の見方)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [リスク管理](#7-リスク管理)

---

## 1. 概要

### システム構成

```
PostgreSQL DB (PC-Keiba)
    ↓ InferenceDataLoader
当日の出走・オッズデータ
    ↓ InferencePreprocessor
特徴量生成（190+特徴量）
    ↓ v13モデル推論
勝率予測 (softmax確率)
    ↓
Discord通知 → 手動購入
```

### 戦略概要

| 項目 | 値 |
|------|-----|
| 馬券種 | 三連複 (sanrenpuku) |
| 方式 | BOX4 (上位4頭の組み合わせ) |
| 1レース点数 | 4点 |
| 1レース投資額 | ¥400 |

---

## 2. 初期設定

### 2.1 Discord Webhook設定

1. Discordでサーバー作成（または既存サーバーを使用）
2. チャンネル設定 → 連携サービス → Webhook作成
3. Webhook URLをコピー
4. `.env` ファイルに追記:

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx/yyyyy
```

詳細: `docs/discord_setup_guide.md`

### 2.2 Dockerの起動確認

```bash
cd C:\Users\masat\MyLab\Dev\keiiba-ai
docker compose up -d
```

### 2.3 PC-Keibaデータ更新

- **毎週土曜の朝までに**出馬表データを取得
- オッズは**レース当日**に更新される

---

## 3. 毎日の運用フロー

### 3.1 レース開催日の朝（推奨）

```
[08:00] Dockerが起動していることを確認
[08:30] レースモニター起動
```

```bash
docker compose exec app python scripts/race_monitor.py
```

### 3.2 自動処理

```
[レース発走15分前] 
    ↓ 自動で予測実行
    ↓ Discord通知受信
    
[通知受信後]
    ↓ JRAネット投票で購入
    ↓ (締切まで余裕を持って)
```

### 3.3 レース終了後（オプション）

結果を記録したい場合:

```bash
docker compose exec app python scripts/paper_trade_settle.py \
  --date 2025-12-21 \
  --config configs/runtime/paper_trade.yaml
```

---

## 4. コマンドリファレンス

### 4.1 レース直前通知（推奨）

```bash
# 当日モニタリング開始（終日自動実行）
docker compose exec app python scripts/race_monitor.py

# 通知タイミング調整（デフォルト15分前）
docker compose exec app python scripts/race_monitor.py --minutes-before 20

# 特定日付でテスト
docker compose exec app python scripts/race_monitor.py --date 2025-12-21 --dry-run
```

### 4.2 全レース一括通知

```bash
# 全レースの予測を一度に通知
docker compose exec app python scripts/daily_notify_realtime.py

# プレビューのみ
docker compose exec app python scripts/daily_notify_realtime.py --dry-run
```

### 4.3 結果精算

```bash
# 日次精算（結果記録・レポート生成）
docker compose exec app python scripts/paper_trade_settle.py \
  --date 2025-12-21 \
  --config configs/runtime/paper_trade.yaml
```

### 4.4 期間集計

```bash
# 週次・月次集計
docker compose exec app python scripts/paper_trade_aggregate.py \
  --start 2025-12-01 --end 2025-12-31 \
  --config configs/runtime/paper_trade.yaml
```

---

## 5. 通知の見方

### 5.1 通知サンプル

```
🏇 **中山 11R** 15:25
_ステイヤーズS_

📊 **予測スコア** (勝率)
⭐  5番 ダイヤモンド    18.2%
⭐  3番 ステイゴールド   15.6%
⭐  8番 シルヴァーソニッ  13.4%
⭐ 12番 テーオーロイヤル   9.8%
    1番 マイネルファンロ   8.5%
    7番 ディバインフォー   7.3%
    ...

🎯 **推奨 TOP4**: 5番, 3番, 8番, 12番
📝 **買目**: 5-3-8, 5-3-12, 5-8-12, 3-8-12 (各¥100)
💰 合計: 4点 ¥400
⚠️ 締切に注意！
```

### 5.2 各項目の意味

| 項目 | 説明 |
|------|------|
| **⭐マーク** | 購入対象のTOP4 |
| **勝率%** | モデルが予測した1着確率（合計100%） |
| **買目** | 三連複の馬番組み合わせ |
| **締切** | JRAは発走2分前まで |

### 5.3 購入方法

1. JRAネット投票 (https://www.ipat.jra.go.jp/)
2. 馬券種: **三連複**
3. 方式: **ボックス** または **通常**
4. 馬番: 通知のTOP4を入力
5. 金額: 各100円

---

## 6. トラブルシューティング

### 6.1 通知が届かない

| 原因 | 対処 |
|------|------|
| Webhook URL未設定 | `.env` を確認 |
| Dockerが停止 | `docker compose up -d` |
| ネットワークエラー | インターネット接続確認 |

### 6.2 「データがありません」エラー

| 原因 | 対処 |
|------|------|
| PC-Keiba未更新 | 出馬表データを取得 |
| 日付形式エラー | `YYYY-MM-DD` 形式で指定 |
| 非開催日 | JRAカレンダー確認 |

### 6.3 予測に時間がかかる

- 初回実行時は履歴データ読み込みで1-2分かかります
- 2回目以降は30秒程度

### 6.4 Dockerエラー

```bash
# コンテナ再起動
docker compose restart

# ログ確認
docker compose logs app --tail 50
```

---

## 7. リスク管理

### 7.1 資金管理ルール

| 項目 | 推奨値 |
|------|--------|
| 初期資金 | ¥100,000 |
| 1レース上限 | ¥400 |
| 1日上限 | ¥12,000 |

### 7.2 MaxDD（最大ドローダウン）ルール

| 閾値 | アクション |
|------|------------|
| -30% | ⚠️ 賭け金半減 |
| -50% | 🛑 1週間停止 |
| -70% | ❌ 戦略見直し |

### 7.3 実績との乖離確認

週次で以下を確認:
- バックテストROI (612%) との乖離
- 的中率 (目標: ~35%)

---

## 付録

### ファイル構成

```
scripts/
├── race_monitor.py          # レース直前通知（推奨）
├── daily_notify_realtime.py  # 全レース一括通知
├── paper_trade_settle.py     # 結果精算
└── paper_trade_aggregate.py  # 期間集計

configs/
└── runtime/paper_trade.yaml  # 運用設定

docs/
├── user_manual.md           # このマニュアル
├── discord_setup_guide.md   # Discord設定
└── operation_checklist.md   # 運用チェックリスト
```

### サポート

問題発生時はログを確認:
```bash
docker compose logs app --tail 100
```

---

**免責事項**: 本システムは投資助言ではありません。馬券購入は自己責任で行ってください。
