# 自動通知システム セットアップガイド

**目的**: 毎朝自動で予測を実行し、Discord に買い目を送信

---

## 1. 必要な環境変数

`.env` ファイルに以下を設定:

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/あなたのWebhook URL
```

→ 取得方法: `docs/discord_setup_guide.md` を参照

---

## 2. 手動実行

```bash
# 今日の買い目を Discord に送信
docker compose exec app python scripts/daily_notify.py

# 特定日付を指定
docker compose exec app python scripts/daily_notify.py --date 2025-12-07

# プレビューのみ（送信しない）
docker compose exec app python scripts/daily_notify.py --date 2025-12-07 --dry-run
```

---

## 3. スケジューラ設定

### Windows タスクスケジューラ

1. **タスクスケジューラを開く**
   - Win+R → `taskschd.msc`

2. **基本タスクの作成**
   - 名前: `KeibaAI_Daily_Notify`
   - トリガー: 毎日 8:00

3. **操作**
   - プログラム: `cmd.exe`
   - 引数: `/c cd /d C:\Users\masat\MyLab\Dev\keiiba-ai && docker compose exec app python scripts/daily_notify.py`

### Linux cron

```bash
# crontab -e
0 8 * * * cd /path/to/keiiba-ai && docker compose exec -T app python scripts/daily_notify.py
```

---

## 4. 通知内容サンプル

```
🏇 **2025-12-07 買い目速報**
戦略: sanrenpuku BOX4 (v13モデル)
==============================

**中山 1R** 
推奨馬: 1番トコシエノヒトミ, 15番ドナソール, 4番イーガービーバー, 2番ベンガルボダイジュ
買い目:
  ・三連複 1-15-4 ¥100
  ・三連複 1-15-2 ¥100
  ・三連複 1-4-2 ¥100
  ・三連複 15-4-2 ¥100

...

==============================
合計: 36レース, 144点, ¥14,400

⚠️ 購入は自己責任でお願いします
```

---

## 5. 運用フロー

```
[毎朝 8:00] 自動実行
    ↓
[Discord] 買い目通知受信
    ↓
[手動] JRAネット投票で購入
    ↓
[レース後] paper_trade_settle.py で結果確認
```

---

## 6. トラブルシューティング

| 問題 | 対処 |
|------|------|
| 通知が届かない | DISCORD_WEBHOOK_URL を確認 |
| データなしエラー | 前処理データが最新か確認 |
| Dockerエラー | `docker compose up -d` で起動確認 |

---

## 7. ファイル一覧

| ファイル | 説明 |
|----------|------|
| `scripts/daily_notify.py` | メイン通知スクリプト |
| `.env` | Discord Webhook URL設定 |
| `data/paper_trade/{date}/notification.txt` | 送信内容のバックアップ |
