# Auto-Prediction System (Phase J) Deployment Guide

## 1. 運用手順 (How to Run)

本システムは `src/runtime/race_scheduler.py` を常駐プロセスとして実行することで稼働します。

### 起動コマンド (Docker環境)
```bash
# バックグラウンドで起動 (推奨)
docker compose exec -d app python -u src/runtime/race_scheduler.py >> logs/scheduler.log 2>&1

# または、フォアグラウンドで動作確認
docker compose exec app python src/runtime/race_scheduler.py
```

### 停止方法
```bash
# プロセスIDを探して kill (簡易的)
docker compose exec app pkill -f race_scheduler.py
```

### ログ確認
```bash
tail -f logs/scheduler.log
```

---

## 2. システムの仕様と堅牢性

### 状態管理 (Persistence)
- **ファイル**: `data/runtime_state.json`
- **役割**:
  - **重複送信防止**: 送信済みのレースIDを記録し、再起動しても二重送信を防ぎます。
  - **予算管理**: 当日の使用金額累計 (`daily_spent`) を保持し、1日上限キャップ (`day_cap_total`) を正しく機能させます。
  - **自動リセット**: 日付が変わると自動でリセットされます。

### ログ・監査
- **実行ログ**: `logs/scheduler.log` (標準出力)
- **詳細ログ (Pre-race)**: `reports/logs/prerace_{race_id}.json`
  - 通知時点の全データ（予測スコア、オッズ、計算されたEV、判断理由、買い目）がJSONで保存されます。後日の分析やデバッグに使用します。

---

## 3. 拡張性 (Extensibility)

本システムは **「設定(Config)」「取得(Fetcher)」「決定(Strategy)」「通知(Notifier)」** が疎結合になっており、拡張が容易です。

### Q1. モデルを新しくしたい (V14, V15...)
- **変更箇所**:
  1. `config/runtime/phase_j_v1.yaml` の `prediction.model_version` を更新。
  2. `race_scheduler.py` 内の `_mock_predict` (現在はモック) を新しいモデルの推論クラスに差し替えるだけでOK。
  - **予測ロジック**と**意思決定ロジック(StrategyEngine)**が分離されているため、モデルが変わっても「買い方（予算・分散）」のロジックはそのまま再利用できます。

### Q2. 買い目を新しくしたい (三連単など)
- **変更箇所**:
  1. `config/runtime/phase_j_v1.yaml` の `strategies` セクションに新ルール (`trifecta_opt` 等) を追加。
  2. `src/runtime/strategy_engine.py` にそのルールの判定ロジックを追記。
  - 既存の予算管理ロジックが自動的に新しい戦略も考慮して優先順位付けを行います。

### Q3. 地方競馬 (NAR) に対応したい
- **変更箇所**:
  1. **Fetcher**: `OddsFetcher` を拡張し、地方競馬のテーブル (`n_sokuho` 等) を参照するようにする。
  2. **Scheduler**: `get_todays_races` を拡張し、NARの開催スケジュールを取得するようにする。
  - **StrategyEngine** や **DiscordNotifier** は、入力データ形式さえ合わせれば**そのまま流用可能**です。

---

## 3. 次のアクション (Go Liveに向けて)

現在のコード (`src/runtime/race_scheduler.py`) は、推論部分が **モック (Mock)** になっています。
実運用を開始するには、以下のステップが必要です：

1. **Model Integration**: `_mock_predict` を削除し、既存の `AutoPredictor` (V13モデル) を呼び出すように書き換える。
2. **Cron/Daemon化**: サーバー再起動時にも自動で立ち上がるように、Dockerの `entrypoint` に含めるか、プロセス管理ツール (Supervisor等) を導入する。
