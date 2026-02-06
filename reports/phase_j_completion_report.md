# Phase I2 & J 完了報告書

本セッションでは、予測システムの「多角化 (Phase I2)」と「実運用化 (Phase J)」を集中的に実施しました。
これにより、AIモデルは単なる予測スコアを出す存在から、**「オッズを監視し、予算を管理し、最適な買い目を自動で決定・通知するエージェント」**へと進化しました。

---

## 1. Phase I2: 券種ポートフォリオの拡張

過去オッズに依存しない「Top予想からの買い目構築」をシミュレーションし、資産形成以外の「楽しむためのレイヤー」を設計しました。

### 実施内容
- **シミュレーション**: `src/simulation/optimize_combinations.py` を作成し、Wide (ワイド), Trio (三連複), Trifecta (三連単) の各種パターンを検証。
- **評価指標**: ROIだけでなく「的中日率 (Hit Day Rate)」を重視。

### 決定事項
- **採用**: **Wide W1 (Top1-Top2 1点)**
  - **位置づけ**: Frequency Layer (的中体験重視)
  - **条件**: `p1 >= 0.4` (本命サイドかつ信頼度中)
  - **実績 (2024)**: ROI 94.4%, **的中日率 77.4%**
  - **運用**: 100円固定。他戦略が発動した場合は購入を見送る（予算節約）。
- **見送り**: 三連単・三連複
  - ROIの振れ幅が大きく、現行のCore戦略 (単勝/馬連) の足を引っ張るリスクがあるため、今回は採用を見送りました。

---

## 2. Phase J: 実運用システムの構築

「発走10分前に自動で買い目を推奨する」完全自動運用システムを実装しました。

### システムアーキテクチャ

| コンポーネント | ソースコード | 役割 |
| :--- | :--- | :--- |
| **RaceScheduler** | `src/runtime/race_scheduler.py` | **司令塔**。JRA全レースを監視し、発走10分前にパイプラインを起動。 |
| **OddsFetcher** | `src/runtime/odds_fetcher.py` | **目**。DB (`apd_sokuho`) から単勝・馬連のリアルタイムオッズを取得。 |
| **ModelWrapper** | `src/runtime/model_wrapper.py` | **脳 (予測)**。V13 LightGBMモデルをロードし、特徴量パルケットとオッズを結合して予測を実行。 |
| **StrategyEngine** | `src/runtime/strategy_engine.py` | **脳 (判断)**。予測・オッズ・予算状況に基づき、購入可否と優先順位を決定。 |
| **DiscordNotifier** | `src/runtime/discord_notifier.py` | **口**。決定内容を日本語でDiscordに通知。 |
| **Config** | `config/runtime/phase_j_v1.yaml` | **設定**。戦略パラメータ、予算上限、システム設定を一元管理。 |
| **State** | `data/runtime_state.json` | **状態**。送信済みレースIDと当日予算消化額を永続化（再起動耐性）。 |

### 実装された運用ロジック (StrategyEngine)

複数の戦略が競合した場合、以下の**優先順位(Priority)**と**排他ルール**で制御されます。

1. **Umaren High Return (Priority 1)**: `EV >= 10.0`
   - 「一撃回収」狙い。最優先で予算を確保。
2. **Umaren Balanced (Priority 2)**: `EV >= 4.0` & `P >= 1.5%`
   - 「安定収益」狙い。ただし **High Return が発動している場合は Skip** (重複リスク回避)。
3. **Win Core (Priority 3)**: `p1 >= 0.6` 等
   - 「堅実な積み上げ」。
4. **Wide Frequency (Priority 4)**: `p1 >= 0.4`
   - 「体験」。**他戦略(1~3)がどれか一つでもBUYなら Skip** (あくまで少額のお楽しみ枠)。

### 予算管理 (Budget Cap)
- **Race Cap**: `race_cap_total` (1レース上限) を設定。
- **Day Cap**: `day_cap_total` (1日上限) を設定。`data/runtime_state.json` で当日の累計額 (`daily_spent`) を管理し、上限を超えた場合は自動で **Skip (1日予算上限超過)** となります。
- **優先順位**: 予算超過時は Priority の低い戦略 (Wide > Win > Balanced > High) から順にカットされます。

### ログ・監査
- 通知時の詳細データ（予測値、オッズ、EV、理由）を `reports/logs/prerace_{race_id}.json` に保存します。これにより、後から「なぜ買ったのか/見送ったのか」を完全に追跡可能です。

---

## 3. 次のステップ

### 運用開始 (Go Live)
以下のコマンドでスケジューラを常駐化させれば、直ちに運用が開始されます。
```bash
docker compose exec -d app python -u src/runtime/race_scheduler.py >> logs/scheduler.log 2>&1
```

### 今後の拡張 (Roadmap)
- **Phase K**: 結果の自動収集と収支集計 (Post-Race Analysis)。
- **Phase L**: Web UI / Dashboard 化 (現在はDiscord通知のみ)。
- **Phase M**: 自動投票 (PAT/IPAT) への連携 (現在は通知のみ)。

---

## 関連ファイル
- [task.md](file:///C:/Users/masat/.gemini/antigravity/brain/99a12be2-ab95-4a83-a1b4-e78b7f1a4667/task.md)
- [walkthrough.md](file:///C:/Users/masat/.gemini/antigravity/brain/99a12be2-ab95-4a83-a1b4-e78b7f1a4667/walkthrough.md)
- [implementation_plan.md](file:///C:/Users/masat/.gemini/antigravity/brain/99a12be2-ab95-4a83-a1b4-e78b7f1a4667/implementation_plan.md)
