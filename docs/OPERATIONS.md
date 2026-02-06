# Keiiba-AI 運用マニュアル (Phase T2: No-IDs High-Generalization)

本ドキュメントでは、Keiiba-AI Phase T2 (No-IDsモデル + 血統拡張) の本番運用手順について記載します。

## 1. 概要

- **モデル**: `models/production/model.pkl`
    - 特徴: 馬ID/騎手IDなどを使用せず、血統統計(Nicks)や変化パターン(体重/休養)を重視した高汎化モデル。
    - 精度: Valid AUC 0.7945 / Test AUC 0.7931
- **戦略**: 単勝 (Win) のみ / 期待値(EV) > 1.0

## 2. 日次ルーチン (Daily Operation)

レース当日の運用は以下の2ステップで行います。

### Step 1: 統計キャッシュの更新 (必須)
本モデルは種牡馬×母父の組み合わせ(Nicks)など、膨大な過去データに基づく統計量を使用します。
これを高速に参照するため、事前に集計キャッシュ(`nicks_stats.parquet` 等)を更新してください。

**実行タイミング**: 当日のレース予測前 (1日1回)
**所要時間**: 約5〜10分

```powershell
docker compose exec app python scripts/update_aggregates_cache.py
```

### Step 2: 予測と実行 (JIT Hybrid)
直近のオッズと、Step 1で作成したキャッシュ、および当日の馬情報を組み合わせて予測を行います。

**実行タイミング**: レース発走10分前 (定期実行) または 手動実行
**実行コマンド**:

```powershell
# 今日の全レースを予測・Discord通知
docker compose exec app python scripts/production_run_t2_jit.py --discord
```

## 3. 設定ファイル

- **ポリシー設定**: `config/production_policy.yaml`
    - 予算、ベット戦略、対象券種などを定義。
    - 現状: `bet_types: [win]`, `min_ev_threshold: 1.0`

## 4. トラブルシューティング

- **Q. "nicks_stats.parquet not found" エラー**
    - A. Step 1 のキャッシュ更新が行われていません。`update_aggregates_cache.py` を実行してください。

- **Q. "Model file not found"**
    - A. `models/production/model.pkl` が存在しません。実験ディレクトリからコピーしてください。
    - `cp models/experiments/exp_t2_bloodline/model.pkl models/production/model.pkl`
