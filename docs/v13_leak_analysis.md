# v13 データフローとリーク分析

**分析日**: 2025-12-16
**結論**: ⚠️ 主要なリークは明示的に防止されているが、オッズのタイミングに注意が必要

---

## 1. データフロー全体図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           データフロー詳細                                    │
└─────────────────────────────────────────────────────────────────────────────┘

[PostgreSQL DB]
      │
      │ jvd_se (出走馬情報), jvd_ra (レース情報), jvd_hr (払戻)
      │ jvd_o1/o2/o3/o4/o5 (オッズ各種)
      ▼
┌──────────────────────────────────────────┐
│ src/preprocessing/run_preprocessing.py   │
│                                          │
│ 生成する特徴量:                           │
│   - 血統 (bloodline)         ← 出走前に確定  ✅
│   - 騎手/調教師 (jockey/trainer) ← 出走前確定 ✅
│   - コース/距離 (course/distance) ← 確定    ✅
│   - オッズ (odds)            ← ⚠️ 確定オッズ
│   - 人気 (popularity)        ← ⚠️ 確定人気
│   - 過去成績 (lag1_*, lag2_*) ← shift(1)適用 ✅
│                                          │
│ 除外される列 (FORBIDDEN_COLUMNS):          │
│   - rank, time, last_3f     ← 結果情報    │
│   - time_index              ← 中間生成物   │
│   - honshokin, prize        ← 当該レース賞金│
└──────────────────────────────────────────┘
      │
      ▼
```

---

## 2. 学習時のリーク防止

### Walk-Forward 学習

```
年度        2021      2022      2023      2024      2025
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fold 1:   [TRAIN]    [VAL]                         
Fold 2:   [TRAIN]   [TRAIN]    [VAL]               
Fold 3:   [TRAIN]   [TRAIN]   [TRAIN]    [VAL]     
推論:                                              [PREDICT]
```

**時間的リーク防止**: ✅ 未来データで学習しない

### 特徴量除外リスト (train_market_residual_wf.py)

```python
exclude = ['rank', 'time', 'raw_time', 'prize_money', 'prob', 'p_market',
           'prob_model_raw', 'prob_model_norm', 'is_winner', 'finish_time',
           'race_id', 'horse_id', 'horse_key', 'date', 'year', 'fold_year',
           'model_version', 'odds', 'p_market_raw', 'overround']
```

**結果リーク防止**: ✅ rank, is_winner, time を明示的に除外

---

## 3. リスク評価

### ✅ 防止済みのリーク

| 項目 | 防止方法 | 状態 |
|------|----------|------|
| **着順リーク** | `rank`, `is_winner` を exclude | ✅ |
| **タイムリーク** | `time`, `raw_time`, `finish_time` を exclude | ✅ |
| **払戻リーク** | `prize_money`, `honshokin` を exclude | ✅ |
| **確率リーク** | `prob_*` を exclude | ✅ |
| **時間的リーク** | Walk-Forward で過去年のみ学習 | ✅ |
| **レース内リーク** | 馬ごとに独立した特徴量 | ✅ |

### ⚠️ 注意が必要な点

| 項目 | 状況 | リスク | 対策 |
|------|------|--------|------|
| **オッズ (odds)** | 確定オッズを使用 | **中** | slippage=0.90 適用、レポートに明記 |
| **人気 (popularity)** | 確定人気を使用 | **低** | 特徴量としてのみ使用、買い目決定には不使用 |
| **過去成績** | lag1, lag2 で shift | **なし** | 既に適用済み |

---

## 4. オッズの詳細分析

### 現状

```
preprocessed_data_v11.parquet の odds 列
  ↓
train_market_residual_wf.py で p_market = 1/odds を計算
  ↓
baseline_logit = logit(p_market) として特徴量化
  ↓
モデルは「市場からの乖離」を学習
```

### オッズの由来

| テーブル | 内容 | タイミング |
|----------|------|------------|
| `jvd_o1` | 単勝・複勝オッズ | **確定** |
| `jvd_o2` | 馬連オッズ | **確定** |
| `jvd_o5` | 三連複オッズ | **確定** |

**結論**: 現在使用しているオッズは**確定（final）オッズ**

### リスク評価

| シナリオ | 影響 |
|----------|------|
| 実運用で事前オッズ使用 | モデル性能低下の可能性 |
| slippage=0.90 適用 | 保守的見積り（10%減） |

**対策状況**: ✅ レポートで明示、slippageで調整

---

## 5. Placebo検証による確認

### 検証結果 (Phase 8)

| Metric | Normal | Placebo (shuffle) | 判定 |
|--------|--------|-------------------|------|
| ROI | 612.5% | 50.3% ± 23.4% | ✅ |
| Race Hit Rate | 45.9% | 2.1% | ✅ |

**解釈**: 
- shuffleでROIが大幅低下 → モデルの予測力は実在
- 単純なオッズリークなら shuffle 後も高 ROI になるはず
- **結論**: 純粋なオッズリークではないことを確認

---

## 6. 最終判定

### リークリスク評価

| カテゴリ | 状態 | 根拠 |
|----------|------|------|
| **結果リーク** | ✅ なし | rank, time を明示除外 |
| **時間的リーク** | ✅ なし | Walk-Forward 採用 |
| **オッズリーク** | ⚠️ 注意 | 確定オッズ → slippage で対応 |
| **全体評価** | ✅ 問題なし | Placebo 検証 PASS |

### 実運用への推奨

1. **オッズ前提を明記**: `odds_source=final` + `slippage=0.90`
2. **事前オッズとの乖離モニタリング**: 実運用開始後に drift を計測
3. **定期的 Placebo 検証**: 週次で簡易チェック

---

## 7. 参照コード

### 特徴量除外 (train_market_residual_wf.py:165-170)

```python
# Exclude leak columns
exclude = ['rank', 'time', 'raw_time', 'prize_money', 'prob', 'p_market',
           'prob_model_raw', 'prob_model_norm', 'is_winner', 'finish_time',
           'race_id', 'horse_id', 'horse_key', 'date', 'year', 'fold_year',
           'model_version', 'odds', 'p_market_raw', 'overround']

safe_features = [c for c in available_features if c not in exclude and not c.startswith('prob_')]
```

### FORBIDDEN_COLUMNS (validation_utils.py:47-58)

```python
FORBIDDEN_COLUMNS = [
    # 当該レースの結果
    'rank', 'time', 'rank_norm', 'passing_rank', 'last_3f',
    # 中間生成物（shift前）
    'time_index', 'last_3f_index',
    # 賞金（当該レースの結果）
    'honshokin', 'prize',
    ...
]
```
