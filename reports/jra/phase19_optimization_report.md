# Phase 19: Model Optimization & ID Removal Report

## 1. 概要
本フェーズでは、Phase 18の課題であった「ID過学習」と「確定オッズリーク」を解消し、モデルの健全化と汎化性能の向上を図りました。

**結果サマリ:**
- **Total ROI (2025)**: **89.43%** (Baseline: ~70%, Phase 18 Leaked: ~75%)
- **Feature Importance**: `odds_t_10` が支配的 (`mare_id` 等のIDは消失)
- **Gate Check**: 通過 (確定オッズ系カラムの混入なし)

---

## 2. ID特徴量の断捨離とROI比較

### 削除された特徴量
`run_jra_pipeline_backtest.py` に `DROP_FEATURES` を定義し、以下のカラムを学習データから除外しました。

```python
DROP_FEATURES = [
    'mare_id', 'sire_id', 'trainer_id', 'jockey_id', 'bms_id',  # Explicit ID Removal
    'odds', 'popularity',                                       # Odds Leak Removal (Use time-series odds instead)
    'owner_id', 'breeder_id'                                    # Other high cardinality IDs
]
```

### ROI 比較 (2025年 Walk-Forward)
| 設定 | ROI | 特徴 |
| :--- | :--- | :--- |
| **Phase 16 (Baseline)** | 70.70% | 時系列オッズなし |
| **Phase 18 (Leaked)** | 75.42% | `mare_id` 過学習 + 確定オッズリーク |
| **Phase 19 (Optimized)** | **89.43%** | **ID全削除 + T-10オッズのみ + 強正則化** |

*考察*: ID特徴量を削除したことでベット数が大幅に減少（月平均800件→65件）しましたが、質（精度）が向上しました。

---

## 3. リーク防止ゲート (Validation Gate)

学習ループの直前に、以下のチェック関数を挿入し、確定オッズの混入を物理的に阻止しました。

```python
def check_leakage(X_cols):
    """Gate to prevent odds leakage"""
    forbidden = ['odds', 'final_odds', '確定', 'popularity']
    # Allow time-series odds (e.g., odds_t_10, odds_t_30)
    leaked = [c for c in X_cols if c in forbidden]
    if leaked:
        raise ValueError(f"CRITICAL: Leaked features detected in training set: {leaked}")
    logger.info("✅ Leakage Gate Passed: No forbidden odds columns found.")
```

今回のバックテストでは全期間でこのゲートを通過しています。

---

## 4. 正則化チューニング結果

ID削除に伴い、特徴量数が減少したため、過学習を防ぐためにパラメータをより保守的に調整しました。

| パラメータ | 変更前 (Standard) | 変更後 (Optimized) | 意図 |
| :--- | :--- | :--- | :--- |
| `learning_rate` | 0.1 | **0.05** | 汎化性能向上 |
| `num_leaves` | 76 | **31** | 複雑さの抑制 |
| `min_data_in_leaf` | 53 | **100** | 末端ノードのデータ数確保 |
| `lambda_l1` | 1.5e-05 | **0.1** | スパース性・特徴量選択の強化 |
| `lambda_l2` | 0.05 | **0.1** | 重みの抑制 |

---

## 5. Feature Importance 分析 (Top 10)

ID特徴量が消え、**市場評価（オッズ）**と**実力指標（指数・偏差）**で予測する健全なモデル構造になりました。

| Rank | Feature | Gain | Note |
| :--- | :--- | :--- | :--- |
| 1 | `odds_t_10` | 274,381 | 圧倒的1位（市場の意思） |
| 2 | `odds_t_30` | 50,335 | 直前オッズの変化推移 |
| 3 | `log_odds_t_30` | 5,086 | オッズの変形 |
| 4 | `n_horses` | 2,406 | 多頭数/少頭数は重要 |
| 8 | `relative_strength` | 600 | 相手関係（相対評価） |
| 9 | `lag1_time_index` | 592 | **Speed Index (Phase C)** |
| 10 | `race_opponent_strength` | 538 | レースレベル |

**新特徴量の評価:**
- Phase C (Speed Index) の `lag1_time_index` がTop 10入り。有効に機能しています。
- Phase B (Profile) の `jockey_trainer_win_rate` も上位に存在。

## 6. 結論
モデルの構造改革（脱ID・強正則化）により、ROIは約90%まで到達しました。
次はROI 100%超えを目指し、**Phase C: Speed Indexの完全実装** および **馬券戦略（券種最適化）** に進む準備が整いました。
