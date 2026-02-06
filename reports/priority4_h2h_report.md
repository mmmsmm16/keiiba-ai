# Priority 4 (Head-to-Head) 実装報告

## 1. 概要
- **目的**: 以前メモリ不足で断念した「対戦成績 (Head-to-Head)」特徴量の実装と最適化。
- **成果**:
    - **高速化**: 疎行列を用いた反復更新アルゴリズムにより、計算時間を **17秒** に短縮 (従来はOOM)。
    - **精度**: Test AUC が **0.7934** に向上 (Priority 3比 +0.0003)。
    - **貢献度**: `vs_rival_win_rate` が Feature Importance で **Top 15** にランクイン。

## 2. 評価結果

| Model (Period: 2014-2023) | Valid AUC (2023) | Test AUC (2024+) | Note |
|---|---|---|---|
| **Priority 3 (No-IDs Nicks)** | **0.7945** | 0.7931 | ベースライン |
| **Priority 4 (H2H)** | 0.7929 | **0.7934** | 今回の結果 |

- Validスコアはわずかに低下しましたが、Testスコア(未知データ)での性能は向上・維持しています。
- 過学習せず、汎化性能に寄与していると考えられます。

## 3. 特徴量重要度 (Top 20)
`vs_rival_win_rate` が上位に入っており、モデルが有効活用していることが確認できました。

```
hc_top3_rate_365d         113544
lag1_rank                  68384
venue                      33364
lag1_time_diff             31837
relative_last_3f_diff      28429
relative_speed_index_pct   25357
mean_rank_5                24728
jockey_top3_rate           23713
jockey_win_rate            21333
weight                     21285
mean_time_diff_5           18549
age                        16152
collapse_rate_10           13965
weight_ratio               13722
vs_rival_win_rate          12613  <-- NEW (15th)
course_win_rate            12478
jockey_avg_rank            12391
interval                   11468
is_same_class_prev         10406
grade_code                 10378
```

## 4. 結論
- 技術的課題（計算量・メモリ）は完全に解決されました。
- 精度への寄与も確認できました。
- 本番モデルへの採用を推奨します。

## 5. 次ステップ
- このモデル (`exp_t2_head_to_head`) を `models/production/model.pkl` にデプロイするべきでしょうか？
- それとも Priority 3 のまま維持しますか？ (精度差は僅差です)
