# 特徴量分析レポート (Feature Analysis Report)

## 分析日時
2025-12-08

## 対象モデル
- LightGBM v4.1 (95 features)

---

## 特徴量重要度ランキング

### TOP 20 (高インパクト)

| 順位 | 特徴量 | 重要度 (Gain) | カテゴリ |
|------|--------|---------------|----------|
| 1 | `jockey_id_win_rate_deviation` | 361,039 | 騎手 |
| 2 | `lag1_rank` | 94,795 | 前走 |
| 3 | `lag1_popularity` | 84,426 | 前走 |
| 4 | `mean_rank_all` | 80,119 | 馬統計 |
| 5 | `lag1_odds` | 73,481 | 前走 |
| 6 | `jockey_id_n_races` | 70,636 | 騎手 |
| 7 | `jockey_id_win_rate_race_rank` | 21,047 | 騎手 |
| 8 | `jockey_recent_win_rate` | 19,408 | 騎手 |
| 9 | `trainer_id_win_rate_deviation` | 17,096 | 調教師 |
| 10 | `interval` | 12,753 | レース間隔 |
| 11 | `weight_relative` | 10,685 | 馬体重 |
| 12 | `jockey_trainer_top3_rate` | 9,272 | 騎手×調教師 |
| 13 | `trainer_id_win_rate_race_rank` | 8,713 | 調教師 |
| 14 | `jockey_course_top3_rate` | 8,584 | 騎手×コース |
| 15 | `mean_last_3f_5` | 8,147 | 上がり3F |
| 16 | `age_relative` | 7,543 | 年齢 |
| 17 | `age_deviation` | 7,251 | 年齢 |
| 18 | `sire_track_win_rate` | 5,599 | 血統 |
| 19 | `weight_deviation` | 5,315 | 馬体重 |
| 20 | `jockey_surface_n_races` | 5,229 | 騎手 |

### BOTTOM 20 (低インパクト/削除候補)

| 順位 | 特徴量 | 重要度 (Gain) | 判定 |
|------|--------|---------------|------|
| 76 | `sire_id_top3_rate` | 418 | ⚠️ |
| 77 | `sire_id_win_rate` | 371 | ⚠️ |
| 78 | `year` | 338 | ⚠️ |
| 79 | `horse_slow_start_rate` | 326 | ⚠️ |
| 80 | `horse_wide_run_rate` | 325 | ⚠️ |
| 81 | `surface_num` | 283 | ⚠️ |
| 82 | `sire_roi_rate` | 281 | ⚠️ |
| 83 | `sire_count` | 233 | ⚠️ |
| 84 | `day` | 228 | ⚠️ |
| 85 | `prev_disadvantage_score` | 207 | ⚠️ |
| 86 | `state_num` | 126 | ⚠️ |
| 87 | `weekday` | 119 | ❌ 削除 |
| 88 | `weather_num` | 92 | ❌ 削除 |
| 89 | `horse_pace_disadv_rate` | 74 | ❌ 削除 |
| 90 | `race_nige_bias` | 46 | ❌ 削除 |
| 91 | `race_nige_horse_count` | 9 | ❌ 削除 |
| 92 | `race_avg_prize` | 0 | ❌ 削除 |
| 93 | `race_pace_cat` | 0 | ❌ 削除 |
| 94 | `total_prize` | 0 | ❌ 削除 |
| 95 | `is_long_break` | 0 | ❌ 削除 |

---

## 削除対象特徴量 (v5 用)

以下の10特徴量を削除することで、ノイズを減らしモデル性能向上を狙う。

```python
DROP_FEATURES_V5 = [
    'race_avg_prize',      # 重要度 0
    'race_pace_cat',       # 重要度 0
    'total_prize',         # 重要度 0
    'is_long_break',       # 重要度 0
    'race_nige_horse_count',  # 重要度 9
    'race_nige_bias',      # 重要度 46
    'horse_pace_disadv_rate',  # 重要度 74
    'weather_num',         # 重要度 92
    'weekday',             # 重要度 119
]
```

---

## 今後の追加候補特徴量

| 特徴量案 | 説明 | 期待効果 |
|----------|------|----------|
| `n_horses` | 出走頭数 | レース難易度の指標 |
| `frame_course_winrate` | 枠番×コース勝率 | コース特性を反映 |
| `dam_sire_distance_fit` | 母父×距離適性 | 血統深化 |
| `recent_3_avg_rank` | 直近3走平均着順 | 短期トレンド |
| `training_time` | 調教タイム | 調子の指標 (データあれば) |

---

## 次のステップ

1. ✅ 分析ドキュメント作成
2. [ ] `src/preprocessing/dataset.py` から削除対象特徴量を除外
3. [ ] データセット再生成 (`python src/preprocessing/build_dataset.py`)
4. [ ] LightGBM v5 学習 (`python src/model/train.py --model lgbm --version v5`)
5. [ ] 評価 (`python src/model/evaluate.py --model lgbm --version v5`)
6. [ ] リーダーボード更新
