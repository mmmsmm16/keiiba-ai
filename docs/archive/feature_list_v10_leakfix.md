# 競馬AI: 学習データ特徴量一覧 (v10_leakfix)

**総特徴量数**: 165

---

## 基本情報 (レースメタデータ) (17個)

1. `race_number`
2. `distance`
3. `frame_number`
4. `horse_number`
5. `age`
6. `impost`
7. `weight_diff`
8. `sex_num`
9. `weather_num`
10. `surface_num`
11. `state_num`
12. `year`
13. `month`
14. `day`
15. `weekday`
16. `class_level`
17. `n_horses`

## 過去走特徴 (Lag/Rolling) (12個)

1. `lag1_rank`
2. `mean_rank_5`
3. `mean_last_3f_5`
4. `total_races`
5. `mean_rank_all`
6. `wins_all`
7. `win_rate_all`
8. `total_prize`
9. `lag1_last_3f_rank`
10. `lag1_race_member_strength`
11. `lag1_performance_value`
12. `lag1_n_horses`

## カテゴリ統計 (基本) (16個)

1. `jockey_id_n_races`
2. `jockey_id_win_rate`
3. `jockey_id_top3_rate`
4. `trainer_id_n_races`
5. `trainer_id_win_rate`
6. `trainer_id_top3_rate`
7. `sire_id_n_races`
8. `sire_id_win_rate`
9. `sire_id_top3_rate`
10. `class_level_n_races`
11. `class_level_win_rate`
12. `class_level_top3_rate`
13. `jockey_id_win_rate_deviation`
14. `trainer_id_win_rate_deviation`
15. `jockey_id_win_rate_race_rank`
16. `trainer_id_win_rate_race_rank`

## コンテキスト統計 (組み合わせ) (29個)

1. `jockey_course_n_races`
2. `jockey_course_win_rate`
3. `jockey_course_top3_rate`
4. `sire_course_n_races`
5. `sire_course_win_rate`
6. `sire_course_top3_rate`
7. `trainer_course_n_races`
8. `trainer_course_win_rate`
9. `trainer_course_top3_rate`
10. `sire_dist_n_races`
11. `sire_dist_win_rate`
12. `sire_dist_top3_rate`
13. `jockey_surface_n_races`
14. `jockey_surface_win_rate`
15. `jockey_surface_top3_rate`
16. `jockey_dist_n_races`
17. `jockey_dist_win_rate`
18. `jockey_dist_top3_rate`
19. `trainer_surface_n_races`
20. `trainer_surface_win_rate`
21. `trainer_surface_top3_rate`
22. `trainer_dist_n_races`
23. `trainer_dist_win_rate`
24. `trainer_dist_top3_rate`
25. `jockey_trainer_n_races`
26. `jockey_trainer_win_rate`
27. `jockey_trainer_top3_rate`
28. `trainer_jockey_count`
29. `frame_surface_winrate`

## 血統特徴 (8個)

1. `sire_avg_rank`
2. `sire_win_rate`
3. `sire_roi_rate`
4. `sire_count`
5. `bms_avg_rank`
6. `bms_win_rate`
7. `bms_roi_rate`
8. `bms_count`

## 展開・ペース特徴 (11個)

1. `nige_rate`
2. `interval`
3. `momentum_slope`
4. `rest_score`
5. `race_avg_nige_rate`
6. `race_nige_horse_count`
7. `race_nige_bias`
8. `race_pace_cat`
9. `race_avg_prize`
10. `race_avg_age`
11. `horse_pace_disadv_rate`

## 不利検出特徴 (5個)

1. `horse_slow_start_rate`
2. `horse_wide_run_rate`
3. `horse_track_bias_rate`
4. `prev_disadvantage_score`
5. `avg_disadvantage_score_3races`

## 相対的特徴 (6個)

1. `weight_deviation`
2. `age_deviation`
3. `impost_deviation`
4. `weight_relative`
5. `age_relative`
6. `impost_relative`

## リアルタイム特徴 (5個)

1. `trend_win_inner_rate`
2. `trend_win_mid_rate`
3. `trend_win_outer_rate`
4. `trend_win_front_rate`
5. `trend_win_fav_rate`

## 埋め込み特徴 (Embedding) (32個)

1. `horse_id_emb_0`
2. `horse_id_emb_1`
3. `horse_id_emb_2`
4. `horse_id_emb_3`
5. `horse_id_emb_4`
6. `horse_id_emb_5`
7. `horse_id_emb_6`
8. `horse_id_emb_7`
9. `jockey_id_emb_0`
10. `jockey_id_emb_1`
11. `jockey_id_emb_2`
12. `jockey_id_emb_3`
13. `jockey_id_emb_4`
14. `jockey_id_emb_5`
15. `jockey_id_emb_6`
16. `jockey_id_emb_7`
17. `trainer_id_emb_0`
18. `trainer_id_emb_1`
19. `trainer_id_emb_2`
20. `trainer_id_emb_3`
21. `trainer_id_emb_4`
22. `trainer_id_emb_5`
23. `trainer_id_emb_6`
24. `trainer_id_emb_7`
25. `sire_id_emb_0`
26. `sire_id_emb_1`
27. `sire_id_emb_2`
28. `sire_id_emb_3`
29. `sire_id_emb_4`
30. `sire_id_emb_5`
31. `sire_id_emb_6`
32. `sire_id_emb_7`

## 経験値特徴 (9個)

1. `course_experience`
2. `course_best_rank`
3. `distance_experience`
4. `distance_best_rank`
5. `first_distance_cat`
6. `first_turf`
7. `first_dirt`
8. `jockey_change_flag`
9. `is_career_high_impost`

## レースレベル特徴 (2個)

1. `race_member_strength`
2. `relative_strength`

## その他 (13個)

1. `sire_track_n_races`
2. `sire_track_win_rate`
3. `sire_track_top3_rate`
4. `is_long_break`
5. `jockey_recent_win_rate`
6. `trainer_recent_win_rate`
7. `frame_zone`
8. `distance_category`
9. `jockey_distance_winrate`
10. `relative_popularity_rank`
11. `estimated_place_rate`
12. `class_gap`
13. `is_class_up`
