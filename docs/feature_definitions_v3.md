
# 特徴量定義書 (v3) - T2 Model (Refined)
**実験ID:** `exp_t2_refined_v3`
**総特徴量数:** 258 (従来の247 + 新規11 lag venue/condition特徴量)

本モデルで使用されている全特徴量の一覧です。
※ `dataset.exclude_features` により学習から除外される特徴量も含まれています。

## モデル設定 (Model Configuration)
**学習パラメータ (Training Parameters)**
- **Model Type**: LightGBM (Gradient Boosting Decision Tree)
- **Objective**: `binary` (ターゲット: 1着 `win`)
- **Metric**: `auc` (Area Under the ROC Curve)
- **Calibration**: `isotonic` (5-fold Out-of-Fold)
- **Seed**: 42

| Parameter | Value | Description |
|:---|:---|:---|
| `learning_rate` | 0.05 | 学習率 |
| `num_leaves` | 63 | 決定木の葉の最大数 |
| `feature_fraction` | 0.8 | 特徴量のサンプリング率 (colsample_bytree) |
| `bagging_fraction` | 0.8 | データのサンプリング率 (subsample) |
| `bagging_freq` | 5 | バギング頻度 |
| `n_estimators` | 3000 | 決定木の数 (Early Stopping: 100) |

**特徴量エンジニアリング (Engineering)**
- **Fixed Baseline**: 2024年以降のFeature Driftを防ぐため、相対指標(`_z`, `_relative`)の基準値を`2024-12-31`までの統計量に固定。
- **Strict Typing**: カテゴリカル変数は厳格に`category`型として扱い、それ以外は`float`に統一。

| No. | カテゴリ | 特徴量名 (Feature Name) | 説明 (Description) | データ例 (Data Example) |
|:---:|:---|:---|:---|:---|
| 1 | 基本属性 | `age` | 馬齢 | `2 ~ 12` (e.g. 4) |
| 2 | 基本属性 | `sex` | 性別コード | `0`:牡, `1`:牝, `2`:セン |
| 3 | 基本属性 | `weight` | 馬体重 | `400 ~ 600` (e.g. 480) |
| 4 | 基本属性 | `weight_diff` | 前走からの馬体重増減 | `-20 ~ +20` (e.g. +2) |
| 5 | 基本属性 | `grade_code` | グレードコード | `0`:G1 ~ `9`:一般 |
| 6 | 基本属性 | `kyoso_joken_code` | 競走条件コード | `0`:未勝利, `5`:オープン |
| 7 | 基本属性 | `distance` | 距離 (m) | `1000 ~ 3600` (e.g. 2000) |
| 8 | 基本属性 | `surface` | トラック種別 | `1`:芝, `2`:ダート, `3`:障害 |
| 9 | 基本属性 | `venue` | 開催場所コード | `01`:札幌 ~ `10`:小倉 |
| 10 | 履歴 | `run_count` | 通算出走回数 | `0 ~ 100` (e.g. 15) |
| 11 | 履歴 | `interval` | 前走からの間隔 (日数) | `0 ~ 365+` (e.g. 14) |
| 12 | 履歴 | `mean_rank_5` | 近5走の平均着順 | `1.0 ~ 18.0` (e.g. 6.5) |
| 13 | 履歴 | `mean_time_diff_5` | 近5走の平均タイム差 | `0.0 ~ 5.0` (e.g. 0.8) |
| 14 | DeepLag | `lag1_rank` | 前走着順 | `1 ~ 18` |
| 15 | DeepLag | `lag1_time_diff` | 前走タイム差 (秒) | `0.0 ~ 5.0` (e.g. 0.5) |
| 16 | DeepLag | `lag1_race_level` | 前走レースレベル (勝馬平均賞金/1万) | `100 ~ 5000` (e.g. 1500) |
| 17 | DeepLag | `lag1_winner_level` | 前走勝馬の強さ (RivalSpeedIndex) | `40 ~ 80` (e.g. 55) |
| 18 | DeepLag | `lag1_grade` | 前走グレード | `0`:G1 ~ `9`:一般 |
| 19 | DeepLag | `lag2_rank` | 2走前着順 | `1 ~ 18` |
| 20 | DeepLag | `lag2_time_diff` | 2走前タイム差 | `0.0 ~ 5.0` |
| 21 | DeepLag | `lag2_race_level` | 2走前レースレベル | `100 ~ 5000` |
| 22 | DeepLag | `lag2_winner_level` | 2走前勝馬レベル | `40 ~ 80` |
| 23 | DeepLag | `lag2_grade` | 2走前グレード | `0 ~ 9` |
| 24 | DeepLag | `lag3_rank` | 3走前着順 | `1 ~ 18` |
| 25 | DeepLag | `lag3_time_diff` | 3走前タイム差 | `0.0 ~ 5.0` |
| 26 | DeepLag | `lag3_race_level` | 3走前レースレベル | `100 ~ 5000` |
| 27 | DeepLag | `lag3_winner_level` | 3走前勝馬レベル | `40 ~ 80` |
| 28 | DeepLag | `lag3_grade` | 3走前グレード | `0 ~ 9` |
| 29 | DeepLag | `lag4_rank` | 4走前着順 | `1 ~ 18` |
| 30 | DeepLag | `lag4_time_diff` | 4走前タイム差 | `0.0 ~ 5.0` |
| 31 | DeepLag | `lag4_race_level` | 4走前レースレベル | `100 ~ 5000` |
| 32 | DeepLag | `lag4_winner_level` | 4走前勝馬レベル | `40 ~ 80` |
| 33 | DeepLag | `lag4_grade` | 4走前グレード | `0 ~ 9` |
| 34 | DeepLag | `lag5_rank` | 5走前着順 | `1 ~ 18` |
| 35 | DeepLag | `lag5_time_diff` | 5走前タイム差 | `0.0 ~ 5.0` |
| 36 | DeepLag | `lag5_race_level` | 5走前レースレベル | `100 ~ 5000` |
| 37 | DeepLag | `lag5_winner_level` | 5走前勝馬レベル | `40 ~ 80` |
| 38 | DeepLag | `lag5_grade` | 5走前グレード | `0 ~ 9` |
| 39 | 騎手 | `jockey_n_races` | 騎手の通算レース数 | `10 ~ 20000` (e.g. 5000) |
| 40 | 騎手 | `jockey_win_rate` | 騎手の通算勝率 | `0.0 ~ 0.3` (e.g. 0.08) |
| 41 | 騎手 | `jockey_top3_rate` | 騎手の通算複勝率 | `0.0 ~ 0.6` (e.g. 0.25) |
| 42 | 騎手 | `jockey_avg_rank` | 騎手の通算平均着順 | `1.0 ~ 18.0` (e.g. 7.5) |
| 43 | 脚質 | `last_nige_rate` | 近走の逃げ率 | `0.0 ~ 1.0` (e.g. 0.2) |
| 44 | 脚質 | `avg_first_corner_norm` | 第1コーナー通過順 (正規化) | `0.0`(先頭) ~ `1.0`(最後方) |
| 45 | 脚質 | `avg_pci` | 平均PCI (50=平均ペース, >50スロー, <50ハイ) | `30 ~ 70` (e.g. 52.5) |
| 46 | 血統 | `sire_n_races` | 種牡馬の産駒出走数 | `100 ~ 50000` |
| 47 | 血統 | `sire_win_rate` | 種牡馬の勝率 | `0.0 ~ 0.2` (e.g. 0.09) |
| 48 | 血統 | `sire_turf_win_rate` | 種牡馬の芝勝率 | `0.0 ~ 0.2` (e.g. 0.10) |
| 49 | 血統 | `sire_dirt_win_rate` | 種牡馬のダート勝率 | `0.0 ~ 0.2` (e.g. 0.08) |
| 50 | 負担重量 | `impost` | 斤量 (正規化済み) | `48 ~ 60` (e.g. 55.0) |
| 51 | 負担重量 | `weight_ratio` | 斤量体重比 (斤量/馬体重) | `0.08 ~ 0.15` (e.g. 0.11) |
| 52 | 騎手 | `jockey_change` | 乗り替わりフラグ | `0`:継続, `1`:乗り替わり |
| 53 | 騎手 | `jockey_class_diff` | 騎手ランク差分 | `-1.0 ~ +1.0` (正規化値) |
| 54 | 変化 | `dist_change_category` | 距離変化 | `0`:同距離, `1`:短縮, `2`:延長 |
| 55 | 変化 | `interval_category` | 間隔カテゴリ | `0`:連闘, `1`:中1週... |
| 56 | 適性 | `course_win_rate` | コース別勝率 (その馬の) | `0.0 ~ 1.0` |
| 57 | 適性 | `dist_win_rate` | 距離別勝率 | `0.0 ~ 1.0` |
| 58 | 適性 | `surface_win_rate` | トラック別勝率 | `0.0 ~ 1.0` |
| 59 | スピード指数 | `avg_speed_index` | 平均スピード指数 | `30 ~ 80` (e.g. 55) |
| 60 | スピード指数 | `max_speed_index` | 最高スピード指数 | `30 ~ 90` (e.g. 65) |
| 61 | 展開 | `race_nige_count_bin` | 逃げ馬数カテゴリ | `0`:なし, `1`:1頭, `2`:複数 |
| 62 | 展開 | `race_nige_pressure_sum` | 逃げ馬プレッシャー合計 | `0.0 ~ 2.0` |
| 63 | 展開 | `is_nige_interaction` | 逃げ争い発生フラグ | `0`:なし, `1`:あり |
| 64 | 展開 | `nige_pressure_interaction` | 逃げプレッシャー相互作用 | `0.0 ~ 1.0` |
| 65 | 相対指標 | `relative_speed_index_z` | スピード指数偏差値 | `-3.0 ~ +3.0` (0=平均) |
| 66 | 相対指標 | `relative_speed_index_pct` | スピード指数パーセンタイル | `0.0 ~ 1.0` (1.0=Top) |
| 67 | 相対指標 | `relative_speed_index_diff` | スピード指数平均差 | `-20 ~ +20` |
| 68 | 相対指標 | `relative_last_3f_z` | 上がり3F偏差値 | `-3.0 ~ +3.0` |
| 69 | 相対指標 | `relative_last_3f_pct` | 上がり3Fパーセンタイル | `0.0 ~ 1.0` |
| 70 | 相対指標 | `relative_last_3f_diff` | 上がり3F平均差 | `-2.0 ~ +2.0` (秒) |
| 71 | 相対指標 | `relative_nige_rate_z` | 逃げ率偏差値 | `-3.0 ~ +3.0` |
| 72 | 相対指標 | `relative_nige_rate_pct` | 逃げ率パーセンタイル | `0.0 ~ 1.0` |
| 73 | 相対指標 | `relative_nige_rate_diff` | 逃げ率平均差 | `-0.5 ~ +0.5` |
| 74 | 相対指標 | `relative_interval_z` | 間隔偏差値 | `-3.0 ~ +3.0` |
| 75 | 相対指標 | `relative_interval_pct` | 間隔パーセンタイル | `0.0 ~ 1.0` |
| 76 | 相対指標 | `relative_interval_diff` | 間隔平均差 | `-100 ~ +100` |
| 77 | 相対指標 | `relative_impost_z` | 斤量偏差値 | `-3.0 ~ +3.0` |
| 78 | 相対指標 | `relative_impost_pct` | 斤量パーセンタイル | `0.0 ~ 1.0` |
| 79 | 相対指標 | `relative_impost_diff` | 斤量平均差 | `-5.0 ~ +5.0` |
| 80 | 騎手調教師 | `jt_run_count` | 騎手x調教師コンビ出走数 | `0 ~ 500` (e.g. 25) |
| 81 | 騎手調教師 | `jt_top3_rate_smoothed` | コンビ複勝率 (補正済) | `0.0 ~ 0.5` |
| 82 | 騎手調教師 | `jt_win_rate_smoothed` | コンビ勝率 (補正済) | `0.0 ~ 0.3` |
| 83 | 騎手調教師 | `jt_avg_rank` | コンビ平均着順 | `1.0 ~ 18.0` |
| 84 | 騎手(時系列) | `jockey_n_races_180d` | シーズン(180日)騎乗数 | `0 ~ 500` |
| 85 | 騎手(時系列) | `jockey_id_is_win_sum_180d` | シーズン勝利数 | `0 ~ 100` |
| 86 | 騎手(時系列) | `jockey_id_is_top3_sum_180d` | シーズン複勝数 | `0 ~ 300` |
| 87 | 騎手(時系列) | `jockey_n_races_365d` | 年間(365日)騎乗数 | `0 ~ 1000` |
| 88 | 騎手(時系列) | `jockey_id_is_win_sum_365d` | 年間勝利数 | `0 ~ 200` |
| 89 | 騎手(時系列) | `jockey_id_is_top3_sum_365d` | 年間複勝数 | `0 ~ 600` |
| 90 | 騎手(時系列) | `jockey_win_rate_180d` | シーズン勝率 | `0.0 ~ 0.3` |
| 91 | 騎手(時系列) | `jockey_top3_rate_180d` | シーズン複勝率 | `0.0 ~ 0.6` |
| 92 | 騎手(時系列) | `jockey_win_rate_365d` | 年間勝率 | `0.0 ~ 0.3` |
| 93 | 騎手(時系列) | `jockey_top3_rate_365d` | 年間複勝率 | `0.0 ~ 0.6` |
| 94 | 騎手(相対) | `jockey_n_races_365d_relative_z` | 年間騎乗数偏差値 | `-3.0 ~ +3.0` |
| 95 | 騎手(相対) | `jockey_win_rate_365d_relative_z` | 年間勝率偏差値 | `-3.0 ~ +3.0` |
| 96 | 騎手(相対) | `jockey_top3_rate_365d_relative_z` | 年間複勝率偏差値 | `-3.0 ~ +3.0` |
| 97 | 調教師 | `trainer_n_races_180d` | 調教師シーズン出走数 | `0 ~ 100` |
| 98 | 調教師 | `trainer_id_is_win_sum_180d` | 調教師シーズン勝利数 | `0 ~ 30` |
| 99 | 調教師 | `trainer_id_is_top3_sum_180d` | 調教師シーズン複勝数 | `0 ~ 60` |
| 100 | 調教師 | `trainer_n_races_365d` | 調教師年間出走数 | `0 ~ 500` |
| 101 | 調教師 | `trainer_id_is_win_sum_365d` | 調教師年間勝利数 | `0 ~ 50` |
| 102 | 調教師 | `trainer_id_is_top3_sum_365d` | 調教師年間複勝数 | `0 ~ 150` |
| 103 | 調教師 | `trainer_win_rate_180d` | 調教師シーズン勝率 | `0.0 ~ 0.3` |
| 104 | 調教師 | `trainer_top3_rate_180d` | 調教師シーズン複勝率 | `0.0 ~ 0.6` |
| 105 | 調教師 | `trainer_win_rate_365d` | 調教師年間勝率 | `0.0 ~ 0.3` |
| 106 | 調教師 | `trainer_top3_rate_365d` | 調教師年間複勝率 | `0.0 ~ 0.6` |
| 107 | 調教師(相対) | `trainer_n_races_365d_relative_z` | 調教師出走数偏差値 | `-3.0 ~ +3.0` |
| 108 | 調教師(相対) | `trainer_win_rate_365d_relative_z` | 調教師勝率偏差値 | `-3.0 ~ +3.0` |
| 109 | 調教師(相対) | `trainer_top3_rate_365d_relative_z` | 調教師複勝率偏差値 | `-3.0 ~ +3.0` |
| 110 | クラス成績 | `hc_n_races_365d` | 同クラス出走回数(年間) | `0 ~ 10` |
| 111 | クラス成績 | `hc_top3_rate_365d` | 同クラス複勝率(年間) | `0.0 ~ 1.0` |
| 112 | クラス成績 | `is_same_class_prev` | 前走同クラスフラグ | `0`, `1` |
| 113 | クラス成績 | `class_trend_3` | クラス昇降トレンド | `-2`(降級) ~ `+2`(昇級) |
| 114 | セグメント | `horse_small_top3_rate` | 小回りコース複勝率 | `0.0 ~ 1.0` |
| 115 | セグメント | `horse_mile_top3_rate` | マイル適性複勝率 | `0.0 ~ 1.0` |
| 116 | セグメント | `small_n_total` | 小回り経験回数 | `0 ~ 50` |
| 117 | セグメント | `mile_n_total` | マイル経験回数 | `0 ~ 50` |
| 118 | リスク | `rank_std_5` | 着順の安定度(標準偏差) | `0.0 ~ 8.0` (小さいほど安定) |
| 119 | リスク | `time_diff_std_5` | タイム差の安定度(標準偏差) | `0.0 ~ 2.0` |
| 120 | リスク | `collapse_rate_10` | 近10走大敗率 | `0.0 ~ 1.0` |
| 121 | リスク | `resurrection_flag` | 復活フラグ (長期休養明け実力馬) | `0`, `1` |
| 122 | コース適性 | `apt_rot_win_rate` | 回り(右/左)別勝率 | `0.0 ~ 1.0` |
| 123 | コース適性 | `apt_rot_top3_rate` | 回り(右/左)別複勝率 | `0.0 ~ 1.0` |
| 124 | コース適性 | `apt_rot_count` | 回り経験回数 | `0 ~ 50` |
| 125 | コース適性 | `is_first_rot` | 初回りフラグ | `0`, `1` |
| 126 | コース適性 | `apt_str_win_rate` | 直線タイプ別勝率 | `0.0 ~ 1.0` |
| 127 | コース適性 | `apt_str_top3_rate` | 直線タイプ別複勝率 | `0.0 ~ 1.0` |
| 128 | コース適性 | `apt_str_count` | 直線タイプ経験回数 | `0 ~ 50` |
| 129 | コース適性 | `is_first_str` | 初直線タイプフラグ | `0`, `1` |
| 130 | コース適性 | `apt_slp_win_rate` | 坂タイプ別勝率 | `0.0 ~ 1.0` |
| 131 | コース適性 | `apt_slp_top3_rate` | 坂タイプ別複勝率 | `0.0 ~ 1.0` |
| 132 | コース適性 | `apt_slp_count` | 坂タイプ経験回数 | `0 ~ 50` |
| 133 | コース適性 | `is_first_slp` | 初坂タイプフラグ | `0`, `1` |
| 134 | 血統適性 | `sire_apt_rot` | 種牡馬の回り適性スコア | `0.0 ~ 3.0` |
| 135 | 血統適性 | `sire_apt_str` | 種牡馬の直線適性スコア | `0.0 ~ 3.0` |
| 136 | 血統適性 | `sire_apt_slp` | 種牡馬の坂適性スコア | `0.0 ~ 3.0` |
| 137 | 騎手適性 | `jockey_apt_rot` | 騎手の回り適性スコア | `0.0 ~ 3.0` |
| 138 | 騎手適性 | `jockey_apt_str` | 騎手の直線適性スコア | `0.0 ~ 3.0` |
| 139 | 騎手適性 | `jockey_apt_slp` | 騎手の坂適性スコア | `0.0 ~ 3.0` |
| 140 | 血統適性 | `sire_apt_rot_top3` | 種牡馬の回り適性(複勝率) | `0.0 ~ 1.0` (e.g. 0.25) |
| 141 | 血統適性 | `sire_apt_str_top3` | 種牡馬の直線適性(複勝率) | `0.0 ~ 1.0` |
| 142 | 血統適性 | `sire_apt_slp_top3` | 種牡馬の坂適性(複勝率) | `0.0 ~ 1.0` |
| 143 | 騎手適性 | `jockey_apt_rot_top3` | 騎手の回り適性(複勝率) | `0.0 ~ 1.0` (e.g. 0.28) |
| 144 | 騎手適性 | `jockey_apt_str_top3` | 騎手の直線適性(複勝率) | `0.0 ~ 1.0` |
| 145 | 騎手適性 | `jockey_apt_slp_top3` | 騎手の坂適性(複勝率) | `0.0 ~ 1.0` |
| 146 | 展開予測 | `pred_runstyle` | 予測脚質 | `1`(逃), `2`(先), `3`(差), `4`(追) |
| 147 | 展開予測 | `fit_nige_short` | 逃げ馬x短距離の適合スコア | `0.0 ~ 1.0` |
| 148 | 展開予測 | `fit_inner_nige` | 逃げ馬x内枠の適合スコア | `0.0 ~ 1.0` |
| 149 | 展開予測 | `fit_sashi_long` | 差し馬x長距離の適合スコア | `0.0 ~ 1.0` |
| 150 | 展開予測 | `n_nige_in_race` | レース内の逃げ馬数 | `0 ~ 5` |
| 151 | 展開予測 | `n_other_nige` | 自分以外の逃げ馬数 | `0 ~ 4` |
| 152 | 展開予測 | `fit_nige_slow` | 逃げ馬xスローペース適合 | `0.0 ~ 1.0` |
| 153 | 展開予測 | `fit_sashi_high` | 差し馬xハイペース適合 | `0.0 ~ 1.0` |
| 154 | コンビ相性 | `jt_win_diff` | コンビ勝率期待値乖離 | `-0.2 ~ +0.2` |
| 155 | コンビ相性 | `jt_top3_diff` | コンビ複勝率期待値乖離 | `-0.3 ~ +0.3` |
| 156 | コンビ相性 | `is_first_combo` | 初コンビ結成フラグ | `0`, `1` |
| 157 | コンビ相性 | `jt_log_count` | 騎乗回数の対数 | `0.0 ~ 6.0` |
| 158 | 間隔適性 | `interval_days` | 前走間隔日数(生値) | `0 ~ 365+` |
| 159 | 間隔適性 | `interval_type_code` | 間隔タイプコード | `1`(連闘) ~ `5`(休養) |
| 160 | 間隔適性 | `tataki_count` | 叩き何戦目か | `0 ~ 10` (e.g. 2戦目) |
| 161 | 間隔適性 | `apt_int_win` | 間隔タイプ別勝率 | `0.0 ~ 1.0` |
| 162 | 間隔適性 | `apt_int_top3` | 間隔タイプ別複勝率 | `0.0 ~ 1.0` |
| 163 | 間隔適性 | `is_first_int_type` | 初間隔タイプフラグ | `0`, `1` |
| 164 | 馬体・調教 | `weight_impact_score` | 馬体重影響スコア | `0.8 ~ 1.2` (1.0=標準) |
| 165 | 馬体・調教 | `best_weight_diff` | ベスト体重との差 | `-20 ~ +20` |
| 166 | 馬体・調教 | `training_accel_score` | 調教加速スコア (旧) | `0 ~ 100` |
| 167 | 騎手戦略 | `jockey_rate_diff` | 近走調子と通算勝率の乖離 | `-0.2 ~ +0.2` |
| 168 | 騎手戦略 | `is_jockey_change` | 乗り替わり (重複) | `0`, `1` |
| 169 | 騎手戦略 | `is_jockey_return` | 騎手戻りフラグ | `0`, `1` |
| 170 | 騎手戦略 | `is_top_jockey_switch` | トップ騎手への乗り替わり | `0`, `1` |
| 171 | 展開 | `front_runner_count` | 先行馬総数 | `0 ~ 10` |
| 172 | 展開 | `race_pace_level_3f` | レースペースレベル | `20 ~ 50` (低いほどハイペース) |
| 173 | 展開 | `relative_3f_score` | 相対上がり3Fスコア | `-3.0 ~ +3.0` |
| 174 | 展開 | `is_sole_leader` | 単騎逃げ濃厚フラグ | `0`, `1` |
| 175 | 展開 | `is_high_pace_warn` | ハイペース警報 | `0`, `1` |
| 176 | 血統詳細 | `sire_course_win_rate` | 種牡馬コース勝率 | `0.0 ~ 1.0` (e.g. 0.08) |
| 177 | 血統詳細 | `sire_dist_win_rate` | 種牡馬距離勝率 | `0.0 ~ 1.0` |
| 178 | 血統詳細 | `sire_surface_win_rate` | 種牡馬芝ダート勝率 | `0.0 ~ 1.0` |
| 179 | トラックバイアス | `bias_adversity_score_mean_5` | バイアス不利克服スコア平均 | `0.0 ~ 5.0` |
| 180 | 枠順バイアス | `frame_number` | 枠番 | `1 ~ 8` |
| 181 | 枠順バイアス | `is_inner_frame` | 内枠フラグ (1-3枠) | `0`, `1` |
| 182 | 枠順バイアス | `frame_cond_win_rate` | 条件別枠番勝率 | `0.0 ~ 0.2` (e.g. 0.07) |
| 183 | 枠順バイアス | `inner_frame_win_rate` | 内枠有利コーススコア | `0.0 ~ 0.2` |
| 184 | 枠順バイアス | `frame_advantage_score` | 枠番総合有利度 | `0.0 ~ 5.0` |
| 185 | 馬体重パターン | `weight_volatility_5` | 体重変動率 (近5走) | `0.0 ~ 20.0` (e.g. 4.0) |
| 186 | 馬体重パターン | `weight_trend_3` | 体重増減トレンド | `-10 ~ +10` (e.g. +2.0) |
| 187 | 馬体重パターン | `weight_vs_avg` | 平均体重比 | `-20 ~ +20` |
| 188 | 馬体重パターン | `optimal_weight_diff` | 理想体重との差分 | `-15 ~ +15` (e.g. -2.0) |
| 189 | 休養パターン | `rest_success_rate` | 休み明け好走率 | `0.0 ~ 1.0` (e.g. 0.2) |
| 190 | 休養パターン | `optimal_rest_diff` | 理想間隔との差 | `-300 ~ +300` (e.g. 14.0) |
| 191 | 休養パターン | `tataki_effectiveness` | 叩き良化度 | `-10 ~ +10` (e.g. +2.0) |
| 192 | 休養パターン | `is_long_rest` | 長期休養フラグ (180日+) | `0`, `1` |
| 193 | コーナー戦術 | `corner_advance_score` | コーナー位置取り改善度 | `0.0 ~ 1.0` (e.g. 0.03) |
| 194 | コーナー戦術 | `avg_final_corner_pct` | 4コーナー平均通過順位(%) | `0.0 ~ 1.0` (e.g. 0.4) |
| 195 | コーナー戦術 | `corner_acceleration` | コーナー加速力 (4角-3角) | `-5.0 ~ +5.0` (e.g. 0.2) |
| 196 | 対戦成績 | `vs_rival_win_rate` | ライバル対戦勝率 | `0.0 ~ 1.0` (e.g. 0.05) |
| 197 | 対戦成績 | `vs_rival_match_count` | ライバル対戦回数 | `0 ~ 50` (e.g. 3) |
| 198 | 調教詳細 | `training_course_cat` | 調教コース種別 | `1`:坂路, `2`:W, `3`:他 |
| 199 | 調教詳細 | `training_acceleration` | 調教終い加速 (Last1F - Avg) | `0.0 ~ 2.0` (e.g. 0.5) |
| 200 | 調教詳細 | `training_intensity_score` | 調教強度スコア | `0.0 ~ 1.0` (e.g. 0.6) |
| 201 | ニックス(血統) | `nicks_count` | ニックス(配合)出走数 | `0 ~ 5000` (e.g. 500) |
| 202 | ニックス(血統) | `nicks_win_rate` | ニックス勝率 | `0.0 ~ 0.2` (e.g. 0.09) |
| 203 | ニックス(血統) | `nicks_top3_rate` | ニックス複勝率 | `0.0 ~ 0.5` (e.g. 0.26) |
| 204 | BMS(血統) | `bms_count` | 母父出走数 | `0 ~ 50000` (e.g. 7000) |
| 205 | BMS(血統) | `bms_win_rate` | 母父勝率 | `0.0 ~ 0.2` (e.g. 0.08) |
| 206 | BMS(血統) | `bms_top3_rate` | 母父複勝率 | `0.0 ~ 0.5` (e.g. 0.26) |
| 207 | レース属性 | `field_size` | 出走頭数 | `5 ~ 18` |
| 208 | レース属性 | `race_impost_std` | 斤量のばらつき(標準偏差) | `0.0 ~ 5.0` |
| 209 | レース属性 | `is_handicap_race_guess` | ハンデ戦推定フラグ | `0`, `1` |
| 210 | 負担重量(変化) | `impost_change` | 前走斤量差 | `-5.0 ~ +5.0` |
| 211 | 負担重量(変化) | `impost_change_abs` | 前走斤量差(絶対値) | `0.0 ~ 5.0` |
| 212 | 履歴トレンド | `rank_ewm_3` | 着順指数平滑移動平均(Span=3) | `1.0 ~ 18.0` |
| 213 | 履歴トレンド | `rank_ewm_5` | 着順指数平滑移動平均(Span=5) | `1.0 ~ 18.0` |
| 214 | 履歴トレンド | `rank_slope_5` | 着順トレンド傾き(5走) | `-2.0 ~ +2.0` (負=良化) |
| 215 | 履歴トレンド | `time_diff_ewm_3` | タイム差EWM(Span=3) | `0.0 ~ 5.0` |
| 216 | 履歴トレンド | `time_diff_ewm_5` | タイム差EWM(Span=5) | `0.0 ~ 5.0` |
| 217 | 履歴トレンド | `speed_index_ewm_3` | スピード指数EWM(Span=3) | `30 ~ 90` |
| 218 | 履歴トレンド | `speed_index_ewm_5` | スピード指数EWM(Span=5) | `30 ~ 90` |
| 219 | レース条件 | `weather_code` | 天候コード | `1`:晴 ~ `6`:雪 |
| 220 | レース条件 | `going_code` | 馬場状態コード | `1`:良 ~ `4`:不良 |
| 221 | レース条件 | `track_variant` | トラック速度補正値 | `-3.0 ~ +3.0` (負=速い) |
| 222 | レース構造 | `struct_early_speed_sum` | 先行力合計(レースペース圧) | `0.0 ~ 10.0` |
| 223 | レース構造 | `struct_nige_count` | 逃げ馬候補数 | `0 ~ 5` |
| 224 | レース構造 | `pace_expectation_proxy` | 展開予想プロキシ | `0.0 ~ 10.0` |
| 225 | レース構造 | `style_entropy` | 脚質エントロピー(展開の乱雑さ) | `0.0 ~ 2.0` |
| 226 | 展開適合 | `lap_fit_interaction` | 馬適性xレースペース適合度 | `0.0 ~ 1.0` |
| 227 | レーティング | `horse_elo` | Eloレーティング(出走前) | `1000 ~ 1800` |
| 228 | レーティング | `field_elo_mean` | 全体Elo平均(レースレベル) | `1000 ~ 1800` |
| 229 | レーティング | `elo_gap_to_top` | 最強馬とのレート差 | `0 ~ 500` |
| 230 | DeepLag(拡張) | `lag1_impost` | 前走斤量 | `48 ~ 60` |
| 231 | DeepLag(拡張) | `lag1_field_size` | 前走頭数 | `5 ~ 18` |
| 232 | DeepLag(拡張) | `lag1_last_3f` | 前走上がり3F | `30.0 ~ 45.0` |
| 233 | DeepLag(拡張) | `lag1_first_corner_rank` | 前走1角順位 | `1 ~ 18` |
| 234 | DeepLag(拡張) | `lag1_surface` | 前走トラック | `1`:芝, `2`:ダ |
| 235 | DeepLag(拡張) | `lag1_distance` | 前走距離 | `1000 ~ 3600` |
| 236 | DeepLag(拡張) | `lag1_going_code` | 前走馬場状態 | `1`:良 ~ `4`:不良 |
| 237 | DeepLag(拡張) | `impost_diff_prev` | 斤量増減(前走比) | `-5.0 ~ +5.0` |
| 238 | 近走成績 | `trainer_win_rate_30d` | 調教師近30日勝率 | `0.0 ~ 1.0` |
| 239 | 近走成績 | `trainer_top3_rate_30d` | 調教師近30日複勝率 | `0.0 ~ 1.0` |
| 240 | 近走成績 | `jockey_win_rate_30d` | 騎手近30日勝率 | `0.0 ~ 1.0` |
| 241 | 近走成績 | `jockey_top3_rate_30d` | 騎手近30日複勝率 | `0.0 ~ 1.0` |
| 242 | 馬イベント | `trainer_change` | 転厩(調教師変更)フラグ | `0`, `1` |
| 243 | 馬イベント | `first_run_after_gelding` | 去勢明け初戦フラグ | `0`, `1` |
| 244 | 適性平滑化 | `jockey_win_rate_smoothed` | 騎手勝率(ベイズ平滑化) | `0.0 ~ 0.5` |
| 245 | 適性平滑化 | `jockey_top3_rate_smoothed` | 騎手複勝率(ベイズ平滑化) | `0.0 ~ 0.7` |
| 246 | 適性平滑化 | `trainer_win_rate_365d_smoothed`| 調教師年間勝率(平滑化) | `0.0 ~ 0.5` |
| 247 | 相対指標(拡張)| `relative_horse_elo_z` | Eloレーティング偏差値 | `-3.0 ~ +3.0` |
| 248 | DeepLag(拡張) | `lag1_venue` | 前走の競馬場コード | `01`:札幌 ~ `10`:小倉 |
| 249 | DeepLag(拡張) | `lag2_venue` | 2走前の競馬場コード | `01`:札幌 ~ `10`:小倉 |
| 250 | DeepLag(拡張) | `lag3_venue` | 3走前の競馬場コード | `01`:札幌 ~ `10`:小倉 |
| 251 | DeepLag(拡張) | `lag2_surface` | 2走前トラック | `1`:芝, `2`:ダ |
| 252 | DeepLag(拡張) | `lag2_distance` | 2走前距離 | `1000 ~ 3600` |
| 253 | DeepLag(拡張) | `lag2_going_code` | 2走前馬場状態 | `1`:良 ~ `4`:不良 |
| 254 | DeepLag(拡張) | `lag2_impost` | 2走前斤量 | `48 ~ 60` |
| 255 | DeepLag(拡張) | `lag2_field_size` | 2走前頭数 | `5 ~ 18` |
| 256 | DeepLag(拡張) | `lag3_surface` | 3走前トラック | `1`:芝, `2`:ダ |
| 257 | DeepLag(拡張) | `lag3_distance` | 3走前距離 | `1000 ~ 3600` |
| 258 | DeepLag(拡張) | `lag3_going_code` | 3走前馬場状態 | `1`:良 ~ `4`:不良 |


