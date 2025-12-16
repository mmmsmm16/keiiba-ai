# 競馬AI: 学習データ特徴量詳細ドキュメント (v10_leakfix)

**総特徴量数**: 165個

**データリーク対策**: 全特徴量はリーク対策済み。過去の情報のみを使用し、未来の情報（同レース内の他馬の結果など）は含まれません。

---

## 1. 基本情報 (レースメタデータ) - 17個

レースの基本的なメタデータです。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `race_number` | レース番号 | 生データより |
| 2 | `distance` | レース距離(m) | 生データより |
| 3 | `frame_number` | 枠番 | 生データより |
| 4 | `horse_number` | 馬番 | 生データより |
| 5 | `age` | 馬齢 | 生データより |
| 6 | `impost` | 斤量(kg) | 生データより |
| 7 | `weight_diff` | 馬体重増減(kg) | 生データより |
| 8 | `sex_num` | 性別(数値化) | 牡=1, 牝=2, セ=3 |
| 9 | `weather_num` | 天候(数値化) | 晴=1, 曇=2, 雨=3, 小雨=4, 雪=5 |
| 10 | `surface_num` | 馬場種別(数値化) | 芝=1, ダート=2, 障害=3 |
| 11 | `state_num` | 馬場状態(数値化) | 良=1, 稍重=2, 重=3, 不良=4 |
| 12 | `year` | 年 | `date.dt.year` |
| 13 | `month` | 月 | `date.dt.month` |
| 14 | `day` | 日 | `date.dt.day` |
| 15 | `weekday` | 曜日 | `date.dt.weekday` (0=月曜, 6=日曜) |
| 16 | `class_level` | クラスレベル | G1=9, G2=8, G3=7, OP=6, 3勝=5, 2勝=4, 1勝=3, 未勝利=2, 新馬=1 |
| 17 | `n_horses` | 出走頭数 | レース内の馬数 |

---

## 2. 過去走特徴 (Lag/Rolling) - 12個

馬の過去の成績を時系列で集計した特徴量です。**全てshift(1)を使用し、当該レースより前のデータのみを使用**。

### Lag特徴 (前走)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `lag1_rank` | 前走着順 | `groupby('horse_id')['rank'].shift(1)` |
| 2 | `lag1_last_3f_rank` | 前走上がり3F順位 | 前走last_3fから順位化 |
| 3 | `lag1_race_member_strength` | 前走レースメンバー強度 | 前走時の出走メンバーの平均強度 |
| 4 | `lag1_performance_value` | 前走パフォーマンス値 | 前走の相対的な走破能力 |
| 5 | `lag1_n_horses` | 前走出走頭数 | 前走のレース規模 |

### Rolling特徴 (近走平均)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 6 | `mean_rank_5` | 近5走平均着順 | `shift(1).rolling(5).mean()` |
| 7 | `mean_last_3f_5` | 近5走平均上がり3F | `shift(1).rolling(5).mean()` |

### Expanding特徴 (通算成績)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 8 | `total_races` | 通算出走数 | `cumcount()` (当該レース除く) |
| 9 | `mean_rank_all` | 通算平均着順 | `shift(1).expanding().mean()` |
| 10 | `wins_all` | 通算勝利数 | `(rank==1).shift(1).expanding().sum()` |
| 11 | `win_rate_all` | 通算勝率 | `wins_all / total_races` |
| 12 | `total_prize` | 通算獲得賞金 | `shift(1).expanding().sum()` |

---

## 3. カテゴリ統計 (基本) - 16個

騎手・調教師・種牡馬・クラスレベルごとの集計統計。**レース単位で集約してからshift(1)で累積計算**し、同一レース内のデータリークを防止。

### 基本カテゴリ統計

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1-3 | `jockey_id_n_races/win_rate/top3_rate` | 騎手: 出走数/勝率/複勝率 | レース単位集約→shift(1)→expanding統計 |
| 4-6 | `trainer_id_n_races/win_rate/top3_rate` | 調教師: 出走数/勝率/複勝率 | 同上 |
| 7-9 | `sire_id_n_races/win_rate/top3_rate` | 種牡馬: 出走数/勝率/複勝率 | 同上 |
| 10-12 | `class_level_n_races/win_rate/top3_rate` | クラスレベル: 出走数/勝率/複勝率 | 同上 |

### 偏差・ランク特徴

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 13 | `jockey_id_win_rate_deviation` | 騎手勝率の偏差 | レース内での騎手勝率の標準化 |
| 14 | `trainer_id_win_rate_deviation` | 調教師勝率の偏差 | レース内での調教師勝率の標準化 |
| 15 | `jockey_id_win_rate_race_rank` | 騎手勝率のレース内順位 | レース内での騎手勝率ランク |
| 16 | `trainer_id_win_rate_race_rank` | 調教師勝率のレース内順位 | レース内での調教師勝率ランク |

---

## 4. コンテキスト統計 (組み合わせ) - 29個

複数条件の組み合わせによる集計統計。**全てリーク対策済み**。

### 騎手コンテキスト (8個)

- `jockey_course_*`: 騎手×コース(venue+surface)
- `jockey_surface_*`: 騎手×馬場種別
- `jockey_dist_*`: 騎手×距離区分
- `jockey_distance_winrate`: 騎手×距離カテゴリ勝率

各3指標: `n_races`, `win_rate`, `top3_rate`

### 種牡馬コンテキスト (9個)

- `sire_course_*`: 種牡馬×コース
- `sire_dist_*`: 種牡馬×距離区分
- `sire_track_*`: 種牡馬×トラック種別

### 調教師コンテキスト (9個)

- `trainer_course_*`: 調教師×コース
- `trainer_surface_*`: 調教師×馬場種別
- `trainer_dist_*`: 調教師×距離区分

### 組み合わせ統計 (3個)

- `jockey_trainer_*` (3個): 騎手×調教師コンビネーション
- `trainer_jockey_count`: 調教師が起用する騎手の多様性
- `frame_surface_winrate`: 枠番×馬場種別勝率

---

## 5. 血統特徴 - 8個

種牡馬・母父(BMS)の累積統計。**expanding統計でリーク防止**。

### 種牡馬統計 (4個)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `sire_avg_rank` | 種牡馬産駒平均着順 | `shift(1).expanding().mean()` |
| 2 | `sire_win_rate` | 種牡馬産駒勝率 | `(rank==1).shift(1).expanding().mean()` |
| 3 | `sire_roi_rate` | 種牡馬産駒複勝率 | `(rank<=3).shift(1).expanding().mean()` |
| 4 | `sire_count` | 種牡馬産駒出走数 | `shift(1).expanding().count()` |

### 母父(BMS)統計 (4個)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 5 | `bms_avg_rank` | 母父産駒平均着順 | 同上 |
| 6 | `bms_win_rate` | 母父産駒勝率 | 同上 |
| 7 | `bms_roi_rate` | 母父産駒複勝率 | 同上 |
| 8 | `bms_count` | 母父産駒出走数 | 同上 |

---

## 6. 展開・ペース特徴 - 11個

レース展開やペースに関する特徴量。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `nige_rate` | 逃げ傾向 | 過去の通過順位から逃げ馬を判定: `passing_rank=='01-01-01'` |
| 2 | `interval` | 前走からの休養日数 | `(date - prev_date).days` |
| 3 | `momentum_slope` | モメンタム(着順変化率) | lag1~lag5の線形回帰傾き (負=上昇傾向) |
| 4 | `rest_score` | 休養スコア | `interval`を0~1にスケール(最適14~28日) |
| 5 | `race_avg_nige_rate` | レース平均逃げ率 | レース内の平均逃げ傾向 |
| 6 | `race_nige_horse_count` | レース内逃げ馬頭数 | 逃げ傾向が強い馬の数 |
| 7 | `race_nige_bias` | 逃げ馬バイアス | 逃げ馬が有利かどうかの指標 |
| 8 | `race_pace_cat` | レースペース予測 | 0=スロー, 1=平均, 2=ハイペース |
| 9 | `race_avg_prize` | レース平均獲得賞金 | 出走メンバーの平均賞金額 |
| 10 | `race_avg_age` | レース平均馬齢 | 出走メンバーの平均年齢 |
| 11 | `horse_pace_disadv_rate` | ペース不利率 | 過去のペース不利を受けた割合 |

---

## 7. 不利検出特徴 - 5個

レース中の不利を検出・累積した特徴量。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `horse_slow_start_rate` | 出遅れ回復率 | 前半後方→追い込みで好走したパターン |
| 2 | `horse_wide_run_rate` | 外回し率 | 外枠や外を回ったレースの割合 |
| 3 | `horse_track_bias_rate` | 馬場不利率 | 当日の不利な枠・脚質で走った割合 |
| 4 | `prev_disadvantage_score` | 前走不利スコア | 前走での不利要因の合計 |
| 5 | `avg_disadvantage_score_3races` | 近3走平均不利スコア | 直近3走の平均不利スコア |

---

## 8. 相対的特徴 - 6個

レース内での相対的な位置を示す特徴量。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `weight_deviation` | 馬体重の偏差 | レース内で標準化: `(x - mean) / std` |
| 2 | `age_deviation` | 馬齢の偏差 | 同上 |
| 3 | `impost_deviation` | 斤量の偏差 | 同上 |
| 4 | `weight_relative` | 相対馬体重 | レース内最大値との比: `x / max` |
| 5 | `age_relative` | 相対馬齢 | 同上 |
| 6 | `impost_relative` | 相対斤量 | 同上 |

---

## 9. リアルタイム特徴 - 5個

**当日のそれまでのレース結果**から計算されるトラックバイアス指標。**レース番号でshift**しリーク防止。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `trend_win_inner_rate` | 内枠勝率(当日) | 同日の前レースでの内枠(1-3)勝率 |
| 2 | `trend_win_mid_rate` | 中枠勝率(当日) | 同日の前レースでの中枠(4-6)勝率 |
| 3 | `trend_win_outer_rate` | 外枠勝率(当日) | 同日の前レースでの外枠(7-8)勝率 |
| 4 | `trend_win_front_rate` | 先行勝率(当日) | 同日の前レースでの先行馬勝率 |
| 5 | `trend_win_fav_rate` | 人気馬勝率(当日) | 同日の前レースでの1-3番人気勝率 |

**計算方法**: 同日・同会場のレース番号 < 当該レース番号の結果を集計。第1レースは平均値で埋める。

---

## 10. 埋め込み特徴 (Embedding) - 32個

Entity Embedding: 馬・騎手・調教師・種牡馬のID情報を低次元ベクトル(各8次元)に変換。

### 構成 (各8次元)

- `horse_id_emb_0` ~ `horse_id_emb_7`: 馬ID埋め込み
- `jockey_id_emb_0` ~ `jockey_id_emb_7`: 騎手ID埋め込み
- `trainer_id_emb_0` ~ `trainer_id_emb_7`: 調教師ID埋め込み
- `sire_id_emb_0` ~ `sire_id_emb_7`: 種牡馬ID埋め込み

**生成方法**: 
1. 各IDをLabelEncodingで整数化
2. Embedding層(8次元)で学習済みベクトルを取得
3. PCA等で次元削減した表現を使用

**効果**: 高次元のカテゴリID(数万種類)を低次元の連続値に圧縮し、類似馬・類似騎手を表現可能に。

---

## 11. 経験値特徴 - 9個

コース・距離・条件での経験を示す特徴量。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `course_experience` | コース経験回数 | `groupby(['horse_id', 'venue', 'surface']).cumcount()` |
| 2 | `course_best_rank` | コース最高着順 | `shift(1).expanding().min()` per course |
| 3 | `distance_experience` | 距離経験回数 | `groupby(['horse_id', 'distance_cat']).cumcount()` |
| 4 | `distance_best_rank` | 距離最高着順 | `shift(1).expanding().min()` per distance |
| 5 | `first_distance_cat` | 距離カテゴリ初挑戦 | 当該距離カテゴリ初出走=1 |
| 6 | `first_turf` | 初芝フラグ | 芝初出走=1 |
| 7 | `first_dirt` | 初ダートフラグ | ダート初出走=1 |
| 8 | `jockey_change_flag` | 騎手乗り替わり | 前走と騎手が異なる=1 |
| 9 | `is_career_high_impost` | 最高斤量挑戦 | 今回斤量 > 過去最高斤量=1 |

**距離カテゴリ**: 短距離(<1400m), マイル(1400-1800m), 中距離(1800-2200m), 長距離(≥2200m)

---

## 12. レースレベル特徴 - 2個

レース全体の質と馬の相対的な強さを示す指標。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `race_member_strength` | レースメンバー強度 | 出走メンバー全体の平均能力値(平均着順の逆数等) |
| 2 | `relative_strength` | 相対強度 | 自分の能力値 - レース平均強度 |

---

## 13. その他 - 13個

上記カテゴリに分類されない特徴量。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1-3 | `sire_track_n_races/win_rate/top3_rate` | 種牡馬×トラック統計 | expanding統計 |
| 4 | `is_long_break` | 長期休養フラグ | interval > 180日 = 1 |
| 5 | `jockey_recent_win_rate` | 騎手直近勝率 | 直近30日間の勝率 |
| 6 | `trainer_recent_win_rate` | 調教師直近勝率 | 直近30日間の勝率 |
| 7 | `frame_zone` | 枠ゾーン | 内枠=0, 中枠=1, 外枠=2 |
| 8 | `distance_category` | 距離カテゴリ | 短距離/マイル/中距離/長距離 |
| 9 | `jockey_distance_winrate` | 騎手×距離勝率 | カスタム集計 |
| 10 | `relative_popularity_rank` | 相対人気順位 | レース内での人気順位の正規化 |
| 11 | `estimated_place_rate` | 推定複勝率 | オッズから逆算した確率 |
| 12 | `class_gap` | クラスギャップ | 前走クラス - 今回クラス |
| 13 | `is_class_up` | 昇級フラグ | 前走より高いクラス=1 |

---

## データリーク対策の詳細

### 主要な対策

1. **時系列順処理**: 全ての集計は`date`でソート後に実行
2. **shift(1)の徹底**: 過去走統計は必ず`shift(1)`で当該レースを除外
3. **レース単位集約**: カテゴリ統計は`race_id`単位で集約してから累積計算
4. **expanding統計**: 累積統計は`expanding()`で全期間を使用
5. **リアルタイム特徴**: レース番号でフィルタリング

### 検証済み項目

- ✅ 同一レース内の他馬の結果を参照していない
- ✅ 未来の情報(レース後のデータ)を使用していない
- ✅ テストセット期間のデータで学習していない

---

## 特徴量の重要度順 (参考)

実際のモデルでは、以下のような特徴量が高重要度とされています:

1. **過去走成績**: `mean_rank_5`, `lag1_rank`, `win_rate_all`
2. **カテゴリ統計**: `jockey_id_win_rate`, `trainer_id_win_rate`
3. **埋め込み**: `horse_id_emb_*`, `jockey_id_emb_*`
4. **コンテキスト**: `jockey_course_win_rate`, `sire_dist_win_rate`
5. **展開予測**: `race_pace_cat`, `nige_rate`, `momentum_slope`

---

**生成日時**: 2025-12-15  
**データバージョン**: v10_leakfix  
**学習期間**: 1954年～2024年12月  
**総レコード数**: 2,816,319行
