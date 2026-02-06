# モデル特徴量一覧 (V16.1 - Time Difference Features)

| カテゴリ | 特徴量名 (カラム名) | 説明 | 計算方法/備考 |
| :--- | :--- | :--- | :--- |
| **レース属性** | `venue` | 開催場所 | 札幌, 東京, 中山 など |
| | `race_number` | レース番号 | 1R ~ 12R |
| | `distance` | 距離 (m) | 数値データ (例: 1600, 2400) |
| | `surface` | コース種別 | 芝, ダート, 障害 |
| | `weather` | 天候 | 晴, 曇, 雨, 小雨, 雪, 小雪 |
| | `state` | 馬場状態 | 良, 稍重, 重, 不良 |
| | `course_id` | コースID | 場所・距離・種別の組み合わせID (例: 101) |
| | `direction` | 回り | 右, 左, 直, 他 |
| | `day`, `weekday` | 開催日・曜日 | 日付情報より抽出 |
| **馬・騎手・調教師** | `horse_id` | 馬ID | JRA公式ID |
| | `jockey_id` | 騎手ID | JRA公式ID |
| | `trainer_id` | 調教師ID | JRA公式ID |
| | `sire_id` | 父馬ID | 血統情報 |
| | `mare_id` | 母馬ID | 血統情報 |
| | `sex` | 性別 | 牡, 牝, セン |
| | `age` | 年齢 | レース時点の年齢 |
| | `weight` | 馬体重 | レース当日の馬体重 (欠損時は中央値補完 + `weight_is_missing`フラグ) |
| | `weight_diff` | 馬体重増減 | 前走からの増減 (数値) |
| | `impost` | 斤量 | 負担重量 |
| | `frame_number` | 枠番 | 1 ~ 8 |
| | `horse_number` | 馬番 | 1 ~ 18 |
| **過去成績 (Lag)** | `lag1_rank` | 前走着順 | 前走の着順 (例: 1, 2, ..., 18. 初出走等は欠損扱い) |
| | `lag1_rank_norm` | 前走正規化着順 | `lag1_rank / lag1_n_horses` (0.0~1.0, 小さいほど良い) |
| | `lag1_last_3f` | 前走上がり3F | 前走の上がりタイム (秒) |
| | `lag1_odds` | 前走オッズ | 前走の単勝オッズ |
| | `lag1_popularity` | 前走人気 | 前走の単勝人気順 |
| | `lag1_class_level` | 前走クラス | 前走のクラスレベル |
| | `lag1_performance_value` | 前走パフォーマンス値 | 独自スコア (速度・着順等を統合) |
| | `lag1_time_diff` | 前走着差 | 前走の1着馬とのタイム差 (秒) |
| **集計統計量 (Past Stats)** | `mean_rank_5` | 近5走平均着順 | 直近5レースの平均着順 |
| | `mean_last_3f_5` | 近5走平均上がり | 直近5レースの平均上がり3Fタイム |
| | `mean_time_diff_5` | 近5走平均着差 | 直近5レースの平均タイム差 |
| | `wins_all` | 通算勝利数 | 過去の総勝利数 |
| | `win_rate_all` | 通算勝率 | `wins_all / total_races` |
| | `total_prize` | 獲得賞金 | 過去の総獲得賞金 (円) |
| | `n_horses` | 前走頭数 | 前走の出走頭数 |
| **関係者実績 (Expanding)** | `jockey_id_win_rate` | 騎手勝率 | 騎手ごとの通算勝率 (ヒストリカル Expanding Mean) |
| | `trainer_id_win_rate` | 調教師勝率 | 調教師ごとの通算勝率 |
| | `sire_id_win_rate` | 種牡馬勝率 | 種牡馬ごとの産駒勝率 |
| | `jockey_course_win_rate` | 騎手×コース勝率 | 特定コースにおける騎手の勝率 |
| | `sire_course_win_rate` | 種牡馬×コース勝率 | 特定コースにおける種牡馬の勝率 |
| | `jockey_distance_winrate` | 騎手×距離区分勝率 | 距離区分(短距離/マイル/中距離/長距離)ごとの騎手勝率 |
| | `frame_surface_winrate` | 枠番×馬場勝率 | 枠番と馬場(芝/ダート)の組み合わせ勝率 |
| **高度な指標 (Advanced)** | `lag1_time_index` | スピード指数 (Lag1) | 走破タイムの偏差値スコア (前走)。コース・距離・馬場で補正。 |
| | `lag1_last_3f_index` | 上がり指数 (Lag1) | 上がり3Fの偏差値スコア (前走)。 |
| | `class_gap` | クラス昇降 | 今回クラス - 前走クラス (正=昇級, 負=降級) |
| | `is_class_up` | 昇級初戦フラグ | `class_gap > 0` の場合 1 |
| | `rest_score` | 休養スコア | 出走間隔の最適性 (中2-4週=高スコア, 連闘/長期=低スコア) |
| | `momentum_slope` | 着順上昇度 | 近5走の着順トレンドの傾き (負=上昇傾向) |
| | `nige_rate` | 逃げ率 | 過去に脚質「逃げ」を選択した割合 |
| | `race_nige_bias` | レース内逃げ率 | 出走馬全体の逃げ馬の割合 (展開予測用) |
| | `race_avg_nige_rate` | 平均逃げ率 | 出走馬の平均逃げ率 |
| | `disadvantage_score` | 不利スコア | 出遅れ、大外枠など不利条件の蓄積スコア (判定器による) |
| | `relative_popularity_rank` | 相対的評価 | メンバー内での前走人気順位の相対値 |
| | `estimated_place_rate` | 推定連対率 | `1 / (mean_rank + 1)` |
| **オッズ変動 (Odds)** | `log_odds_t10` | T-10単勝対数オッズ | 締切10分前の単勝オッズの対数 (`log(odds)`) |
| | `dlog_odds_t60_t10` | 長期トレンド | `log(T10) - log(T60)` (50分間の変動幅) |
| | `dlog_odds_t30_t10` | 短期トレンド | `log(T10) - log(T30)` (20分間の変動幅) |
| | `odds_drop_rate_t60_t10` | オッズ下落率 | `Odds(T10) / Odds(T60)` (1.0未満なら支持が集まっている) |
| | `rank_change_t60_t10` | 人気順位変動 | `Rank(T10) - Rank(T60)` (負なら人気上昇) |
| | `odds_volatility` | オッズ変動ボラティリティ | T60, T30, T10 時点間のオッズ標準偏差 |
| **埋め込み (Embeddings)** | `*_emb_0` ~ `*_emb_7` | エンティティ埋め込み | 馬、騎手、調教師、種牡馬IDをLightGBMまたはNNで学習した潜在ベクトル (8次元) |

## 除外された特徴量 (リーク防止)
以下の特徴量は「未来情報」であるため、現在のモデルでは**使用していません**。
- `time` (走破タイム)
- `agari`, `last_3f` (上がり3F)
- `rank`, `着順` (着順)
- `odds`, `logit_final_odds` (確定単勝オッズ)
- `ninki`, `popularity` (確定人気順)
- `pass_*`, `passing_rank` (通過順)
