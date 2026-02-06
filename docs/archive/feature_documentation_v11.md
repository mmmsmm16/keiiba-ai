# 競馬AI: 学習データ特徴量詳細ドキュメント (v11)

**総特徴量数**: 約170個（欠損フラグ追加により増加）

**データバージョン**: v11 (2025-12-15更新)

**主な変更点 (v10_leakfix → v11)**:
- A1: rank系の0埋め廃止 → ニュートラル値(8.0)補完 + 欠損フラグ追加
- A2: 番兵値（odds=0, weight=999等）のNaN化・適正値補完
- A3: unknown/0 の集計汚染監視ログ追加
- A4: trend_* の運用モード切替（事前予測/逐次更新）
- A6: embedding 無効化フラグ追加
- B2: weight本体 + weight_is_missing フラグ追加
- B3: total_prize全0時の警告ログ追加

---

## 1. 基本情報 (レースメタデータ) - 19個

レースの基本的なメタデータです。

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `race_number` | レース番号 | 生データより |
| 2 | `distance` | レース距離(m) | 生データより |
| 3 | `frame_number` | 枠番 | 生データより |
| 4 | `horse_number` | 馬番 | 生データより |
| 5 | `age` | 馬齢 | 生データより |
| 6 | `impost` | 斤量(kg) | 生データより |
| 7 | `weight` | [v11 B2] 馬体重(kg) | 生データより、欠損→中央値(約470) |
| 8 | `weight_is_missing` | [v11 B2] 馬体重欠損フラグ | 未計量の場合=1 |
| 9 | `weight_diff` | 馬体重増減(kg) | 生データより |
| 10 | `sex_num` | 性別(数値化) | 牡=1, 牝=2, セ=3 |
| 11 | `weather_num` | 天候(数値化) | 晴=1, 曇=2, 雨=3, 小雨=4, 雪=5 |
| 12 | `surface_num` | 馬場種別(数値化) | 芝=1, ダート=2, 障害=3 |
| 13 | `state_num` | 馬場状態(数値化) | 良=1, 稀重=2, 重=3, 不良=4 |
| 14 | `year` | 年 | `date.dt.year` |
| 15 | `month` | 月 | `date.dt.month` |
| 16 | `day` | 日 | `date.dt.day` |
| 17 | `weekday` | 曜日 | `date.dt.weekday` (0=月曜, 6=日曜) |
| 18 | `class_level` | クラスレベル | G1=9, G2=8, G3=7, OP=6, 3勝=5, 2勝=4, 1勝=3, 未勝利=2, 新馬=1 |
| 19 | `n_horses` | 出走頭数 | レース内の馬数 |

---

## 2. 過去走特徴 (Lag/Rolling) - 12個 + 欠損フラグ

馬の過去の成績を時系列で集計した特徴量です。**全てshift(1)を使用し、当該レースより前のデータのみを使用**。

### [v11変更] 欠損補完方針

| 対象カラム | v10以前 | v11 |
|-----------|--------|-----|
| `lag1_rank`, `mean_rank_5`, `mean_rank_all` | 0埋め | **8.0（中位）で補完** + 欠損フラグ |
| `mean_last_3f_5`, `lag1_last_3f` | 0埋め | **35.0（平均値）で補完** + 欠損フラグ |

### Lag特徴 (前走)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 1 | `lag1_rank` | 前走着順 | `groupby('horse_id')['rank'].shift(1)`、欠損→8.0 |
| 2 | `lag1_rank_is_missing` | [v11] 前走着順欠損フラグ | 初出走の場合=1 |
| 3 | `lag1_last_3f_rank` | 前走上がり3F順位 | 前走last_3fから順位化 |
| 4 | `lag1_last_3f_is_missing` | [v11] 上がり3F欠損フラグ | 初出走の場合=1 |

### Rolling特徴 (近走平均)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 5 | `mean_rank_5` | 近5走平均着順 | `shift(1).rolling(5).mean()`、欠損→8.0 |
| 6 | `mean_rank_5_is_missing` | [v11] 近5走欠損フラグ | 5走未満の場合=1 |
| 7 | `mean_last_3f_5` | 近5走平均上がり3F | `shift(1).rolling(5).mean()`、欠損→35.0 |
| 8 | `mean_last_3f_5_is_missing` | [v11] 近5走上がり欠損フラグ | 5走未満の場合=1 |

### Expanding特徴 (通算成績)

| # | 特徴量名 | 説明 | 計算方法 |
|---|----------|------|----------|
| 9 | `total_races` | 通算出走数 | `cumcount()` (当該レース除く) |
| 10 | `mean_rank_all` | 通算平均着順 | `shift(1).expanding().mean()`、欠損→8.0 |
| 11 | `mean_rank_all_is_missing` | [v11] 通算着順欠損フラグ | 初出走の場合=1 |
| 12 | `wins_all` | 通算勝利数 | `(rank==1).shift(1).expanding().sum()`、欠損→0 |
| 13 | `win_rate_all` | 通算勝率 | `wins_all / total_races`、欠損→0 |
| 14 | `total_prize` | 通算獲得賞金 | `shift(1).expanding().sum()` |

---

## 3. カテゴリ統計 (基本) - 16個

騎手・調教師・種牡馬・クラスレベルごとの集計統計。**レース単位で集約してからshift(1)で累積計算**し、同一レース内のデータリークを防止。

### [v11変更] unknown監視

- ID欠損は全て `'unknown'` に統一
- `*_n_races` の上位にunknownが5%以上占める場合は警告ログ出力

| # | 特徴量名 | 説明 |
|---|----------|------|
| 1-3 | `jockey_id_n_races/win_rate/top3_rate` | 騎手: 出走数/勝率/複勝率 |
| 4-6 | `trainer_id_n_races/win_rate/top3_rate` | 調教師: 出走数/勝率/複勝率 |
| 7-9 | `sire_id_n_races/win_rate/top3_rate` | 種牡馬: 出走数/勝率/複勝率 |
| 10-12 | `class_level_n_races/win_rate/top3_rate` | クラスレベル: 出走数/勝率/複勝率 |
| 13-16 | `*_deviation`, `*_race_rank` | 偏差・レース内順位 |

---

## 9. リアルタイム特徴 - 5個

### [v11変更] 運用モード切替

**当日のそれまでのレース結果**から計算されるトラックバイアス指標。

| モード | 設定 | 説明 |
|--------|------|------|
| 事前予測モード（デフォルト） | `--use_realtime` なし | ニュートラル値で埋める（朝一括予測向け） |
| 逐次更新モード | `--use_realtime` あり | 前レース結果から計算（リアルタイム予測向け） |

| # | 特徴量名 | 説明 | デフォルト値 |
|---|----------|------|-------------|
| 1 | `trend_win_inner_rate` | 内枠勝率(当日) | 0.25 |
| 2 | `trend_win_mid_rate` | 中枠勝率(当日) | 0.50 |
| 3 | `trend_win_outer_rate` | 外枠勝率(当日) | 0.25 |
| 4 | `trend_win_front_rate` | 先行勝率(当日) | 0.20 |
| 5 | `trend_win_fav_rate` | 人気馬勝率(当日) | 0.33 |

---

## 10. 埋め込み特徴 (Embedding) - 32個

### [v11変更] 無効化オプション

`--no_embedding` フラグで無効化可能（リーク対策）。

Entity Embedding: 馬・騎手・調教師・種牡馬のID情報を低次元ベクトル(各8次元)に変換。

- `horse_id_emb_0` ~ `horse_id_emb_7`
- `jockey_id_emb_0` ~ `jockey_id_emb_7`
- `trainer_id_emb_0` ~ `trainer_id_emb_7`
- `sire_id_emb_0` ~ `sire_id_emb_7`

---

## 異常値・番兵値処理 (v11)

### [A2] 番兵値のNaN化ルール

| カラム | 番兵値 | 処理 |
|--------|--------|------|
| `odds` | 0 | NaN化 |
| `weight` | 0, >=999, <300 | 中央値（約470）で補完 |
| `weight_diff` | <=-99, >=999 | 0で補完 |
| `impost` | 0 | 平均値（約55）で補完 |
| `frame_number`, `horse_number` | 0 | NaN化 |
| `rank` | 0 | 削除（取消・中止扱い） |

---

## パイプライン実行オプション (v11)

```bash
# 事前予測モード（デフォルト）
python src/preprocessing/run_preprocessing.py --suffix _v11

# 逐次更新モード（trend_*を計算）
python src/preprocessing/run_preprocessing.py --suffix _v11_realtime --use_realtime

# embedding無効化
python src/preprocessing/run_preprocessing.py --suffix _v11_no_emb --no_embedding
```

---

**更新日時**: 2025-12-15  
**データバージョン**: v11  
**前バージョン**: v10_leakfix（docs/archive/ に移動）
