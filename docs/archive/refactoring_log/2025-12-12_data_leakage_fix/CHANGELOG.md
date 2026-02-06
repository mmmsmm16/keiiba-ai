# データリーク修正ログ

**日付:** 2025-12-12  
**発見経緯:** 第三者コードレビュー (`reports/project_review.md`)

---

## 問題概要

### CategoryAggregator におけるデータリーク

**場所:** `src/preprocessing/category_aggregators.py`

**問題コード:**
```python
# _aggregate_basic() および _aggregate_context() 内
grouped = df.groupby(col)
history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count())
wins = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum())
```

**リークのメカニズム:**
1. データフレームは `date`, `race_id` でソートされている
2. 同一レースに同じカテゴリ（例: 同じ**調教師**の管理馬が複数頭）がある場合、それらは隣接する
3. `shift(1)` は直前の**行**を参照するため、**同一レース内**の先に処理された馬の結果が後の馬の特徴量に含まれる
4. これは「予測時点で知り得ない情報」を使用しているため、リークとなる

> [!NOTE]
> 騎手(`jockey_id`)は同一レースに1人1頭のため実質的に影響は少ないですが、
> `trainer_id`(調教師)、`sire_id`(種牡馬)、`class_level`(クラス)では同一レース内に
> 同カテゴリが複数存在し、リークが発生していました。

---

## 影響を受ける特徴量

以下のすべての特徴量がリークの影響を受けている:

### 基本集計 (`_aggregate_basic`)
| カテゴリ | 影響特徴量 |
|---------|-----------|
| `jockey_id` | `jockey_id_n_races`, `jockey_id_win_rate`, `jockey_id_top3_rate` |
| `trainer_id` | `trainer_id_n_races`, `trainer_id_win_rate`, `trainer_id_top3_rate` |
| `sire_id` | `sire_id_n_races`, `sire_id_win_rate`, `sire_id_top3_rate` |
| `class_level` | `class_level_n_races`, `class_level_win_rate`, `class_level_top3_rate` |

### 条件別集計 (`_aggregate_context`)
| 組み合わせ | 影響特徴量 |
|-----------|-----------|
| `jockey_id` × `venue` | `jockey_course_*` |
| `jockey_id` × `surface` | `jockey_surface_*` |
| `jockey_id` × `distance_cat` | `jockey_dist_*` |
| `jockey_id` × `trainer_id` | `jockey_trainer_*` |
| `trainer_id` × `venue` | `trainer_course_*` |
| `trainer_id` × `surface` | `trainer_surface_*` |
| `trainer_id` × `distance_cat` | `trainer_dist_*` |
| `sire_id` × `venue` | `sire_course_*` |
| `sire_id` × `distance_cat` | `sire_dist_*` |
| `sire_id` × `surface` | `sire_track_*` |
| `bms_id` × `distance_cat` | `bms_dist_*` |

**影響特徴量数:** 約 33+ 特徴量

---

## 修正内容

### 修正方針
race_id 単位で事前に集約し、同一レース内の情報が混入しないようにする。

### 修正前後の比較

```diff
# _aggregate_basic() の修正

- # 現在のコード (リークあり)
- grouped = df.groupby(col)
- history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count())
- wins = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum())
- top3 = grouped['rank'].transform(lambda x: (x <= 3).astype(int).shift(1).expanding().sum())

+ # 修正後 (リークなし)
+ # Step 1: race_id単位で集約 (同レース内の重複を排除)
+ race_stats = df.groupby(['race_id', col]).agg({
+     'rank': ['min', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()],
+     'date': 'first'
+ }).reset_index()
+ race_stats.columns = ['race_id', col, 'best_rank', 'wins', 'top3', 'date']
+ race_stats = race_stats.sort_values('date')
+ 
+ # Step 2: カテゴリごとにshift(1) + expanding (レース単位なのでリークなし)
+ grouped = race_stats.groupby(col)
+ race_stats['cum_races'] = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count())
+ ...
+ 
+ # Step 3: 元のDataFrameにマージ
+ df = df.merge(race_stats[...], on=['race_id', col], how='left')
```

---

## 修正ファイル一覧

| ファイル | 変更内容 | ステータス |
|---------|---------|----------|
| `src/preprocessing/category_aggregators.py` | リークロジック修正 | ✅ 完了 |
| `src/preprocessing/test_category_aggregators.py` | 新規テスト作成 | ✅ 完了 |
| `docker/python/requirements.txt` | NumPy バージョン固定 | ✅ 完了 |

---

## 次のステップ

1. [x] `category_aggregators.py` の修正実装
2. [x] テストコード作成
3. [x] テスト実行・確認 (6テスト全てパス)
4. [x] データセット再生成 (`lgbm_datasets_v10_leakfix.pkl`)
5. [ ] モデル再学習
6. [ ] 性能比較 (修正前 vs 修正後)
