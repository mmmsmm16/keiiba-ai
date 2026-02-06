# 特徴量重要度 Top 20 詳細解説

本番モデル (`exp_t2_head_to_head`) における重要度上位20の特徴量について、その意味と作成ロジックを解説します。

## 概要
モデルは「**クラス適性**」「**近走パフォーマンス**」「**相対的な能力**」「**騎手・血統・対戦成績**」をバランスよく評価しています。特に、**「現在のクラスで通用するか？」(`hc_top3_rate_365d`)** が圧倒的な重要度（1位）を持っている点が特徴的です。

## Top 20 特徴量リスト

| Rank | 特徴量名 | カテゴリ | 意味・解説 | 作成ロジック (Technical) |
|---|---|---|---|---|
| **1** | `hc_top3_rate_365d` | Class Stats | **現クラスでの複勝率 (近1年)**<br>その馬が「今回のクラス(1勝クラスなど)」で過去1年間にどれだけ馬券に絡んだか。クラスの壁や適性を表す最強の特徴量。 | GroupBy `[horse_id, class_label]`. 直近365日の `is_top3` 累積和 / 出走数. (hc = Horse Class) |
| **2** | `lag1_rank` | History Stats | **前走着順**<br>最も直近のパフォーマンス。シンプルだが極めて強力。 | GroupBy `horse_id`. `shift(1)` |
| **3** | `venue` | Base | **競馬場 (会場)**<br>東京、中山、京都などの場所。コース形状や起伏によるバイアスが大きいため重要。 | Categorical Data (e.g., "05" for Tokyo) |
| **4** | `lag1_time_diff` | History Stats | **前走タイム差**<br>着順だけでは分からない「勝ち馬との差」。着順が悪くても僅差なら評価される。 | `time - winner_time` (Previous Race). 欠損/初出走は3.0sで埋める。 |
| **5** | `relative_last_3f_diff` | Relative Stats | **推定上がり3Fの優位性**<br>メンバー内で、その馬の「平均上がり3F」がどれだけ速いか。末脚勝負での相対的な強さ。 | `Base Last 3F` (過去5走平均) と `Race Mean` の差分取。 (Value - Mean) |
| **6** | `relative_speed_index_pct` | Relative Stats | **スピード指数の相対順位**<br>メンバー内で、その馬の「持ちスピード指数」が上位何%に位置するか。絶対能力の比較。 | `Base Speed Index` (過去5走平均) のレース内 Percentile Rank (0.0~1.0). |
| **7** | `mean_rank_5` | History Stats | **近5走平均着順**<br>短期的な安定感・好調度。 | 過去5走の `rank` 平均. (Rolling 5, Shift 1) |
| **8** | `jockey_top3_rate` | Jockey Stats | **騎手複勝率 (近1年)**<br>騎手の手腕。全コース・全クラスでの過去1年間の実績。 | GroupBy `jockey_id`. Rolling 365 days `is_top3` rate. |
| **9** | `jockey_win_rate` | Jockey Stats | **騎手勝率 (近1年)**<br>騎手の「勝ち切る力」。 | GroupBy `jockey_id`. Rolling 365 days `is_win` rate. |
| **10** | `weight` | Base | **馬体重**<br>パワーの源。特にダートや短距離で大型馬が有利になりやすい傾向を反映。 | 当日馬体重 (Raw Value). |
| **11** | `mean_time_diff_5` | History Stats | **近5走平均タイム差**<br>着順以上に実力を反映する、中期的な能力指標。 | 過去5走の `time_diff` 平均. (Rolling 5, Shift 1) |
| **12** | `age` | Base | **馬齢**<br>成長曲線や衰え。若馬(3-4歳)の成長力や高齢馬の割引。 | Raw Value (e.g., 3, 4, 5...). |
| **13** | `collapse_rate_10` | Risk Stats | **大敗率 (近10走)**<br>「10着以下」または「タイム差2.0秒以上」の負けを喫した割合。脆さ・不安定さのリスク指標。 | 過去10走での (Rank>=10 or TimeDiff>=2.0) の発生率。 |
| **14** | `weight_ratio` | Burden Stats | **斤量比重 (斤量/体重)**<br>馬格に対して背負っている斤量の重さ。小柄な馬に重い斤量は不利。 | `impost` / `weight`. |
| **15** | `vs_rival_win_rate` | Head-to-Head | **対ライバル勝率 (New!)**<br>今回の出走メンバー(ライバル)と直接対決した際の勝率。「この相手には勝ったことがある」という相性。 | 全期間の対戦マトリクスを保持し、今回の相手との過去対戦成績を集計。 (Optimization Feature) |
| **16** | `course_win_rate` | Aptitude Stats | **コース適性 (勝率)**<br>「東京」「中山」など、今回の競馬場での過去の勝率。 | `Expanding Mean` of `is_win` grouped by `[horse_id, venue]`. |
| **17** | `jockey_avg_rank` | Jockey Stats | **騎手平均着順 (近1年)**<br>騎手の安定感を示す指標。 | GroupBy `jockey_id`. Rolling 365 days average rank. |
| **18** | `interval` | History Stats | **出走間隔 (日数)**<br>休み明けか、連闘か。ローテーションの影響。 | 前走日付からの経過日数。 |
| **19** | `is_same_class_prev` | Class Stats | **前走と同クラスか？**<br>昇級戦(0)、降級戦(0)、同クラス(1)の区別。クラス慣れの指標。 | `current_class_label == prev_class_label` ? 1 : 0. |
| **20** | `grade_code` | Base | **レースグレード**<br>G1, G2, 1勝クラスなどの格。レースレベルそのもの。 | Categorical Data (Raw). |

## 考察

1.  **クラス実績の重要性 (`hc_top3_rate_365d`)**:
    「その馬が強いか」だけでなく、「**今のクラスで通用するか**」が最も重視されています。昇級直後の馬が苦戦したり、現級勝ちしている馬が安定したりする傾向を捉えています。

2.  **相対評価の採用 (`relative_*`)**:
    単なる持ちタイムではなく、「今回のメンバーの中で速いのか？」という**相対的な優位性**（偏差値や順位）が上位に来ています。これにより、メンバーレベルが異なるレースでも汎用的に能力を評価できています。

3.  **対戦成績 (`vs_rival_win_rate`) の貢献**:
    新しく実装した対戦成績が15位に入っています。これは、従来の「個人の能力値」だけでなく、「相手との力関係（直接対決）」が予測に有用であることを証明しています。

4.  **騎手要素**:
    騎手の勝率・複勝率が上位(8,9,17位)に複数ランクインしており、AI予想において騎手ファクターが非常に重要であることを示しています。
