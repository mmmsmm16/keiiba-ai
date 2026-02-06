# タスク: Batch 1 特徴量実装 (マイニング & 馬場適性)

## 1. 準備・設計
- [x] 実装計画書 (`implementation_plan.md`) の作成 (Batch 1) <!-- id: 0 -->
- [x] `jvd_wf` スキーマの確認（クッション値・含水率の有無） <!-- id: 1 -->

## 2. 実装: データロード
- [x] `src/preprocessing/loader.py` の修正: `jvd_se` から `mining_kubun`, `yoso_soha_time`, `yoso_gosa_plus`, `yoso_gosa_minus` を読み込む <!-- id: 2 -->
- [x] `src/preprocessing/loader.py` の修正: `jvd_wf` は今回対象外（詳細情報なしのため） <!-- id: 3 -->

## 3. 実装: 特徴量エンジニアリング
- [x] `src/preprocessing/features/mining_features.py` の新規作成 <!-- id: 4 -->
    - [x] マイニング区分のカテゴリ変数化
    - [x] 予想タイム乖離・予想順位乖離の算出
- [x] `src/preprocessing/features/track_aptitude.py` の新規作成 <!-- id: 5 -->
    - [x] `horse_going_win_rate` (馬場状態別勝率) の実装
    - [x] `horse_going_top3_rate` (馬場状態別複勝率) の実装
- [x] `src/preprocessing/feature_pipeline.py` へのジェネレータ登録 <!-- id: 6 -->

## 4. 更新・実行
- [x] `docs/feature_definitions.md` (または v4) の更新 <!-- id: 7 -->
- [x] 前処理の実行: `preprocessed_data_v12.parquet` の生成 (高速化再実行中) <!-- id: 8 -->
- [x] 特徴量分布の確認 / Nullチェック <!-- id: 9 -->

## 5. 学習・評価
- [x] 新特徴量を用いた LambdaRank モデルの学習 (`exp_lambdarank_v12_batch1`) <!-- id: 10 -->
- [x] ベースライン (v11/v3) との精度・ROI比較 <!-- id: 11 -->

## 6. ドキュメント更新
- [x] `walkthrough.md` の作成 (実施内容・結果の記録) <!-- id: 12 -->
- [x] (New) `docs/walkthrough_batch1.md` created.
- [x] `comprehensive_feature_list.md` への結果反映 <!-- id: 13 -->

# タスク: Batch 2 特徴量実装 (騎手・調教師属性 & 輸送ロジスティクス)

## 1. 準備・設計
- [ ] 実装計画書 (`implementation_plan.md`) の更新 (Batch 2) <!-- id: 14 -->

## 2. 実装: データロード
- [ ] `src/preprocessing/loader.py` の修正: `jvd_ks`, `jvd_ch` のロード処理追加 <!-- id: 15 -->
- [ ] マスタデータとの結合ロジック実装 (`Jockey/Trainer` IDベース) <!-- id: 16 -->

## 3. 実装: 特徴量エンジニアリング
- [ ] `src/preprocessing/features/attribute_features.py` (仮) の新規作成 <!-- id: 17 -->
    - [ ] **騎手/調教師のキャリア**: 年齢 (`seinengappi`), 経験年数 (`menkyo_kofu_nengappi`)
    - [ ] **所属情報**: 東西区分 (`tozai_shozoku_code`)
- [ ] `src/preprocessing/features/logistics_features.py` (仮) の新規作成 <!-- id: 18 -->
    - [ ] **輸送フラグ**: 調教師所属 (栗東/美浦) × 開催競馬場 (関東/関西) の不一致判定
    - [ ] **遠征騎手**: 騎手所属 × 開催競馬場の不一致判定

## 4. 更新・実行
- [ ] `docs/feature_definitions_v4.md` の更新 <!-- id: 19 -->
- [ ] 前処理再実行: `preprocessed_data_v12.parquet` の更新 (Batch 2追加) <!-- id: 20 -->
- [ ] 特徴量分布確認 (`check_features_v12.py` 更新) <!-- id: 21 -->

## 5. 学習・評価
- [x] LambdaRankモデル再学習 (`exp_lambdarank_v12_batch2`) <!-- id: 22 -->
- [x] Batch 1 (No Odds) モデルとの比較 <!-- id: 23 -->

### Batch 2 結果 (2024 Test Set)
| Model | NDCG@3 | Win ROI |
| :--- | :--- | :--- |
| v12 Batch 2 (No Odds) | 0.5266 | 44.61% |
| v12 Batch 1 (No Odds) | 0.5268 | 44.65% |

*Note*: Batch 2 logistics features (`is_transported`, `is_away_jockey`) showed minimal impact on overall accuracy, likely due to already-captured jockey/trainer performance stats. Attribute features (`jockey_age`, `trainer_career_years`) are now available for the model.

# タスク: Batch 3 特徴量実装 (ラップ解析 & ペース特徴量)

## 1. 準備・設計
- [x] 実装計画書 (`implementation_plan.md`) の更新 (Batch 3) <!-- id: 24 -->
- [x] `jvd_ra.lap_time` のフォーマット確認 <!-- id: 25 -->

## 2. 実装: データロード
- [ ] `src/preprocessing/loader.py` の修正: `lap_time` 列を読み込む <!-- id: 26 -->

## 3. 実装: 特徴量エンジニアリング
- [x] `src/preprocessing/features/pace_features.py` の新規作成 <!-- id: 27 -->
    - [x] **ペース差 (Pace Diff)**: 前半3F - 後半3F
    - [x] **ペースタイプ**: Slow/Medium/Fast (カテゴリ)
    - [x] **ラップ分散**: ラップタイムの標準偏差 (ペースの乱れ度合い)
    - [x] **過去走ハイペース経験**: 馬の過去レースでのハイペース経験回数

## 4. 更新・実行
- [x] 前処理再実行: `preprocessed_data_v12.parquet` の更新 (Batch 3追加) <!-- id: 29 -->
- [x] 特徴量分布確認 <!-- id: 30 -->

## 5. 学習・評価
- [x] LambdaRank モデル再学習 <!-- id: 31 -->
- [x] Batch 2 (No Odds) モデルとの比較 <!-- id: 32 -->

### Batch 3 結果 (2024 Test Set)
| Model | NDCG@3 | Win ROI |
| :--- | :--- | :--- |
| **v12 Batch 3 (No Odds)** | **0.5289** | **45.89%** |
| v12 Batch 2 (No Odds) | 0.5266 | 44.61% |
| Baseline LTR (With Odds) | 0.5269 | 45.19% |
| Production Binary (No Odds) | 0.5313 | 45.17% |

*Note*: **+0.70% ROI improvement** from Batch 2. Now essentially matches Production Binary ROI while excluding odds.

# タスク: Batch 4 特徴量実装 (コーナー位置取り & Optuna HPO)

## 1. 実装
- [x] `src/preprocessing/features/corner_features.py` の新規作成
    - [x] `corner_position_change`: コーナー位置変化率
    - [x] `makuri_positions`: まくり度合い
    - [x] `late_charge`: ラストスパート
    - [x] `horse_avg_corner_change`: 過去平均コーナー変化
    - [x] `horse_makuri_rate`: マクリ率
    - [x] `horse_total_makuri`: 累計マクリ回数

## 2. Optuna HPO
- [x] 30回トライアルでハイパーパラメータ最適化
- [x] 最適パラメータでモデル再学習

### Batch 4 結果 (2024 Test Set)
| Model | NDCG@3 | Win ROI | 備考 |
| :--- | :--- | :--- | :--- |
| **v12 Batch 4 + Optuna** | **0.5605** | **46.77%** | ✨ **最高** |
| v12 Batch 3 (No Odds) | 0.5289 | 45.89% | - |
| Baseline LTR (With Odds) | 0.5269 | 45.20% | オッズあり |
| Production Binary (No Odds) | 0.5313 | 45.18% | 現行モデル |

*Note*: **+0.87% ROI improvement**, **+3.16% NDCG improvement**. Best model achieved.


