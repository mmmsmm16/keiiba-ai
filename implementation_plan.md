# 実装計画書 - Batch 1 特徴量 (マイニング & 馬場適性)

本計画書は、**マイニング指数関連特徴量** と **馬場状態適性特徴量** の実装手順を概説します。これらは「Quick Win」として高い優先度で実装されます。

## ユーザーレビュー事項
> [!NOTE]
> `jvd_wf` テーブルは Win5 払戻情報 ("WF") であり、詳細な馬場情報（含水率・クッション値）は含まれていないことが判明しました。
> **変更点**: 今回は `jvd_wf` の利用を見送り、確実に取得可能な **`going_code` (良・稍重・重・不良) を用いた適性特徴量** と **マイニング指数** の実装に集中します。

# 実装計画書 -# Batch 3 Implementation Plan: Lap Analysis & Pace Features

## Goal Description
Implement "Lap Analysis" and "Pace Features" to capture race pace characteristics and horse's experience with different pace scenarios.
Based on **Section B** in the feature gap analysis.

## Data Format
- `lap_time`: Fixed-width string of 3-character furlong times (e.g., "071109118..." = 7.1s, 10.9s, 11.8s per furlong)
- `zenhan_3f`: Front 3F time in 1/10 seconds (e.g., "349" = 34.9s)
- `kohan_3f`: Last 3F time in 1/10 seconds

## Proposed Changes

### Data Loading (`src/preprocessing/loader.py`)
#### [MODIFY] `loader.py`
- Ensure `lap_time`, `zenhan_3f`, `kohan_3f` are loaded from `jvd_ra`.
- These columns may already be available via the race data merge.

### Feature Engineering
#### [NEW] `src/preprocessing/features/pace_features.py`
- `PaceFeatureGenerator`:
    - **Race-Level (Current Race) - Require special handling:**
        - `pace_diff`: (zenhan_3f - kohan_3f) / 10.0 (seconds difference, positive = front-loaded pace)
        - `pace_type`: Slow (pace_diff < -2), Medium (-2 <= pace_diff <= 2), Fast (pace_diff > 2) (Categorical)
    - **Horse-Level (Historical - Aggregated from past races):**
        - `horse_fast_pace_count`: Number of past races where pace_diff > 2
        - `horse_slow_pace_count`: Number of past races where pace_diff < -2
        - `horse_avg_pace_diff`: Average pace_diff in past races
        - `horse_pace_versatility`: Std of pace_diff (how varied are the paces the horse experienced)

### Pipeline Integration (`src/preprocessing/feature_pipeline.py`)
- Register `PaceFeatureGenerator`.

## Verification Plan
### Automated Tests
- Check distribution of `pace_diff` (should be roughly normal around 0).
- Check `horse_fast_pace_count` for horses with many races.

### Manual Verification
- Validate pace_diff for known races (e.g., slow-paced G1 races).

    - **Note**: Requires mapping Race Track Codes to Regions (East/West).
        - East (Kanto): Tokyo, Nakayama, Fukushima, Niigata
        - West (Kansai): Kyoto, Hanshin, Chukyo, Kokura
        - Other: Sapporo, Hakodate (Considered 'Away' for everyone? Or specific logic?) -> Usually Ritto/Miho divide applies.

- Register new generators.

## 検証計画

### 自動テスト
    - `yoso_soha_time` が正しく秒単位に変換されているか（文字列パース確認）。
    - `horse_going_win_rate` が生データの戦績と整合しているか。

### 手動検証
- **分布確認**: `scripts/adhoc/check_feature_distribution.py` (作成予定) を用いて、新規特徴量の平均・標準偏差・欠損率をチェックします。
