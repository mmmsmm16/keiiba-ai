# データベース探索記録 (Database Exploration Log)

このドキュメントは、将来のモデリングに役立つ可能性があるデータベース内のテーブルやカラムを記録したものです。

## 調教データ
- **`jvd_hc`**: 馬ごとの調教（追い切り）記録。
    - 主なカラム: `time_gokei_4f`, `lap_time_4f`, `time_gokei_3f`, `lap_time_3f`, `time_gokei_2f`, `lap_time_2f`, `lap_time_1f`
    - 特徴: 調教時の4F〜1Fのラップタイムが含まれており、仕上がり状態の把握に有用。
    - **注意**: `chokyo_course_code` や `oikiri_level_code` はこのテーブルには存在しない（`pckeiba`データベースの`jvd_hc`テーブル定義において確認済み）。

## マスターデータ（厩舎・騎手）
- **`jvd_ch`**: 調教師マスター。
- **`jvd_ks`**: 騎手マスター。
    - 特徴: 生年月日、免許交付日などの基本属性が含まれる。

## マイニング / 外部予想系
- **`jvd_se`** (成績テーブル) 内の以下の項目:
    - `mining_kubun`: マイニング区分。
    - `yoso_soha_time`: 予想走破タイム。
    - `yoso_juni`: 予想着順。
    - 特徴: JRA-VANやPC-KEIBAが内部で算出している予測値。レース前に取得可能であれば、強力なメタ特徴量になる可能性がある。
