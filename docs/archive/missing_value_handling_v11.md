# 競馬AI: 欠損値・異常値処理ドキュメント (v11)

## 概要

**データバージョン**: v11 (2025-12-15更新)

**主な変更点 (v10_leakfix → v11)**:
- A1: rank系の0埋め廃止 → ニュートラル値(8.0)補完
- A2: 番兵値の検出・変換処理を体系化
- 欠損フラグ（*_is_missing）の追加

---

## 1. 欠損値処理方針

### [v11] 基本方針

| 種別 | v10以前 | v11 |
|------|--------|-----|
| rank系（着順） | 0埋め | **8.0（中位）で補完** + 欠損フラグ |
| 番兵値 | そのまま | **NaN化または適正値補完** |
| ID欠損 | 0 or unknown | **'unknown'に統一** |

### 詳細ルール

1. **初出走馬**: lag/rolling特徴は自然とNaNになる → **ニュートラル値で補完 + 欠損フラグ**
2. **番兵値**: odds=0, weight=999等 → **NaN化または適正値補完**
3. **データ欠落**: mare_id/bms_idなど → **'unknown'で統一**
4. **計算不能**: 標準化・相対化特徴(単頭レース等) → **0で埋める**

---

## 2. 番兵値処理詳細 (A2)

### 検出・変換ルール

| カラム | 無効値 | 処理方法 | 備考 |
|--------|--------|----------|------|
| `odds` | 0, 0.0 | NaN化 | 単勝オッズ未設定 |
| `weight` | 0, >=999, <300 | 中央値（約470kg）で補完 | 計測不能・未入力 |
| `weight_diff` | <=-99, >=999 | 0で補完 | 初出走または計測不能 |
| `impost` | 0, NaN | 平均値（約55kg）で補完 | 未設定 |
| `frame_number` | 0 | NaN化 | エントリーエラー |
| `horse_number` | 0 | NaN化 | エントリーエラー |
| `rank` | 0, NaN | **行削除** | 取消・中止 |
| `time` | 0 | 検出のみ（警告ログ） | レース結果なので学習には使用しない |

### 実装箇所

- `cleansing.py`: `_handle_sentinel_values()` メソッド
- `validation_utils.py`: `handle_sentinel_values()` 関数（共通ユーティリティ）

---

## 3. 欠損フラグ (A1)

### [v11新規] 欠損フラグカラム

rank系の欠損を明示的に追跡するためのフラグカラムを追加。

| フラグカラム | 対応する特徴量 | 説明 |
|-------------|---------------|------|
| `lag1_rank_is_missing` | `lag1_rank` | 初出走の場合=1 |
| `mean_rank_5_is_missing` | `mean_rank_5` | 5走未満の場合=1 |
| `mean_rank_all_is_missing` | `mean_rank_all` | 初出走の場合=1 |
| `lag1_last_3f_is_missing` | `lag1_last_3f` | 初出走の場合=1 |
| `mean_last_3f_5_is_missing` | `mean_last_3f_5` | 5走未満の場合=1 |

### 実装箇所

- `aggregators.py`: `HistoryAggregator.aggregate()` 内で生成

---

## 4. ニュートラル補完値 (A1)

### [v11変更] rank系の補完値

| カラム | v10以前 | v11 | 根拠 |
|--------|--------|-----|------|
| `lag1_rank` | 10 | **8.0** | 16頭立て中位 |
| `mean_rank_5` | 0 | **8.0** | 16頭立て中位 |
| `mean_rank_all` | 0→NaN→0 | **8.0** | 16頭立て中位 |
| `lag1_last_3f` | 0 | **35.0** | 上がり3F平均値 |
| `mean_last_3f_5` | 0 | **35.0** | 上がり3F平均値 |

### 定数定義

```python
# aggregators.py
RANK_NEUTRAL_VALUE = 8.0       # 16頭立て中位
LAST_3F_NEUTRAL_VALUE = 35.0   # 上がり3F平均値
```

---

## 5. unknown/0 汚染監視 (A3)

### 監視ログ

`category_aggregators.py` で以下を監視:

1. **上位カテゴリ分布**: 各ID列の上位10件をログ出力
2. **unknown警告**: unknownが5%以上を占める場合は警告
3. **n_races異常値**: 最大値が99パーセンタイルの5倍以上の場合は警告

### ログ出力例

```
INFO - v11: カテゴリ分布の監視中...
WARNING - [A3警告] trainer_id の 'unknown' が 12.3% を占めています。集計特徴量が汚染されている可能性があります。
```

---

## 6. データ品質バリデーション

### パイプライン終了時チェック

`run_preprocessing.py` の最後に `validate_data_quality()` を呼び出し:

- rank系の0率チェック（5%以上で警告）
- 番兵値残存チェック
- unknown率チェック（10%以上で警告）
- 全体NaN率チェック（5%以上で警告）

### 実装箇所

- `validation_utils.py`: `validate_data_quality()` 関数

---

## 7. コード例: v11補完処理

```python
# aggregators.py より抜粋

# 欠損フラグを先に生成
df['lag1_rank_is_missing'] = df['lag1_rank'].isna().astype('int8')

# ニュートラル値で補完
df['lag1_rank'] = df['lag1_rank'].fillna(RANK_NEUTRAL_VALUE).astype('float32')
```

---

## 8. まとめ

### v11の欠損値処理フロー

1. **クレンジング段階** (`cleansing.py`)
   - rank=0/NaN → 行削除
   - weight/impost/odds の番兵値 → NaN化または適正値補完

2. **特徴量生成段階** (`aggregators.py`)
   - 欠損フラグ生成 (*_is_missing)
   - ニュートラル値補完（rank系→8.0, last_3f系→35.0）

3. **カテゴリ集計段階** (`category_aggregators.py`)
   - ID欠損 → 'unknown' に統一
   - unknown監視ログ出力

4. **パイプライン終了時** (`run_preprocessing.py`)
   - データ品質バリデーション実行

---

**更新日時**: 2025-12-15  
**データバージョン**: v11  
**前バージョン**: v10_leakfix（docs/archive/ に移動）
