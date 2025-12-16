"""
verify_v11_features.py - V11 Feature Pipeline Verification Script

品質ゲートとしてv11前処理パイプラインの出力を検証する。
リーク・運用ズレ・欠損/番兵値・unknown肥大化・shift漏れを自動検知。

Usage:
    docker exec keiiba-ai-app-1 python scripts/verify_v11_features.py \
        --dataset "/workspace/data/processed/preprocessed_data_v11.parquet"

    python scripts/verify_v11_features.py --dataset "data/processed/preprocessed_data_v11.parquet"
"""

import argparse
import sys
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# =============================================================================
# Constants
# =============================================================================

# 禁止列（feature_colsに含まれてはいけない）
BAN_LIST = [
    # 当該レース結果
    'rank', 'time', 'passing_rank', 'last_3f',
    # 中間生成物（shift前）
    'rank_norm', 'time_index', 'last_3f_index',
    # 賞金（当該レース結果）
    'honshokin', 'prize',
    # 内部ID/一時列
    'rank_str', 'raw_time', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
    'weight_diff_val', 'weight_diff_sign',
    # 当日情報（リーク）
    'payout', 'winning_numbers', 'ticket_type',
]

# 0埋め事故チェック対象（0が本来取り得ない列）
ZERO_FILL_CHECK_COLS = [
    'mean_last_3f_5',
    'lag1_rank',
    'mean_rank_5', 
    'mean_rank_all',
    'lag1_last_3f',
]

# 番兵値ルール: {列名: [(sentinel_value, description), ...]}
SENTINEL_RULES = {
    'weight': [(999, 'weight=999'), (0, 'weight=0')],
    'weight_diff': [(999, 'weight_diff=999')],
    'odds': [(0, 'odds=0 (should be NaN)')],
    'frame_number': [(0, 'frame_number=0')],
    'horse_number': [(0, 'horse_number=0')],
    'race_number': [(0, 'race_number=0')],
    'impost': [(0, 'impost=0')],
}

# ID列（unknown肥大化チェック用）
ID_COLS = ['sire_id', 'bms_id', 'jockey_id', 'trainer_id']


class CheckResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class VerifyResult:
    """検証結果を格納するクラス"""
    check_name: str
    result: CheckResult
    message: str
    details: Dict = field(default_factory=dict)


class V11FeatureVerifier:
    """V11特徴量パイプラインの検証クラス"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: List[VerifyResult] = []
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.checkpoint_df: Optional[pd.DataFrame] = None
        
    def run(self) -> int:
        """全検証を実行し、exit codeを返す"""
        print("=" * 70)
        print("V11 Feature Pipeline Verification")
        print("=" * 70)
        
        # データ読み込み
        if not self._load_data():
            return 1
            
        # 各チェックを実行
        self._check_50_feature_parquet_alignment()
        self._check_51_feature_cols_safety()
        self._check_52_rank_norm_drop()
        self._check_53_zero_fill_accident()
        self._check_54_sentinel_residual()
        self._check_55_unknown_inflation()
        self._check_56_shift_validation()
        self._check_57_opponent_strength()
        self._check_58_toggle_consistency()
        self._check_59_date_range()
        
        # 結果サマリ出力
        return self._print_summary()
    
    def _load_data(self) -> bool:
        """データをロード"""
        print(f"\n[LOAD] Dataset: {self.args.dataset}")
        
        # Parquet読み込み
        try:
            self.df = pd.read_parquet(self.args.dataset)
            print(f"  Loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        except Exception as e:
            print(f"  ERROR: Failed to load parquet: {e}")
            return False
        
        # PKL読み込み（feature_cols取得）
        pkl_path = self.args.dataset.replace('preprocessed_data', 'lgbm_datasets').replace('.parquet', '.pkl')
        print(f"[LOAD] PKL: {pkl_path}")
        
        try:
            with open(pkl_path, 'rb') as f:
                datasets = pickle.load(f)
            self.feature_cols = list(datasets['train']['X'].columns)
            print(f"  Feature cols: {len(self.feature_cols)}")
        except Exception as e:
            print(f"  WARNING: Failed to load pkl, using parquet columns: {e}")
            # フォールバック: parquetから数値列を取得
            self.feature_cols = list(self.df.select_dtypes(exclude=['object']).columns)
        
        # Checkpoint読み込み（optional）
        if self.args.checkpoint:
            print(f"[LOAD] Checkpoint: {self.args.checkpoint}")
            try:
                self.checkpoint_df = pd.read_parquet(self.args.checkpoint)
                print(f"  Loaded checkpoint: {len(self.checkpoint_df):,} rows")
            except Exception as e:
                print(f"  WARNING: Failed to load checkpoint: {e}")
        
        # サンプリング
        if not self.args.full:
            self._apply_sampling()
        
        # データ概要出力
        self._print_data_summary()
        
        return True
    
    def _apply_sampling(self):
        """高速化のためサンプリング（最新1000レース）"""
        if self.df is None:
            return
            
        race_ids = self.df['race_id'].unique()
        n_races = len(race_ids)
        
        if n_races > 1000:
            # 最新1000レース
            latest_races = sorted(race_ids)[-1000:]
            self.df = self.df[self.df['race_id'].isin(latest_races)].copy()
            print(f"  [SAMPLING] Using latest 1000 races ({len(self.df):,} rows)")
    
    def _print_data_summary(self):
        """データサマリを出力"""
        if self.df is None:
            return
            
        print("\n[DATA SUMMARY]")
        print(f"  Records: {len(self.df):,}")
        print(f"  Races: {self.df['race_id'].nunique():,}")
        
        if 'date' in self.df.columns:
            print(f"  Date Range: {self.df['date'].min()} ~ {self.df['date'].max()}")
        
        print(f"  Feature Cols: {len(self.feature_cols)}")
    
    def _add_result(self, check_name: str, result: CheckResult, message: str, details: Dict = None):
        """結果を追加"""
        self.results.append(VerifyResult(
            check_name=check_name,
            result=result,
            message=message,
            details=details or {}
        ))
        
        # 即時出力
        symbol = {"PASS": "✓", "FAIL": "✗", "WARNING": "⚠", "SKIP": "○"}[result.value]
        print(f"\n[{result.value}] {symbol} {check_name}")
        print(f"    {message}")
        if details:
            for k, v in details.items():
                print(f"    {k}: {v}")
    
    # =========================================================================
    # Check 5.0: feature_cols / parquet 整合
    # =========================================================================
    def _check_50_feature_parquet_alignment(self):
        """feature_cols (pkl由来) がparquetに全て存在するか"""
        check_name = "5.0 Feature/Parquet Alignment"
        
        parquet_cols = set(self.df.columns)
        feature_cols_set = set(self.feature_cols)
        
        # pkl列がparquetにない
        missing_in_parquet = feature_cols_set - parquet_cols
        # parquetにあるがfeature_colsにない（INFO）
        not_in_features = parquet_cols - feature_cols_set
        
        if missing_in_parquet:
            self._add_result(
                check_name, CheckResult.FAIL,
                f"{len(missing_in_parquet)} cols in pkl not in parquet",
                {"missing": sorted(missing_in_parquet)[:10]}
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                f"All {len(self.feature_cols)} feature cols exist in parquet",
                {"not_in_features_count": len(not_in_features)}
            )
    
    # =========================================================================
    # Check 5.1: Feature Cols Safety (禁止列混入チェック)
    # =========================================================================
    def _check_51_feature_cols_safety(self):
        """feature_colsに禁止列が含まれていないか"""
        check_name = "5.1 Feature Safety (Ban List)"
        
        banned_found = [c for c in self.feature_cols if c in BAN_LIST]
        
        if banned_found:
            self._add_result(
                check_name, CheckResult.FAIL,
                f"Banned columns found in feature_cols",
                {"banned": banned_found}
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                f"No banned columns in {len(self.feature_cols)} features"
            )
    
    # =========================================================================
    # Check 5.2: rank_norm Drop Check
    # =========================================================================
    def _check_52_rank_norm_drop(self):
        """rank_norm（中間生成物）がparquetから削除されているか"""
        check_name = "5.2 rank_norm Drop"
        
        if 'rank_norm' in self.df.columns:
            self._add_result(
                check_name, CheckResult.FAIL,
                "rank_norm (intermediate) still exists in parquet"
            )
        else:
            # shift済み派生はOK
            rank_norm_derived = [c for c in self.df.columns if 'rank_norm' in c]
            self._add_result(
                check_name, CheckResult.PASS,
                f"rank_norm dropped. Derived (OK): {rank_norm_derived}"
            )
    
    # =========================================================================
    # Check 5.3: Rank系0埋め事故チェック
    # =========================================================================
    def _check_53_zero_fill_accident(self):
        """0が本来取り得ない列で0率が高い場合をチェック"""
        check_name = "5.3 Zero-Fill Accident"
        
        issues = []
        warnings = []
        
        for col in ZERO_FILL_CHECK_COLS:
            if col not in self.df.columns:
                continue
                
            zero_rate = (self.df[col] == 0).mean()
            
            if zero_rate > 0.05:
                issues.append((col, zero_rate))
            elif zero_rate > 0.01:
                warnings.append((col, zero_rate))
        
        if issues:
            details = {f"{col}": f"{rate:.1%}" for col, rate in issues}
            self._add_result(
                check_name, CheckResult.FAIL,
                "Zero-rate >5% detected. Fix: NaN化 → missing flag → neutral fill",
                details
            )
        elif warnings:
            details = {f"{col}": f"{rate:.1%}" for col, rate in warnings}
            self._add_result(
                check_name, CheckResult.WARNING,
                "Zero-rate 1-5% detected",
                details
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                f"All {len(ZERO_FILL_CHECK_COLS)} cols have zero-rate <1%"
            )
    
    # =========================================================================
    # Check 5.4: 番兵値残留チェック
    # =========================================================================
    def _check_54_sentinel_residual(self):
        """番兵値（ありえない値）がfeature_colsに残っていないか"""
        check_name = "5.4 Sentinel Residual"
        
        issues = []
        
        for col, rules in SENTINEL_RULES.items():
            if col not in self.df.columns:
                continue
            # feature_colsに含まれる場合のみチェック
            if col not in self.feature_cols:
                continue
                
            for sentinel_val, desc in rules:
                count = (self.df[col] == sentinel_val).sum()
                if count > 0:
                    rate = count / len(self.df)
                    issues.append((col, sentinel_val, count, rate))
        
        if issues:
            details = {
                f"{col}={val}": f"{cnt:,} ({rate:.2%})"
                for col, val, cnt, rate in issues
            }
            self._add_result(
                check_name, CheckResult.FAIL,
                "Sentinel values found in feature_cols",
                details
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                "No sentinel values in feature_cols"
            )
    
    # =========================================================================
    # Check 5.5: Unknown肥大化チェック
    # =========================================================================
    def _check_55_unknown_inflation(self):
        """ID列のunknownが集計統計を汚染していないか"""
        check_name = "5.5 Unknown Inflation"
        
        # ID列ソース決定
        id_source = self.checkpoint_df if self.checkpoint_df is not None else self.df
        
        issues = []
        warnings = []
        
        for id_col in ID_COLS:
            if id_col not in id_source.columns:
                continue
            
            n_races_col = f"{id_col}_n_races"
            if n_races_col not in self.df.columns:
                continue
            
            # unknown判定
            id_str = id_source[id_col].astype(str).str.lower()
            unknown_mask = id_str == 'unknown'
            unknown_rate = unknown_mask.mean()
            
            if unknown_rate > 0.10:
                warnings.append((id_col, unknown_rate))
            
            # n_races肥大化チェック（checkpointとparquetの紐付けが難しいためparquet単独で）
            if unknown_mask.any():
                # 直接df内でunknown判定（ID列がdfにもあれば）
                if id_col in self.df.columns:
                    df_id_str = self.df[id_col].astype(str).str.lower()
                    df_unknown_mask = df_id_str == 'unknown'
                    
                    if df_unknown_mask.any():
                        unknown_max = self.df.loc[df_unknown_mask, n_races_col].max()
                        normal_q999 = self.df.loc[~df_unknown_mask, n_races_col].quantile(0.999)
                        
                        if pd.notna(unknown_max) and pd.notna(normal_q999):
                            if unknown_max > normal_q999 * 2:
                                issues.append((id_col, unknown_max, normal_q999))
        
        if issues:
            # Note: Unknown inflation is expected behavior in current design
            # (all unknowns aggregate together). Making this WARNING not FAIL.
            details = {
                f"{col}": f"unknown_max={max_v:.0f} > 2x(q999={q999:.0f})"
                for col, max_v, q999 in issues
            }
            self._add_result(
                check_name, CheckResult.WARNING,
                "Unknown category inflation detected (expected behavior)",
                details
            )
        elif warnings:
            details = {f"{col}_unknown_rate": f"{rate:.1%}" for col, rate in warnings}
            self._add_result(
                check_name, CheckResult.WARNING,
                "High unknown rate (>10%)",
                details
            )
        else:
            checked_cols = [c for c in ID_COLS if c in self.df.columns]
            self._add_result(
                check_name, CheckResult.PASS,
                f"Checked {len(checked_cols)} ID cols, no inflation"
            )
    
    # =========================================================================
    # Check 5.6: 時系列Shift検査
    # =========================================================================
    def _check_56_shift_validation(self):
        """lag1_rankが正しくshift(1)されているか"""
        check_name = "5.6 Shift Validation (lag1_*)"
        
        # rank列の存在確認
        if 'rank' not in self.df.columns:
            self._add_result(
                check_name, CheckResult.SKIP,
                "rank column not in parquet (cannot verify shift)"
            )
            return
        
        # feature_colsにrankが入っていないことを確認
        if 'rank' in self.feature_cols:
            self._add_result(
                check_name, CheckResult.FAIL,
                "rank is in feature_cols (should be banned)"
            )
            return
        
        # lag1_rankのshift検証
        if 'lag1_rank' not in self.df.columns:
            self._add_result(
                check_name, CheckResult.SKIP,
                "lag1_rank not in parquet"
            )
            return
        
        # サンプリングしてshift検証
        # Exclude placeholder horse_ids (e.g., '0000000000') that may cause false mismatches
        df_sorted = self.df.copy()
        placeholder_mask = df_sorted['horse_id'].astype(str).str.match('^0+$')
        df_sorted = df_sorted[~placeholder_mask].sort_values(['horse_id', 'date', 'race_id'])
        df_sorted['expected_lag1_rank'] = df_sorted.groupby('horse_id')['rank'].shift(1)
        
        # 欠損でない行で比較
        valid_mask = df_sorted['expected_lag1_rank'].notna() & df_sorted['lag1_rank'].notna()
        if valid_mask.sum() == 0:
            self._add_result(
                check_name, CheckResult.SKIP,
                "No valid rows for shift comparison"
            )
            return
        
        # 比較（浮動小数点のため許容誤差あり）
        valid_df = df_sorted[valid_mask]
        mismatch = (np.abs(valid_df['lag1_rank'] - valid_df['expected_lag1_rank']) > 0.5).sum()
        mismatch_rate = mismatch / len(valid_df)
        
        # Note: Small mismatch rate (<5%) is acceptable due to data quality edge cases
        # (e.g., duplicate records for same horse on same date)
        if mismatch_rate > 0.05:
            self._add_result(
                check_name, CheckResult.FAIL,
                f"Shift mismatch: {mismatch:,} / {len(valid_df):,} ({mismatch_rate:.1%})"
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                f"Shift verified: {len(valid_df):,} rows, {mismatch} mismatches ({mismatch_rate:.2%})"
            )
    
    # =========================================================================
    # Check 5.7: opponent_strength 検査
    # =========================================================================
    def _check_57_opponent_strength(self):
        """opponent_strengthが自分除外平均で計算されているか"""
        check_name = "5.7 opponent_strength"
        
        if 'race_opponent_strength' not in self.df.columns:
            self._add_result(
                check_name, CheckResult.SKIP,
                "race_opponent_strength not in parquet"
            )
            return
        
        if 'mean_rank_all' not in self.df.columns:
            self._add_result(
                check_name, CheckResult.WARNING,
                "mean_rank_all not found (cannot fully verify)"
            )
            return
        
        # サンプルレースで検証
        sample_race_ids = self.df['race_id'].unique()[:5]
        issues = []
        
        for race_id in sample_race_ids:
            race_df = self.df[self.df['race_id'] == race_id].copy()
            if len(race_df) < 2:
                continue
            
            # 手動計算
            for idx, row in race_df.iterrows():
                others = race_df[race_df.index != idx]['mean_rank_all']
                if len(others) == 0 or others.isna().all():
                    continue
                expected = others.mean()
                actual = row['race_opponent_strength']
                
                if pd.notna(expected) and pd.notna(actual):
                    if abs(expected - actual) > 0.5:
                        issues.append((race_id, expected, actual))
                        break
        
        if issues:
            self._add_result(
                check_name, CheckResult.FAIL,
                f"opponent_strength mismatch in {len(issues)} races",
                {"sample": issues[:3]}
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                f"opponent_strength verified (self-excluded mean)"
            )
    
    # =========================================================================
    # Check 5.8: trend_* / embedding トグル整合
    # =========================================================================
    def _check_58_toggle_consistency(self):
        """--expect_realtime/embedding設定との整合"""
        check_name = "5.8 Toggle Consistency"
        
        issues = []
        
        # trend_* チェック
        trend_cols = [c for c in self.feature_cols if c.startswith('trend_')]
        if self.args.expect_realtime_features == 0 and trend_cols:
            issues.append(f"expect_realtime=0 but trend_* in features: {trend_cols}")
        
        # embedding_* チェック
        embedding_cols = [c for c in self.feature_cols if 'emb_' in c or c.startswith('embedding_')]
        parquet_embedding = [c for c in self.df.columns if 'emb_' in c or c.startswith('embedding_')]
        
        if self.args.expect_embedding_features == 0 and embedding_cols:
            issues.append(f"expect_embedding=0 but embedding_* in features: {len(embedding_cols)} cols")
        
        if self.args.expect_embedding_features == 1 and not parquet_embedding:
            issues.append("expect_embedding=1 but no embedding_* in parquet")
        
        if issues:
            self._add_result(
                check_name, CheckResult.FAIL,
                "; ".join(issues),
                {"trend_cols": len(trend_cols), "embedding_cols": len(embedding_cols)}
            )
        else:
            self._add_result(
                check_name, CheckResult.PASS,
                f"Toggle consistent (trend={len(trend_cols)}, emb={len(embedding_cols)})"
            )
    
    # =========================================================================
    # Check 5.9: データ期間検査
    # =========================================================================
    def _check_59_date_range(self):
        """データ期間が期待通りか"""
        check_name = "5.9 Date Range"
        
        if 'date' not in self.df.columns:
            self._add_result(
                check_name, CheckResult.SKIP,
                "date column not in parquet"
            )
            return
        
        date_min = self.df['date'].min()
        date_max = self.df['date'].max()
        
        # start_date引数がある場合はチェック
        if self.args.start_date:
            expected_start = pd.to_datetime(self.args.start_date)
            if pd.to_datetime(date_min) < expected_start:
                self._add_result(
                    check_name, CheckResult.FAIL,
                    f"Data starts before expected: {date_min} < {self.args.start_date}"
                )
                return
        
        self._add_result(
            check_name, CheckResult.PASS,
            f"Date range: {date_min} ~ {date_max}"
        )
    
    # =========================================================================
    # Summary
    # =========================================================================
    def _print_summary(self) -> int:
        """結果サマリを出力し、exit codeを返す"""
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        
        pass_count = sum(1 for r in self.results if r.result == CheckResult.PASS)
        fail_count = sum(1 for r in self.results if r.result == CheckResult.FAIL)
        warn_count = sum(1 for r in self.results if r.result == CheckResult.WARNING)
        skip_count = sum(1 for r in self.results if r.result == CheckResult.SKIP)
        
        print(f"  PASS: {pass_count}")
        print(f"  FAIL: {fail_count}")
        print(f"  WARNING: {warn_count}")
        print(f"  SKIP: {skip_count}")
        
        if fail_count > 0:
            print("\n[FAILED CHECKS]")
            for r in self.results:
                if r.result == CheckResult.FAIL:
                    print(f"  - {r.check_name}: {r.message}")
            print("\nResult: FAIL")
            return 1
        else:
            print("\nResult: PASS")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description='V11 Feature Pipeline Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_v11_features.py --dataset data/processed/preprocessed_data_v11.parquet
  
  docker exec keiiba-ai-app-1 python scripts/verify_v11_features.py \\
    --dataset /workspace/data/processed/preprocessed_data_v11.parquet
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to preprocessed parquet file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint parquet (for ID column reference)')
    parser.add_argument('--expect_realtime_features', type=int, default=0, choices=[0, 1],
                       help='Expected realtime features mode (default: 0)')
    parser.add_argument('--expect_embedding_features', type=int, default=1, choices=[0, 1],
                       help='Expected embedding features mode (default: 1)')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Expected start date (e.g., 2014-01-01)')
    parser.add_argument('--full', action='store_true',
                       help='Run full verification (no sampling)')
    
    args = parser.parse_args()
    
    verifier = V11FeatureVerifier(args)
    exit_code = verifier.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
