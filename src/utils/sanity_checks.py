"""
Sanity Check Utilities for Backtest Validation
バックテストの整合性検証ユーティリティ

Usage:
    from utils.sanity_checks import validate_win_payout_integrity, validate_odds_vs_payout_separation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SanityCheckResult:
    """サニティチェック結果"""
    
    def __init__(self, check_name: str):
        self.check_name = check_name
        self.passed = True
        self.violations = 0
        self.warnings = 0
        self.message = ""
        self.sample_race_ids = []
        self.details = {}
    
    def fail(self, message: str, violations: int = 0, samples: List = None):
        self.passed = False
        self.message = message
        self.violations = violations
        self.sample_race_ids = samples or []
        return self
    
    def warn(self, message: str, warnings: int = 0, samples: List = None):
        self.warnings = warnings
        self.message = message
        self.sample_race_ids = samples or []
        return self
    
    def pass_check(self, message: str = "OK"):
        self.passed = True
        self.message = message
        return self
    
    def to_dict(self) -> Dict:
        return {
            'check_name': self.check_name,
            'passed': self.passed,
            'violations': self.violations,
            'warnings': self.warnings,
            'message': self.message,
            'sample_race_ids': self.sample_race_ids[:20],
            'details': self.details
        }


def validate_win_payout_integrity(
    df: pd.DataFrame,
    race_id_col: str = 'race_id',
    horse_key_col: str = 'horse_id',
    y_win_col: str = 'is_win',
    bet_col: str = 'bet_amount',
    payout_col: str = 'payout',
    strict: bool = True,
    bets_only: bool = True  # NEW: if True, df contains only bet rows (not full race data)
) -> SanityCheckResult:
    """
    単勝の勝敗→払戻整合性を検証
    
    ルール:
    1. bet=0 → payout=0 or NaN (if not bets_only)
    2. bet>0 & y_win=0 → payout=0
    3. bet>0 & y_win=1 → payout>0
    4. 各レースで y_win=1 がちょうど1行 (if not bets_only; for bets_only, 0 or 1 is valid)
    """
    result = SanityCheckResult("win_payout_integrity")
    violations = []
    
    # Check 1: bet=0 → payout=0 or NaN (only if full race data)
    if not bets_only and bet_col in df.columns and payout_col in df.columns:
        no_bet = df[df[bet_col] == 0]
        bad_payout = no_bet[(no_bet[payout_col].notna()) & (no_bet[payout_col] > 0)]
        if len(bad_payout) > 0:
            violations.append(f"bet=0なのにpayout>0: {len(bad_payout)}件")
            result.details['bet0_payout_violations'] = len(bad_payout)
            result.sample_race_ids.extend(bad_payout[race_id_col].head(10).tolist())
    
    # Check 2: bet>0 & y_win=0 → payout=0
    if bet_col in df.columns and y_win_col in df.columns and payout_col in df.columns:
        bet_lose = df[(df[bet_col] > 0) & (df[y_win_col] == 0)]
        bad_payout_lose = bet_lose[bet_lose[payout_col] > 0]
        if len(bad_payout_lose) > 0:
            violations.append(f"bet>0 & 負け なのに payout>0: {len(bad_payout_lose)}件")
            result.details['bet_lose_payout_violations'] = len(bad_payout_lose)
            result.sample_race_ids.extend(bad_payout_lose[race_id_col].head(10).tolist())
    
    # Check 3: bet>0 & y_win=1 → payout>0
    if bet_col in df.columns and y_win_col in df.columns and payout_col in df.columns:
        bet_win = df[(df[bet_col] > 0) & (df[y_win_col] == 1)]
        bad_payout_win = bet_win[bet_win[payout_col] <= 0]
        if len(bad_payout_win) > 0:
            violations.append(f"bet>0 & 勝ち なのに payout<=0: {len(bad_payout_win)}件")
            result.details['bet_win_no_payout_violations'] = len(bad_payout_win)
            result.sample_race_ids.extend(bad_payout_win[race_id_col].head(10).tolist())
    
    # Check 4: 各レースで y_win=1 がちょうど1行 (only if full race data)
    # For bets_only mode, we expect 0 or 1 winners per bet race (not exactly 1)
    if y_win_col in df.columns and not bets_only:
        winners_per_race = df.groupby(race_id_col)[y_win_col].sum()
        multi_winners = winners_per_race[winners_per_race != 1]
        if len(multi_winners) > 0:
            violations.append(f"y_win=1が1でないレース: {len(multi_winners)}件")
            result.details['multi_winner_races'] = len(multi_winners)
            result.sample_race_ids.extend(multi_winners.head(10).index.tolist())
    
    # For bets_only mode, add stats about winners
    if y_win_col in df.columns and bets_only:
        total_bets = len(df)
        total_wins = df[y_win_col].sum()
        result.details['total_bets'] = total_bets
        result.details['total_wins'] = int(total_wins)
        result.details['hit_rate'] = float(total_wins / total_bets * 100) if total_bets > 0 else 0
    
    if violations:
        result.fail("; ".join(violations), sum(result.details.values()), result.sample_race_ids)
        if strict:
            raise ValueError(f"Sanity check FAILED: {result.message}")
    else:
        msg = "全ルールPASS"
        if bets_only and 'hit_rate' in result.details:
            msg += f" (Hit Rate: {result.details['hit_rate']:.1f}%)"
        result.pass_check(msg)
    
    return result


def validate_odds_vs_payout_separation(
    df: pd.DataFrame,
    odds_col: str = 'odds',
    payout_col: str = 'returns',
    bet_col: str = 'bet_amount',
    is_win_col: str = 'is_win'
) -> SanityCheckResult:
    """
    odds（事前オッズ）と payout（確定払戻）の関係を検証
    
    単勝では payout/bet = odds が正常動作。
    ただし100%一致はリーク疑い点として WARNING出力。
    """
    result = SanityCheckResult("odds_vs_payout_separation")
    
    if odds_col not in df.columns:
        result.warn(f"Column {odds_col} not found")
        return result
    
    if payout_col not in df.columns:
        result.warn(f"Column {payout_col} not found")
        return result
    
    # 分布統計
    odds_stats = {
        'min': float(df[odds_col].min()),
        'median': float(df[odds_col].median()),
        'p95': float(df[odds_col].quantile(0.95)),
        'max': float(df[odds_col].max()),
    }
    
    payout_stats = {
        'min': float(df[payout_col].min()),
        'median': float(df[payout_col].median()),
        'p95': float(df[payout_col].quantile(0.95)),
        'max': float(df[payout_col].max()),
    }
    
    result.details['odds_stats'] = odds_stats
    result.details['payout_stats'] = payout_stats
    
    # 勝ち馬の payout/bet と odds の差分分布
    if bet_col in df.columns and is_win_col in df.columns:
        winners = df[(df[is_win_col] == 1) & (df[bet_col] > 0) & (df[payout_col] > 0)].copy()
        
        if len(winners) > 0:
            winners['implied_odds'] = winners[payout_col] / winners[bet_col]
            winners['diff'] = winners['implied_odds'] - winners[odds_col]
            winners['abs_diff'] = abs(winners['diff'])
            
            # 差分分布
            diff_stats = {
                'min': float(winners['diff'].min()),
                'median': float(winners['diff'].median()),
                'mean': float(winners['diff'].mean()),
                'p95': float(winners['diff'].quantile(0.95)),
                'max': float(winners['diff'].max()),
                'zero_rate': float((winners['abs_diff'] < 0.01).mean()),
            }
            result.details['diff_stats'] = diff_stats
            
            # 完全一致率（許容誤差0.1以内）
            exact_match_rate = (winners['abs_diff'] < 0.1).mean()
            result.details['winner_count'] = len(winners)
            result.details['exact_match_rate'] = float(exact_match_rate)
            
            # payout が100円刻みになっている割合
            payout_100_rate = ((winners[payout_col] % 100) < 1).mean()
            result.details['payout_100yen_rate'] = float(payout_100_rate)
            
            # 常に WARNING (100%一致でも単勝では正常だが注意喚起)
            if exact_match_rate > 0.99:
                result.warnings = 1
                result.message = f"WARNING: payout/bet = odds が{exact_match_rate:.0%}一致 (単勝では正常だがodds時刻確認推奨)"
            else:
                result.message = f"odds/payout差分あり (一致率={exact_match_rate:.1%})"
            
            # diff が全て 0 の場合は RED FLAG
            if diff_stats['zero_rate'] > 0.999:
                result.warnings = 2
                result.message += " [RED FLAG: diff常に0]"
            
            result.passed = True  # WARNING は PASS 扱い（FAILではない）
        else:
            result.warn("勝ちbet=0、検証不可")
    else:
        result.pass_check("分布は異なる（詳細検証スキップ）")
    
    return result


def validate_ev_definition(
    df: pd.DataFrame,
    prob_col: str = 'prob',
    odds_col: str = 'odds',
    ev_col: str = 'ev',
    tolerance: float = 0.01
) -> SanityCheckResult:
    """
    EV = prob * odds の定義が正しいことを検証
    """
    result = SanityCheckResult("ev_definition")
    
    if ev_col not in df.columns:
        result.warn(f"Column {ev_col} not found")
        return result
    
    if prob_col not in df.columns or odds_col not in df.columns:
        result.warn(f"prob_col or odds_col not found")
        return result
    
    # 期待値の再計算
    expected_ev = df[prob_col] * df[odds_col]
    actual_ev = df[ev_col]
    
    # 差分
    diff = abs(expected_ev - actual_ev)
    bad = diff > tolerance
    
    if bad.sum() > 0:
        result.fail(
            f"EV定義が一致しない行: {bad.sum()}件 (許容誤差={tolerance})",
            violations=int(bad.sum())
        )
    else:
        result.pass_check(f"EV = prob * odds (誤差{tolerance}以内)")
    
    result.details['mean_diff'] = float(diff.mean())
    result.details['max_diff'] = float(diff.max())
    
    return result


def validate_p_market_recomputation(
    df: pd.DataFrame,
    p_market_col: str = 'p_market',
    odds_col: str = 'odds',
    race_id_col: str = 'race_id',
    tolerance: float = 1e-6
) -> SanityCheckResult:
    """
    p_market が odds から正しく計算されているかを検証
    p_market = (1/odds) / Σ(1/odds) per race
    """
    result = SanityCheckResult("p_market_recomputation")
    
    if p_market_col not in df.columns:
        result.warn(f"Column {p_market_col} not found")
        return result
    
    if odds_col not in df.columns:
        result.warn(f"Column {odds_col} not found")
        return result
    
    # odds > 0 のみ対象
    valid = df[(df[odds_col] > 0) & (df[p_market_col].notna())].copy()
    
    if len(valid) == 0:
        result.warn("有効行なし")
        return result
    
    # 再計算
    valid['raw_prob'] = 1.0 / valid[odds_col]
    valid['sum_raw'] = valid.groupby(race_id_col)['raw_prob'].transform('sum')
    valid['p_market_recomputed'] = valid['raw_prob'] / valid['sum_raw']
    
    # 差分
    valid['abs_diff'] = abs(valid[p_market_col] - valid['p_market_recomputed'])
    
    # 統計
    diff_stats = {
        'max': float(valid['abs_diff'].max()),
        'p99': float(valid['abs_diff'].quantile(0.99)),
        'mean': float(valid['abs_diff'].mean()),
        'above_tol_rate': float((valid['abs_diff'] > tolerance).mean()),
    }
    result.details['diff_stats'] = diff_stats
    result.details['total_rows'] = len(valid)
    result.details['total_races'] = valid[race_id_col].nunique()
    
    # 乖離があるレースをサンプル
    bad_races = valid[valid['abs_diff'] > tolerance][race_id_col].unique()[:20].tolist()
    result.details['bad_race_samples'] = bad_races
    
    if diff_stats['above_tol_rate'] > 0.01:
        result.warn(
            f"p_market再計算乖離: {diff_stats['above_tol_rate']:.1%}が許容誤差{tolerance}超",
            warnings=1,
            samples=bad_races
        )
    else:
        result.pass_check(f"p_market = (1/odds)/Σ(1/odds) 一致 (max_diff={diff_stats['max']:.2e})")
    
    return result


def generate_sample_table(
    bets_df: pd.DataFrame,
    full_race_df: pd.DataFrame = None,  # NEW: Full race data for winner lookup
    n_races: int = 100,
    seed: int = 42,
    race_id_col: str = 'race_id',
    horse_key_col: str = 'horse_id',
    prob_col: str = 'prob',
    odds_col: str = 'odds',
    ev_col: str = 'ev',
    bet_col: str = 'bet_amount',
    is_win_col: str = 'is_win',
    payout_col: str = 'returns',
    rank_col: str = 'rank'
) -> Tuple[str, Dict]:
    """
    ランダムN レースのサンプル表を生成（Markdown形式）
    
    Args:
        bets_df: ベット行のみを含むDataFrame
        full_race_df: 全レースデータ（勝ち馬抽出用）。Noneの場合はbets_dfから推定（不正確）
    
    Returns:
        (markdown_str, stats_dict) - サンプル表とサマリー統計
    """
    np.random.seed(seed)
    
    # Use bets_df for race selection
    all_races = bets_df[race_id_col].unique()
    sample_races = np.random.choice(all_races, min(n_races, len(all_races)), replace=False)
    
    # Stats tracking
    stats = {
        'total_sample_races': len(sample_races),
        'races_with_winner': 0,
        'races_without_winner': 0,
        'multi_winner_races': 0,
        'selected_is_winner_count': 0,
    }
    
    lines = []
    lines.append(f"# Sanity Check Sample Table\n")
    lines.append(f"**Sampled Races**: {len(sample_races)}\n")
    lines.append(f"**Seed**: {seed}\n")
    lines.append(f"**Full Race Data Available**: {'Yes' if full_race_df is not None else 'No (winner detection may be incomplete)'}\n\n")
    
    # Choose winner source
    winner_source = full_race_df if full_race_df is not None else bets_df
    
    # サマリー統計
    sample_bets = bets_df[bets_df[race_id_col].isin(sample_races)].copy()
    
    if is_win_col not in sample_bets.columns and rank_col in sample_bets.columns:
        sample_bets[is_win_col] = (sample_bets[rank_col] == 1).astype(int)
    
    lines.append("## Summary\n")
    lines.append("| Metric | Value |\n")
    lines.append("|--------|-------|\n")
    
    if len(sample_bets) > 0:
        n_bet_races = sample_bets[race_id_col].nunique()
        n_hits = sample_bets[is_win_col].sum() if is_win_col in sample_bets.columns else 0
        hit_rate = n_hits / len(sample_bets) * 100 if len(sample_bets) > 0 else 0
        
        lines.append(f"| Bet Rows | {len(sample_bets)} |\n")
        lines.append(f"| Bet Races | {n_bet_races} |\n")
        lines.append(f"| Hits | {int(n_hits)} |\n")
        lines.append(f"| Hit Rate | {hit_rate:.1f}% |\n")
    
    lines.append("\n## Sample Races (first 20)\n\n")
    
    # 各レースの詳細（最大20レース表示）
    display_races = sample_races[:20]
    
    for race_id in display_races:
        # Bets for this race
        race_bets = bets_df[bets_df[race_id_col] == race_id].copy()
        
        # Winner from full race data (or bets_df if not available)
        race_full = winner_source[winner_source[race_id_col] == race_id].copy()
        
        # Compute is_win if not present
        if is_win_col not in race_full.columns and rank_col in race_full.columns:
            race_full[is_win_col] = (race_full[rank_col] == 1).astype(int)
        if is_win_col not in race_bets.columns and rank_col in race_bets.columns:
            race_bets[is_win_col] = (race_bets[rank_col] == 1).astype(int)
        
        # Find winner
        winner_rows = race_full[race_full[is_win_col] == 1] if is_win_col in race_full.columns else pd.DataFrame()
        y_win_count = len(winner_rows)
        
        # Track stats
        if y_win_count == 1:
            stats['races_with_winner'] += 1
        elif y_win_count == 0:
            stats['races_without_winner'] += 1
        else:
            stats['multi_winner_races'] += 1
        
        lines.append(f"### Race: {race_id}\n")
        lines.append(f"**y_win_count_in_race**: {y_win_count} {'✅' if y_win_count == 1 else '⚠️'}\n\n")
        
        # Bets table
        lines.append("**Selected (bet>0):**\n")
        if len(race_bets) > 0:
            lines.append("| Horse | Prob | Odds | EV | Bet | IsWin | Payout | WinMatch |\n")
            lines.append("|-------|------|------|-----|-----|-------|--------|----------|\n")
            
            for _, row in race_bets.iterrows():
                horse = row.get(horse_key_col, '?')
                prob = row.get(prob_col, 0)
                odds = row.get(odds_col, 0)
                ev = row.get(ev_col, prob * odds if prob and odds else 0)
                bet = row.get(bet_col, 0)
                is_win = row.get(is_win_col, 0)
                payout = row.get(payout_col, 0)
                
                # Check if this bet matches winner
                selected_is_winner = is_win == 1
                if selected_is_winner:
                    stats['selected_is_winner_count'] += 1
                
                win_mark = "✅" if selected_is_winner else ""
                lines.append(f"| {horse} | {prob:.3f} | {odds:.1f} | {ev:.2f} | {bet} | {int(is_win)} | {payout:.0f} | {win_mark} |\n")
        else:
            lines.append("(No bets in this race)\n")
        
        # Winner info from full race data
        lines.append("\n**Winner (from full race data):**\n")
        if len(winner_rows) > 0:
            for _, row in winner_rows.iterrows():
                horse = row.get(horse_key_col, '?')
                prob = row.get(prob_col, 0) if prob_col in row else 'N/A'
                odds = row.get(odds_col, 0)
                rank = row.get(rank_col, '?')
                prob_str = f"{prob:.3f}" if isinstance(prob, (int, float)) else prob
                lines.append(f"- Horse: {horse}, Rank: {rank}, Odds: {odds:.1f}, Prob: {prob_str}\n")
        else:
            lines.append("⚠️ No winner found in race data\n")
        
        lines.append("\n---\n\n")
    
    # Add winner detection summary
    lines.insert(8, f"| Races with Winner | {stats['races_with_winner']} |\n")
    lines.insert(9, f"| Races without Winner | {stats['races_without_winner']} |\n")
    lines.insert(10, f"| Multi-winner Races | {stats['multi_winner_races']} |\n")
    lines.insert(11, f"| Selected=Winner | {stats['selected_is_winner_count']} |\n")
    
    return "".join(lines), stats


def run_all_sanity_checks(
    df: pd.DataFrame,
    prob_col: str = 'prob',
    odds_col: str = 'odds',
    strict: bool = True,
    generate_samples: bool = True,
    sample_output_path: str = None
) -> Tuple[List[SanityCheckResult], str]:
    """
    全サニティチェックを実行
    """
    results = []
    
    # 必須列が存在するか確認
    required_cols = ['race_id', 'horse_id', odds_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for sanity check: {missing}")
    
    # Check 1: Win/Payout integrity
    if 'is_win' in df.columns and 'bet_amount' in df.columns:
        result1 = validate_win_payout_integrity(
            df, strict=False  # Don't raise here, collect all results first
        )
        results.append(result1)
    
    # Check 2: Odds vs Payout separation
    if odds_col in df.columns and 'returns' in df.columns:
        result2 = validate_odds_vs_payout_separation(
            df, odds_col=odds_col
        )
        results.append(result2)
    
    # Check 3: EV definition
    if 'ev' in df.columns and prob_col in df.columns and odds_col in df.columns:
        result3 = validate_ev_definition(
            df, prob_col=prob_col, odds_col=odds_col
        )
        results.append(result3)
    
    # Generate sample table
    sample_report = ""
    if generate_samples:
        sample_report = generate_sample_table(
            df, prob_col=prob_col, odds_col=odds_col
        )
        if sample_output_path:
            with open(sample_output_path, 'w', encoding='utf-8') as f:
                f.write(sample_report)
            logger.info(f"Sample table saved to {sample_output_path}")
    
    # Strict mode: fail if any check failed
    if strict:
        failed = [r for r in results if not r.passed]
        if failed:
            msgs = [f"{r.check_name}: {r.message}" for r in failed]
            raise ValueError(f"Sanity checks FAILED: {'; '.join(msgs)}")
    
    return results, sample_report
