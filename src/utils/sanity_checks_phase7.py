"""
Phase 7 Sanity Checks - Ticket Payout Integrity
BOX買いの払戻整合性を検証

Usage:
    from utils.sanity_checks_phase7 import validate_ticket_payout_integrity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Set
from itertools import combinations, permutations
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TicketPayoutResult:
    """Phase7 sanity check result"""
    
    def __init__(self, check_name: str):
        self.check_name = check_name
        self.passed = False
        self.message = ""
        self.warnings = 0
        self.violations = 0
        self.details = {}
        self.samples = []
    
    def pass_check(self, msg: str):
        self.passed = True
        self.message = msg
    
    def fail(self, msg: str, violations: int = 1):
        self.passed = False
        self.message = msg
        self.violations = violations
    
    def warn(self, msg: str, warnings: int = 1):
        self.passed = True  # Warning is still pass
        self.message = msg
        self.warnings = warnings


# =============================================================================
# Ticket Format Conversion Functions (統一フォーマット)
# =============================================================================

def format_combination_padded(horses: List[int], ordered: bool = False) -> str:
    """
    馬番のリストを組み合わせ文字列に変換 (ゼロ埋め連結フォーマット)
    
    Args:
        horses: [1, 3, 5] など
        ordered: Trueなら順あり（三連単）、Falseなら順不同（馬連/三連複）
    
    Returns:
        "010305" など (公式払戻テーブルと同じフォーマット)
    """
    if ordered:
        formatted = horses
    else:
        formatted = sorted(horses)
    
    return "".join([f"{h:02}" for h in formatted])


def format_combination_display(horses: List[int], ordered: bool = False) -> str:
    """
    馬番のリストを表示用文字列に変換
    
    Returns:
        "1-3-5" など (人間が読みやすい形式)
    """
    if ordered:
        return '-'.join(map(str, horses))
    else:
        return '-'.join(map(str, sorted(horses)))


def decode_official_winner(code: str, ticket_type: str) -> Tuple:
    """
    公式当たり組み合わせ文字列をタプルにデコード
    
    Args:
        code: "010305" or "041205" など (2桁ゼロ埋め連結)
        ticket_type: 'sanrenpuku' or 'sanrentan' or 'umaren'
    
    Returns:
        (1, 3, 5) - 順不同の場合はsorted tuple, 順ありの場合は順序維持
    """
    code = str(code).strip()
    
    # Determine number of horses based on ticket type
    if ticket_type in ['sanrenpuku', 'sanrentan']:
        n_horses = 3
    elif ticket_type == 'umaren':
        n_horses = 2
    else:
        n_horses = 3  # Default
    
    # Parse 2-digit chunks
    horses = []
    for i in range(0, len(code), 2):
        if i + 2 <= len(code):
            try:
                horses.append(int(code[i:i+2]))
            except ValueError:
                pass
    
    # Validate
    if len(horses) != n_horses:
        logger.warning(f"decode_official_winner: expected {n_horses} horses, got {len(horses)} from '{code}'")
        return tuple()
    
    # Return appropriate format
    ordered = ticket_type == 'sanrentan'
    if ordered:
        return tuple(horses)
    else:
        return tuple(sorted(horses))


def parse_ticket_tuple(t: Tuple[int, ...], ticket_type: str) -> Tuple:
    """
    チケットタプルを正規化
    
    Args:
        t: (4, 12, 5) など
        ticket_type: 'sanrenpuku' or 'sanrentan'
    
    Returns:
        正規化されたタプル (sanrenpukuならsorted)
    """
    ordered = ticket_type == 'sanrentan'
    if ordered:
        return tuple(t)
    else:
        return tuple(sorted(t))


def ticket_tuple_to_official_key(t: Tuple[int, ...], ticket_type: str) -> str:
    """
    チケットタプルを公式キー文字列に変換
    
    Args:
        t: (4, 12, 5)
        ticket_type: 'sanrenpuku' or 'sanrentan'
    
    Returns:
        "041205" (sanrenpuku sorted) or "041205" (sanrentan ordered)
    """
    ordered = ticket_type == 'sanrentan'
    return format_combination_padded(list(t), ordered=ordered)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_ticket_payout_integrity(
    race_id: str,
    tickets: List[Tuple],
    official_payouts: Dict[str, float],
    ticket_type: str,
    bet_unit: float = 100.0,
    rescale_ratio: float = 1.0
) -> Dict:
    """
    単一レースのチケット払戻整合性を検証
    
    Args:
        race_id: レースID
        tickets: 購入したチケット(タプルのリスト)
        official_payouts: 公式払戻テーブル {組み合わせ: 払戻額} (キーは"010305"形式)
        ticket_type: 券種 (umaren/sanrenpuku/sanrentan)
        bet_unit: 1枚あたりの掛金
        rescale_ratio: bankroll制約による縮小比率
    
    Returns:
        検証結果dict
    """
    ordered = ticket_type == 'sanrentan'
    
    # Official winning combinations for this race
    official_winners = set(official_payouts.keys())
    K = len(official_winners)  # Number of winning combinations (usually 1, can be >1 for dead heat)
    
    # Check each ticket
    hits = []
    non_hits = []
    total_payout = 0
    
    for t in tickets:
        # Convert ticket tuple to official key format
        ticket_key = ticket_tuple_to_official_key(t, ticket_type)
        display_str = format_combination_display(list(t), ordered=ordered)
        
        if ticket_key in official_payouts:
            payout = official_payouts[ticket_key] * rescale_ratio
            hits.append({
                'ticket': display_str,
                'ticket_key': ticket_key,
                'official_payout': official_payouts[ticket_key],
                'actual_payout': payout
            })
            total_payout += payout
        else:
            non_hits.append({'ticket': display_str, 'ticket_key': ticket_key, 'payout': 0})
    
    # Calculate expected payout from official table
    expected_payout = sum(
        official_payouts[ticket_tuple_to_official_key(t, ticket_type)] * rescale_ratio 
        for t in tickets 
        if ticket_tuple_to_official_key(t, ticket_type) in official_payouts
    )
    
    # Validation checks
    result = {
        'race_id': race_id,
        'ticket_type': ticket_type,
        'total_tickets': len(tickets),
        'official_winners_K': K,
        'hits_count': len(hits),
        'non_hits_count': len(non_hits),
        'total_payout': total_payout,
        'expected_payout': expected_payout,
        'payout_match': abs(total_payout - expected_payout) < 0.01,
        'hits': hits,
        'official_combinations': list(official_winners),
        'is_dead_heat': K > 1
    }
    
    # Check (1): Official table not empty
    result['has_official_data'] = K > 0
    
    # Check (2): Payout only to winners
    result['no_invalid_payout'] = all(h['payout'] == 0 for h in non_hits)
    
    # Check (3): Total payout matches expected
    result['integrity_passed'] = (
        result['has_official_data'] and 
        result['no_invalid_payout'] and 
        result['payout_match']
    )
    
    return result


def validate_all_races(
    race_results: List[Dict],
    payout_map: Dict,
    ticket_type: str,
    top_n: int,
    df: pd.DataFrame,
    bet_unit: float = 100.0
) -> TicketPayoutResult:
    """
    全レースの払戻整合性を検証 (+全体dead heat集計)
    """
    result = TicketPayoutResult(f"ticket_payout_integrity_{ticket_type}_box{top_n}")
    
    horse_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in df.columns and df['prob'].notna().any() else 'p_market'
    
    ordered = ticket_type == 'sanrentan'
    
    violations = []
    dead_heat_races = []
    missing_payout_races = []
    payout_mismatch_races = []
    
    total_races = 0
    passed_races = 0
    total_k_values = []  # For dead heat statistics
    
    for race_result in race_results:
        race_id = race_result['race_id']
        total_races += 1
        
        # Get official payouts
        if race_id not in payout_map:
            missing_payout_races.append(race_id)
            continue
        
        official_payouts = payout_map[race_id].get(ticket_type, {})
        
        if not official_payouts:
            missing_payout_races.append(race_id)
            continue
        
        K = len(official_payouts)
        total_k_values.append(K)
        
        if K > 1:
            dead_heat_races.append(race_id)
        
        # Regenerate tickets
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df[race_df[score_col].notna()]
        
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        if ticket_type == 'umaren':
            tickets = list(combinations(horse_numbers, 2))
        elif ticket_type == 'sanrenpuku':
            tickets = list(combinations(horse_numbers, 3))
        elif ticket_type == 'sanrentan':
            tickets = list(permutations(horse_numbers, 3))
        else:
            continue
        
        # Validate
        rescale_ratio = race_result.get('rescale_ratio', 1.0)
        check = validate_ticket_payout_integrity(
            race_id, tickets, official_payouts, ticket_type, bet_unit, rescale_ratio
        )
        
        if check['integrity_passed']:
            passed_races += 1
        else:
            if not check['has_official_data']:
                missing_payout_races.append(race_id)
            elif not check['payout_match']:
                payout_mismatch_races.append({
                    'race_id': race_id,
                    'expected': check['expected_payout'],
                    'actual': check['total_payout']
                })
            violations.append(check)
    
    # Summary
    result.details = {
        'total_races': total_races,
        'passed_races': passed_races,
        'missing_payout_races': len(missing_payout_races),
        'payout_mismatch_races': len(payout_mismatch_races),
        'dead_heat_races': len(dead_heat_races),
        'dead_heat_race_ids': dead_heat_races[:20],
        'missing_payout_race_ids': missing_payout_races[:20],
        'payout_mismatch_samples': payout_mismatch_races[:20],
        # New: K value statistics
        'k_max': max(total_k_values) if total_k_values else 0,
        'k_mean': np.mean(total_k_values) if total_k_values else 0,
    }
    
    # Determine pass/fail
    if payout_mismatch_races:
        result.fail(
            f"払戻不一致: {len(payout_mismatch_races)}レース",
            violations=len(payout_mismatch_races)
        )
    elif missing_payout_races and len(missing_payout_races) > total_races * 0.1:
        result.fail(
            f"払戻データ欠損>10%: {len(missing_payout_races)}/{total_races}",
            violations=len(missing_payout_races)
        )
    else:
        msg = f"✅ PASS ({passed_races}/{total_races}レース整合)"
        if dead_heat_races:
            msg += f", 同着{len(dead_heat_races)}件(K>1)"
        result.pass_check(msg)
    
    return result


def generate_ticket_sample_table(
    df: pd.DataFrame,
    payout_map: Dict,
    ticket_type: str,
    top_n: int,
    n_races: int = 20,
    seed: int = 42,
    bet_unit: float = 100.0
) -> Tuple[str, Dict]:
    """
    チケット検証用サンプル表を生成 (フォーマット修正版)
    
    Returns:
        (markdown_str, stats_dict)
    """
    np.random.seed(seed)
    
    horse_col = 'umaban' if 'umaban' in df.columns else 'horse_number'
    score_col = 'prob' if 'prob' in df.columns and df['prob'].notna().any() else 'p_market'
    
    ordered = ticket_type == 'sanrentan'
    
    # Get races with payout data
    valid_races = [r for r in df['race_id'].unique() if r in payout_map and ticket_type in payout_map[r]]
    
    if len(valid_races) == 0:
        return "(No races with payout data)\n", {}
    
    sample_races = np.random.choice(valid_races, min(n_races, len(valid_races)), replace=False)
    
    stats = {
        'total_sampled': len(sample_races),
        'races_with_hits': 0,
        'races_without_hits': 0,
        'dead_heats': 0,
        'total_hit_tickets': 0,
        'total_payout': 0
    }
    
    lines = []
    lines.append(f"# Ticket Payout Integrity Sample ({ticket_type} BOX{top_n})\n\n")
    lines.append(f"**Sampled Races**: {len(sample_races)}\n")
    lines.append(f"**Seed**: {seed}\n\n")
    
    for race_id in sample_races:
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df[race_df[score_col].notna()]
        
        if len(race_df) < top_n:
            continue
        
        top_horses = race_df.nlargest(top_n, score_col)
        horse_numbers = top_horses[horse_col].astype(int).tolist()
        
        if ticket_type == 'umaren':
            tickets = list(combinations(horse_numbers, 2))
        elif ticket_type == 'sanrenpuku':
            tickets = list(combinations(horse_numbers, 3))
        elif ticket_type == 'sanrentan':
            tickets = list(permutations(horse_numbers, 3))
        else:
            continue
        
        official_payouts = payout_map[race_id].get(ticket_type, {})
        K = len(official_payouts)
        
        if K > 1:
            stats['dead_heats'] += 1
        
        # Decode official winners for display
        official_decoded = []
        for code in official_payouts.keys():
            decoded = decode_official_winner(code, ticket_type)
            official_decoded.append(f"{code} → {decoded}")
        
        lines.append(f"## Race: {race_id}\n\n")
        lines.append(f"**Top {top_n} Horses**: {', '.join(map(str, horse_numbers))}\n")
        lines.append(f"**Official Winners (K={K})**:\n")
        for od in official_decoded:
            lines.append(f"  - {od}\n")
        lines.append("\n")
        
        # Show tickets with hit/miss status
        lines.append("| Ticket | Key | Hit | Payout |\n")
        lines.append("|--------|-----|-----|--------|\n")
        
        hit_count = 0
        race_payout = 0
        hit_tickets_display = []
        
        for t in tickets[:12]:  # Limit display
            ticket_key = ticket_tuple_to_official_key(t, ticket_type)
            display_str = format_combination_display(list(t), ordered=ordered)
            is_hit = ticket_key in official_payouts
            payout = official_payouts.get(ticket_key, 0)
            
            hit_mark = "✅" if is_hit else ""
            if is_hit:
                hit_count += 1
                race_payout += payout
                hit_tickets_display.append(f"{display_str} (¥{payout:,})")
            
            lines.append(f"| {display_str} | {ticket_key} | {hit_mark} | ¥{payout:,.0f} |\n")
        
        if len(tickets) > 12:
            # Check remaining tickets for hits
            for t in tickets[12:]:
                ticket_key = ticket_tuple_to_official_key(t, ticket_type)
                if ticket_key in official_payouts:
                    hit_count += 1
                    payout = official_payouts[ticket_key]
                    race_payout += payout
                    display_str = format_combination_display(list(t), ordered=ordered)
                    hit_tickets_display.append(f"{display_str} (¥{payout:,})")
            lines.append(f"| ... ({len(tickets) - 12} more) | | | |\n")
        
        stats['total_hit_tickets'] += hit_count
        stats['total_payout'] += race_payout
        
        if hit_count > 0:
            stats['races_with_hits'] += 1
        else:
            stats['races_without_hits'] += 1
        
        lines.append(f"\n**Hit Count**: {hit_count}/{len(tickets)}")
        if hit_tickets_display:
            lines.append(f" → {', '.join(hit_tickets_display)}")
        lines.append("\n")
        lines.append(f"**Race Payout**: ¥{race_payout:,.0f}\n")
        lines.append("\n---\n\n")
    
    # Summary stats at the beginning
    summary = f"**Races With Hits**: {stats['races_with_hits']}, **Races Without Hits**: {stats['races_without_hits']}, **Dead Heats**: {stats['dead_heats']}, **Total Hit Tickets**: {stats['total_hit_tickets']}, **Total Payout**: ¥{stats['total_payout']:,}\n\n"
    lines.insert(3, summary)
    
    return "".join(lines), stats


def run_phase7_sanity_checks(
    race_results: List[Dict],
    payout_map: Dict,
    ticket_type: str,
    top_n: int,
    df: pd.DataFrame,
    strict: bool = True
) -> List[TicketPayoutResult]:
    """Run all Phase 7 sanity checks"""
    results = []
    
    # Main integrity check
    integrity_result = validate_all_races(
        race_results, payout_map, ticket_type, top_n, df
    )
    results.append(integrity_result)
    
    # Strict mode
    if strict and not integrity_result.passed:
        raise ValueError(
            f"STRICT SANITY FAILED: {integrity_result.check_name}\n"
            f"Message: {integrity_result.message}\n"
            f"Details: {integrity_result.details}"
        )
    
    return results


# =============================================================================
# Unit test helpers
# =============================================================================

def test_ticket_matching():
    """Basic unit tests for ticket matching"""
    # Test decode
    assert decode_official_winner("041205", "sanrenpuku") == (4, 5, 12)  # sorted
    assert decode_official_winner("041205", "sanrentan") == (4, 12, 5)   # ordered
    assert decode_official_winner("0102", "umaren") == (1, 2)
    
    # Test format
    assert format_combination_padded([4, 12, 5], ordered=False) == "040512"  # sorted
    assert format_combination_padded([4, 12, 5], ordered=True) == "041205"   # ordered
    
    # Test key generation
    assert ticket_tuple_to_official_key((4, 12, 5), "sanrenpuku") == "040512"  # sorted
    assert ticket_tuple_to_official_key((4, 12, 5), "sanrentan") == "041205"   # ordered
    
    print("✅ All ticket matching tests passed")


if __name__ == "__main__":
    test_ticket_matching()
