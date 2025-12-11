"""
最適戦略モジュール (v5_2025 Optimized)
2025年データを基に最適化されたグリッドサーチ結果に基づく推奨

推奨構成 (ROI > 100% & Stable):
1. midrange (4-6番人気): sanrentan_6 (group_top2fix) -> ROI 227.6%
2. top3_dominant: sanrentan_6 (weighted_0.7) -> ROI 212.3%
3. balanced: umaren_3 (score_then_ev_5) -> ROI 166.4%
4. small_gap: sanrentan_6 (score) -> ROI 146.3%
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import combinations, permutations
import numpy as np
from scipy.special import softmax


@dataclass
class BettingRecommendation:
    """買い目推奨"""
    strategy_name: str
    bet_type: str  # 'sanrentan', 'sanrenpuku', 'umaren', 'skip'
    formation: str  # 説明
    tickets: List[Tuple[int, ...]]  # 買い目リスト
    expected_roi: float  # 期待ROI
    confidence: str  # 'high', 'medium', 'low', 'warning'
    message: str
    ranking_method: str = "score"  # 使用したランキング方法
    selected_horses: List[int] = None  # 並べ替え後の馬リスト (Top6程度)


class OptimalStrategy:
    """
    最適戦略クラス (v5_2025 Optimized)
    
    2025年最適化結果:
    - midrange: sanrentan_6 (group_top2fix) → ROI 227.6%
    - top3_dominant: sanrentan_6 (weighted_0.7) → ROI 212.3%
    - balanced: umaren_3 (score_then_ev_5) → ROI 166.4%
    - small_gap: sanrentan_6 (score) → ROI 146.3%
    """
    
    def analyze_race(
        self,
        horse_numbers: List[int],
        scores: List[float],
        popularities: List[int],
        odds: List[float],
        probs: List[float] = None
    ) -> BettingRecommendation:
        """
        Option C戦略 (2025年v7最適化):
        - 7番人気以上 → 三連単1頭軸4頭 (ROI 1057%)
        - 接戦(gap<0.3) → 三連単1頭軸4頭 (ROI 138%)
        - その他 → 単勝 (安定回収)
        """
        if len(horse_numbers) < 6:
            return BettingRecommendation(
                strategy_name="skip",
                bet_type="skip",
                formation="出走頭数不足",
                tickets=[],
                expected_roi=0,
                confidence="warning",
                message="⚠️ 出走頭数が6頭未満のため見送り"
            )
        
        # 確率とEVを計算
        if probs is None:
            probs = list(softmax(scores))
        
        evs = [p * o if o > 0 else 0 for p, o in zip(probs, odds)]
        
        # Top1馬の人気
        top1_pop = popularities[0] if popularities else 99
        
        # スコア差 (Top1 - Top6)
        score_gap = scores[0] - scores[5] if len(scores) >= 6 else 0.5
        
        # Option C戦略の適用 (2025年最適化: 穴狙いのみ)
        # 1. 予測1位が4番人気以上 (Pop>=4) → 三連単1頭軸4頭 (12点) ROI 125%
        # 2. 予測1位が1-3番人気 (Pop<4) → 見送り ROI < 100% (Pattern B Gap<0.05もQ2+でROI 70%のため不採用)
        
        if top1_pop >= 4:
            # 4番人気以上 → 三連単1頭軸4頭
            # 相手は予測2-5位 (1-axis, 4-opps = 12 tickets)
            # Note: _strategy_sanrentan_4 logic uses permutations(opps, 2), which creates 12 tickets for 4 opps.
            # opps should be horses[1:5] (4 horses)
            return self._strategy_sanrentan_4(horse_numbers, scores, evs, "穴馬狙い", 125.5)
        else:
            # その他(人気馬) → 見送り
            return self._strategy_skip(horse_numbers, scores, evs, "人気サイドのため見送り", "top1_pop < 4")
    
    def _strategy_skip(self, h: List[int], s: List[float], e: List[float], reason: str, code: str) -> BettingRecommendation:
        """見送り"""
        return BettingRecommendation(
            strategy_name=f"見送り ({code})",
            bet_type="skip",
            formation=f"見送り: {reason}",
            tickets=[],
            expected_roi=0.0,
            confidence="low",
            message=f"⚠️ {reason}",
            ranking_method="score",
            selected_horses=h[:6]
        )
    
    def _classify_race(self, scores: List[float], popularities: List[int]) -> str:
        """レース分類 (現在は使用しないが互換性のために残す)"""
        return 'all'
    
    # ========================================
    # ランキング方法
    # ========================================
    
    def _rerank_group_top2fix(self, h: List[int], s: List[float], e: List[float]) -> List[int]:
        """Top2固定、3-6をEVでre-rank"""
        # index付きで管理
        items = list(zip(h, s, e))
        
        top2 = items[:2]
        # 3位~6位(index 2~5)をEVでソート
        middle = sorted(items[2:6], key=lambda x: x[2], reverse=True)
        rest = items[6:]
        
        new_order = top2 + middle + rest
        return [x[0] for x in new_order]

    def _rerank_weighted_07(self, h: List[int], s: List[float], e: List[float]) -> List[int]:
        """Weighted 0.7 (Score*0.7 + EV*0.3) でre-rank"""
        s_min, s_max = min(s), max(s)
        e_min, e_max = min(e), max(e)
        
        items = []
        for i in range(len(h)):
            s_norm = (s[i] - s_min) / (s_max - s_min) if s_max > s_min else 0.5
            e_norm = (e[i] - e_min) / (e_max - e_min) if e_max > e_min else 0.5
            w = 0.7 * s_norm + 0.3 * e_norm
            items.append((h[i], w))
            
        # weightedスコアでソート
        items.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in items]

    def _rerank_score_then_ev_5(self, h: List[int], s: List[float], e: List[float]) -> List[int]:
        """Top5をEV順に並び替え"""
        items = list(zip(h, s, e))
        pool = items[:5]
        pool_sorted = sorted(pool, key=lambda x: x[2], reverse=True)
        rest = items[5:]
        new_order = pool_sorted + rest
        return [x[0] for x in new_order]

    # ========================================
    # 戦略実装
    # ========================================

    def _strategy_base_umaren(self, h: List[int], s: List[float], e: List[float]) -> BettingRecommendation:
        """全レース推奨: 馬連流し 3点 (score_then_ev_5) ROI 92.0%"""
        reranked = self._rerank_score_then_ev_5(h, s, e)
        axis = reranked[0]
        opps = reranked[1:4] # 相手3頭
        
        tickets = []
        for opp in opps:
             pair = tuple(sorted((axis, opp)))
             tickets.append(pair)
             
        return BettingRecommendation(
            strategy_name="全レース推奨 (Base Strategy)",
            bet_type="umaren",
            formation=f"馬連 流し: {axis}-{opps} (3点)",
            tickets=tickets,
            expected_roi=92.0,
            confidence="medium",
            message="🛡️ 堅実運用 (Base 92%) - 馬連3点",
            ranking_method="score_then_ev_5",
            selected_horses=reranked[:6]
        )

    def _strategy_midrange(self, h: List[int], s: List[float], e: List[float], pop: int) -> BettingRecommendation:
        """midrange: sanrentan_6 (group_top2fix)"""
        reranked = self._rerank_group_top2fix(h, s, e)
        axis = reranked[0]
        opps = reranked[1:4]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
        return BettingRecommendation(
            strategy_name="中穴狙い (Optimized)",
            bet_type="sanrentan",
            formation=f"3連単 1頭軸マルチなし: {axis}→{opps} (6点)",
            tickets=tickets,
            expected_roi=227.6,
            confidence="high",
            message=f"✨ 中穴チャンス (Top1={pop}人) - 3連単6点 (Top2固定EV)",
            ranking_method="group_top2fix",
            selected_horses=reranked[:6]
        )

    def _strategy_top3_dominant(self, h: List[int], s: List[float], e: List[float]) -> BettingRecommendation:
        """top3_dominant: sanrentan_6 (weighted_0.7)"""
        reranked = self._rerank_weighted_07(h, s, e)
        axis = reranked[0]
        opps = reranked[1:4]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
        return BettingRecommendation(
            strategy_name="Top3優勢 (Optimized)",
            bet_type="sanrentan",
            formation=f"3連単 1頭軸マルチなし: {axis}→{opps} (6点)",
            tickets=tickets,
            expected_roi=212.3,
            confidence="high",
            message=f"📈 Top3優勢 - 3連単6点 (Weighted)",
            ranking_method="weighted_0.7",
            selected_horses=reranked[:6]
        )
        
    def _strategy_balanced(self, h: List[int], s: List[float], e: List[float]) -> BettingRecommendation:
        """balanced: umaren_3 (score_then_ev_5)"""
        reranked = self._rerank_score_then_ev_5(h, s, e)
        axis = reranked[0]
        opps = reranked[1:4] # 相手3頭
        
        tickets = []
        for opp in opps:
            pair = tuple(sorted((axis, opp)))
            tickets.append(pair)
            
        return BettingRecommendation(
            strategy_name="混戦レース (Optimized)",
            bet_type="umaren",
            formation=f"馬連 流し: {axis}-{opps} (3点)",
            tickets=tickets,
            expected_roi=166.4,
            confidence="medium",
            message=f"⚡ 混戦模様 - 馬連3点 (Top5EV)",
            ranking_method="score_then_ev_5",
            selected_horses=reranked[:6]
        )

    def _strategy_small_gap(self, h: List[int], s: List[float], e: List[float]) -> BettingRecommendation:
        """small_gap: sanrentan_6 (score - normal ranking)"""
        # score順そのまま
        axis = h[0]
        opps = h[1:4]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
        return BettingRecommendation(
            strategy_name="小差レース (Optimized)",
            bet_type="sanrentan",
            formation=f"3連単 1頭軸マルチなし: {axis}→{opps} (6点)",
            tickets=tickets,
            expected_roi=146.3,
            confidence="medium",
            message=f"📊 小差レース - 3連単6点 (Score)",
            ranking_method="score",
            selected_horses=h[:6]
        )
    
    # ========================================
    # Option C戦略用メソッド
    # ========================================
    
    def _strategy_sanrentan_4(self, h: List[int], s: List[float], e: List[float], 
                              strategy_name: str, expected_roi: float) -> BettingRecommendation:
        """Option C: 三連単1頭軸4頭 (6点)"""
        axis = h[0]
        opps = h[1:4]  # Top2-4
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
        return BettingRecommendation(
            strategy_name=f"{strategy_name} (Option C)",
            bet_type="sanrentan",
            formation=f"3連単 1頭軸: {axis}→{opps} (6点)",
            tickets=tickets,
            expected_roi=expected_roi,
            confidence="high" if expected_roi > 100 else "medium",
            message=f"🎯 {strategy_name} - 3連単6点",
            ranking_method="score",
            selected_horses=h[:6]
        )
    
    def _strategy_tansho(self, h: List[int], s: List[float], e: List[float]) -> BettingRecommendation:
        """Option C: 単勝 (その他条件)"""
        axis = h[0]
        tickets = [(axis,)]
        
        return BettingRecommendation(
            strategy_name="安定運用 (Option C)",
            bet_type="tansho",
            formation=f"単勝: {axis}番 (1点)",
            tickets=tickets,
            expected_roi=82.0,  # 2025年単勝平均
            confidence="medium",
            message=f"🛡️ 安定運用 - 単勝1点",
            ranking_method="score",
            selected_horses=h[:6]
        )
        
    def format_notification(self, rec: BettingRecommendation, race_info: dict = None) -> str:
        """
        通知用メッセージをフォーマット (Mobile Friendly & Minimal)
        """
        lines = []
        
        # タイトル (レース名など)
        if race_info:
            # 🏇 東京 11R
            # 📍 天皇賞(秋)
            lines.append(f"🏇 {race_info.get('venue', '')} {race_info.get('race_number', '')}R")
            lines.append(f"📍 **{race_info.get('title', '')}**")
            lines.append("")
        
        # 戦略名だけシンプルに
        # lines.append(rec.message) # 削除: 詳細メッセージは不要
        
        # 買い目セクション
        if rec.bet_type != "skip" and rec.tickets:
            # 【UMAREN】
            # 馬連 流し: 2-(5, 1, 4)
            lines.append(f"**【{rec.bet_type.upper()}】**")
            lines.append(rec.formation)
            lines.append("")
            
            # 個別買い目リストは省略 (Formationで十分なため)
            
        elif rec.bet_type == "skip":
             lines.append("⚠️ 見送り")
        
        return "\n".join(lines)
