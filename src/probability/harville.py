
import itertools
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class HarvilleProbability:
    """
    Implements Harville's formula (Plackett-Luce model) to estimate
    probabilities of complex outcomes from win probabilities.
    """
    
    @staticmethod
    def expand_probabilities(win_probs: Dict[int, float], bet_type: str, limit_horses: int = None) -> Dict[Tuple[int, ...], float]:
        """
        Expand win probabilities to combination probabilities.
        
        Args:
            win_probs: Dict mapping horse_number to win probability (sum should differ from 1.0 slightly due to track take or normalization).
                       It is recommended validation is done outside.
            bet_type: 'umaren', 'wide', 'umatan', 'sanrenpuku', 'sanrentan'.
            limit_horses: If provided, only consider top N horses by probability for expansion (to save time).
            
        Returns:
            Dict mapping tuple of selections to probability.
        """
        # Sort and limit if requested
        items = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
        if limit_horses:
            items = items[:limit_horses]
            
        # Re-normalize restricted set? 
        # Standard Harville uses original probs. If we limit, we just ignore tail combinations.
        # We should NOT re-normalize sum to 1.0 because we are estimating specific outcome probabilities within the global event space.
        
        horses = [h for h, p in items]
        probs = {h: p for h, p in items}
        
        results = {}
        
        if bet_type == 'umatan':
            for i in horses:
                pi = probs[i]
                denom = 1.0 - pi
                if denom <= 0: continue
                for j in horses:
                    if i == j: continue
                    pj = probs[j]
                    p_ij = pi * (pj / denom)
                    results[(i, j)] = p_ij

        elif bet_type == 'umaren':
            # Sum of permutations i-j and j-i
            # We iterate combinations to avoid double counting keys
            for i, j in itertools.combinations(sorted(horses), 2):
                if i not in probs or j not in probs: continue
                pi, pj = probs[i], probs[j]
                
                # P(i->j)
                p_ij = pi * (pj / (1.0 - pi))
                # P(j->i)
                p_ji = pj * (pi / (1.0 - pj))
                
                results[(i, j)] = p_ij + p_ji

        elif bet_type == 'sanrentan':
            for i in horses:
                pi = probs[i]
                d1 = 1.0 - pi
                if d1 <= 0: continue
                for j in horses:
                    if i == j: continue
                    pj = probs[j]
                    d2 = d1 - pj
                    if d2 <= 0: continue
                    
                    p_ij = pi * (pj / d1)
                    
                    for k in horses:
                        if k == i or k == j: continue
                        pk = probs[k]
                        p_ijk = p_ij * (pk / d2)
                        results[(i, j, k)] = p_ijk

        elif bet_type == 'sanrenpuku':
            # Sum of 6 permutations
            # Iterate combinations
            for i, j, k in itertools.combinations(sorted(horses), 3):
                if i not in probs or j not in probs or k not in probs: continue
                
                # Calculate sum of all 6 permutations
                # Optimization: reuse common terms or just bruteforce 6 calls
                perm_sum = 0
                for p_seq in itertools.permutations((i, j, k)):
                     h1, h2, h3 = p_seq
                     p1, p2, p3 = probs[h1], probs[h2], probs[h3]
                     # P(h1->h2->h3)
                     val = p1 * (p2 / (1.0 - p1)) * (p3 / (1.0 - p1 - p2))
                     perm_sum += val
                
                results[(i, j, k)] = perm_sum

        elif bet_type == 'wide':
            # Wide P(i,j) = P(i,j,*) + P(j,i,*) + P(i,*,j) + P(j,*,i) + P(*,i,j) + P(*,j,i)
            # This is relatively heavy ($N^3$).
            # We assume 'horses' includes all horses relevant.
            # If 'horses' is limited, the "x" (3rd horse) summation will be incomplete!
            # CRITICAL: For Wide, 'limit_horses' logic is dangerous if we don't sum over ALL horses for the 'x' part.
            # But normally we accept approximation if limit is high enough (e.g. top 10).
            # Or we can iterate Main Pair (i,j) from Limited Set, but sum 'x' from Full Set.
            
            # Using full set for 'x'
            full_horses = list(win_probs.keys())
            
            # Helper to calculate WIDE prob for pair (i,j)
            # Pre-calculate simple exactas?
            # P(i->j) = pi * pj / (1-pi)
            
            memo_exacta = {}
            
            def get_exacta(h1, h2):
                if (h1, h2) in memo_exacta: return memo_exacta[(h1,h2)]
                p1, p2 = win_probs[h1], win_probs[h2]
                val = p1 * (p2 / (1.0 - p1))
                memo_exacta[(h1,h2)] = val
                return val

            # Main loop: Combinations of i, j from valid candidates
            # We use 'horses' (limited) for the tickets we want to generate.
            # But the summation logic requires valid probabilities for 'x'.
            
            for i, j in itertools.combinations(sorted(horses), 2):
                pi, pj = win_probs[i], win_probs[j]
                
                p_total = 0
                
                # 1. i-j-*
                p_total += get_exacta(i, j)
                
                # 2. j-i-*
                p_total += get_exacta(j, i)
                
                # Cases 3-6: Need summation over x
                # 3. i-*-j
                term3 = 0
                d1 = 1.0 - pi
                for x in full_horses:
                    if x == i or x == j: continue
                    px = win_probs[x]
                    # P(i, x, j) = pi * (px / d1) * (pj / (d1 - px))
                    term3 += pi * (px / d1) * (pj / (d1 - px))
                p_total += term3
                
                # 4. j-*-i
                term4 = 0
                d1j = 1.0 - pj
                for x in full_horses:
                    if x == i or x == j: continue
                    px = win_probs[x]
                    term4 += pj * (px / d1j) * (pi / (d1j - px))
                p_total += term4
                
                # 5. *-i-j and 6. *-j-i
                # Combine loops
                term56 = 0
                for x in full_horses:
                    if x == i or x == j: continue
                    px = win_probs[x]
                    d1x = 1.0 - px
                    # P(x, i, j)
                    term56 += px * (pi / d1x) * (pj / (d1x - pi))
                    # P(x, j, i)
                    term56 += px * (pj / d1x) * (pi / (d1x - pj))
                p_total += term56
                
                results[(i, j)] = p_total

        return results
