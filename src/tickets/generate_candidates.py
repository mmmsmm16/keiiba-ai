import pandas as pd
import numpy as np
import itertools

def generate_candidates(race_id, prob_dict, ticket_config=None):
    """
    Generate betting candidates for a single race.
    
    Args:
        race_id (str): Race ID
        prob_dict (dict): Output from ticket_probabilities.map_probs
                          Keys: 'win', 'place', 'umaren', 'wakuren'
                          Values: Series or DataFrame
        ticket_config (dict): Configuration for generation (e.g. topK).
                              Example: {'umaren': {'strategy': 'box', 'top_k': 5}}
                              
    Returns:
        pd.DataFrame: Columns [race_id, ticket_type, combination, p_ticket]
                      combination is a string key (e.g. "1-2") or tuple
    """
    candidates = []
    
    # Defaults
    if ticket_config is None:
        ticket_config = {}
        
    # --- Win ---
    if 'win' in prob_dict:
        s = prob_dict['win']
        # Generate all or top K? Usually all single tickets are fine.
        for horse, prob in s.items():
            candidates.append({
                'race_id': race_id,
                'ticket_type': 'win',
                'combination': str(horse),
                'p_ticket': prob
            })
            
    # --- Place ---
    if 'place' in prob_dict:
        s = prob_dict['place']
        for horse, prob in s.items():
            candidates.append({
                'race_id': race_id,
                'ticket_type': 'place',
                'combination': str(horse),
                'p_ticket': prob
            })
            
    # --- Umaren (Quinella) ---
    if 'umaren' in prob_dict:
        df_probs = prob_dict['umaren'] # Matrix
        # Strategy: Box Top K, or All pairs?
        # N=18 -> 153 pairs. Manageable.
        # But for optimization speed, maybe trim very low prob ones?
        # Let's keep all for now (cleaner) or apply prob threshold.
        # Let's use a small prob threshold (e.g. 0.001) to prune junk.
        
        horses = df_probs.index.tolist()
        n = len(horses)
        for i in range(n):
            for j in range(i + 1, n):
                h1, h2 = horses[i], horses[j]
                prob = df_probs.iloc[i, j]
                if prob < 0.0001: continue # Prune extremely rare
                
                candidates.append({
                    'race_id': race_id,
                    'ticket_type': 'umaren',
                    'combination': f"{h1}-{h2}",
                    'p_ticket': prob
                })
                
    # --- Wakuren (Bracket Quinella) ---
    if 'wakuren' in prob_dict:
        df_probs = prob_dict['wakuren']
        # Matrix 1-8
        frames = df_probs.index.tolist()
        n = len(frames)
        for i in range(n):
            for j in range(i, n): # Includes diagonal (same frame)
                f1, f2 = frames[i], frames[j]
                prob = df_probs.iloc[i, j]
                if prob < 0.0001: continue
                
                candidates.append({
                    'race_id': race_id,
                    'ticket_type': 'wakuren',
                    'combination': f"{f1}-{f2}",
                    'p_ticket': prob
                })
                
    # Future: Wide, Umatan, etc.
    
    return pd.DataFrame(candidates)
