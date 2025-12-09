"""
芝レース限定 馬券戦略シミュレーション (v5 & v4 Model)
"""
import pandas as pd
import numpy as np
import logging
import os
import sys
import itertools
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.inference.loader import InferenceDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_predictions(path):
    df = pd.read_csv(path)
    df['race_id'] = df['race_id'].astype(str)
    
    # If rank is missing (v4), try to merge from v5 file (assuming it exists)
    if 'rank' not in df.columns:
        v5_path = 'reports/predictions_v5_2025.csv'
        if os.path.exists(v5_path):
            v5_df = pd.read_csv(v5_path)
            v5_df['race_id'] = v5_df['race_id'].astype(str)
            actuals = v5_df[['race_id', 'horse_number', 'rank']].drop_duplicates()
            df = pd.merge(df, actuals, on=['race_id', 'horse_number'], how='left', suffixes=('', '_actual'))
            if 'rank' not in df.columns and 'rank_actual' in df.columns:
                 df['rank'] = df['rank_actual']
    
    return df

def fetch_payouts(race_ids):
    logger.info("Fetching Payout Data from jvd_hr...")
    loader = InferenceDataLoader()
    
    chunk_size = 1000
    payout_list = []
    
    unique_ids = list(set(race_ids))
    
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i:i+chunk_size]
        ids_str = ",".join([f"'{rid}'" for rid in chunk])
        
        query = text(f"""
        SELECT
            CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) AS race_id,
            haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
            haraimodoshi_umaren_2a, haraimodoshi_umaren_2b,
            haraimodoshi_wide_1a, haraimodoshi_wide_1b,
            haraimodoshi_wide_2a, haraimodoshi_wide_2b,
            haraimodoshi_wide_3a, haraimodoshi_wide_3b,
            haraimodoshi_wide_4a, haraimodoshi_wide_4b,
            haraimodoshi_wide_5a, haraimodoshi_wide_5b,
            haraimodoshi_wide_6a, haraimodoshi_wide_6b,
            haraimodoshi_wide_7a, haraimodoshi_wide_7b,
            haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b,
            haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b,
            haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
            haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b,
            haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b,
            haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
            haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b,
            haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b,
            haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
        FROM jvd_hr
        WHERE CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) IN ({ids_str})
        """)
        
        try:
            with loader.engine.connect() as conn:
                tmp = pd.read_sql(query, conn)
                payout_list.append(tmp)
        except Exception as e:
            logger.error(f"Error fetching payouts for chunk {i}: {e}")

    payout_map = {}
    if payout_list:
        payout_df = pd.concat(payout_list)
        for _, row in payout_df.iterrows():
            rid = row['race_id']
            if rid not in payout_map: 
                payout_map[rid] = {'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}}
            
            def parse_pay(prefix, count):
                d = {}
                for k in range(1, count+1):
                    comb = row.get(f'{prefix}_{k}a')
                    pay = row.get(f'{prefix}_{k}b')
                    if comb and pay and str(pay).strip():
                        try:
                            d[str(comb).strip()] = int(float(str(pay).strip()))
                        except: pass
                return d

            payout_map[rid]['umaren'] = parse_pay('haraimodoshi_umaren', 3)
            payout_map[rid]['wide'] = parse_pay('haraimodoshi_wide', 7)
            payout_map[rid]['sanrenpuku'] = parse_pay('haraimodoshi_sanrenpuku', 3)
            payout_map[rid]['sanrentan'] = parse_pay('haraimodoshi_sanrentan', 6)
            
    return payout_map

# ==========================================
# Helper Functions (Defined before usage)
# ==========================================

def gen_12_12_flow(top):
    """◎〇→◎〇→△△△△ (1-2位が1-2着、3着は3-6位)"""
    if len(top) < 6: return []
    tickets = []
    for first in top[:2]:
        for second in top[:2]:
            if first == second: continue
            for third in top[2:6]:
                tickets.append((first, second, third))
    return tickets

def gen_12_flow_12(top):
    """◎〇→△△△△→◎〇 (1-2位が1着と3着、2着は3-6位)"""
    if len(top) < 6: return []
    tickets = []
    for first in top[:2]:
        for second in top[2:6]:
            for third in top[:2]:
                if first == third: continue
                tickets.append((first, second, third))
    return tickets

def gen_gap_strategy(top):
    """ギャップ: 1→3-6→2-6 (予測2位を3着候補にのみ入れる)"""
    if len(top) < 6: return []
    r1 = top[0]
    r2 = top[1]
    tickets = []
    for second in top[2:6]:
        for third in top[1:6]:
            if second != third:
                tickets.append((r1, second, third))
    return tickets

def gen_formation_12_123_1236(top):
    """1,2 -> 1,2,3 -> 1,2,3-6 (24点 max)"""
    if len(top) < 6: return []
    firsts = top[:2]
    seconds = top[:3]
    thirds = top[:6]
    tickets = []
    
    for f in firsts:
        for s in seconds:
            if f == s: continue
            for t in thirds:
                if t == f or t == s: continue
                tickets.append((f, s, t))
    return tickets

def gen_12_12_all(top):
    """1,2 -> 1,2 -> 全通り (Top 10)"""
    limit = 10
    pool = top[:min(len(top), limit)]
    
    tickets = []
    for f in top[:2]:
        for s in top[:2]:
            if f == s: continue
            for t in pool:
                if t == f or t == s: continue
                tickets.append((f, s, t))
    return tickets

def gen_12_head_broad(top):
    """1st: [1,2] -> 2nd: [1-6] -> 3rd: [1-6]"""
    if len(top) < 6: return []
    firsts = top[:2]
    others = top[:6]
    tickets = []
    for f in firsts:
        for s in others:
            if f == s: continue
            for t in others:
                if t == f or t == s: continue
                tickets.append((f, s, t))
    return tickets

def gen_12_head_gap_broad(top):
    """1st: [1,2] -> 2nd: [3-6] -> 3rd: [3-6]"""
    if len(top) < 6: return []
    firsts = top[:2]
    others = top[2:6]
    tickets = []
    for f in firsts:
        for s in others:
            for t in others:
                if s == t: continue
                tickets.append((f, s, t))
    return tickets

# ==========================================
# Main Simulation Logic
# ==========================================

def simulate_strategies(df, model_name="v5"):
    logger.info(f"Starting Turf-Only Betting Simulation for {model_name}...")
    
    # Filter for Turf only
    if 'surface' in df.columns:
        df = df[df['surface'] == '芝'].copy()
        logger.info(f"Turf races: {len(df['race_id'].unique())}")
    
    # Cleaning
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # Calculate Pred Rank
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # Fetch Payouts
    payout_map = fetch_payouts(df['race_id'].unique().tolist())
    logger.info(f"Loaded payouts for {len(payout_map)} races.")

    # Group by race
    races = df.groupby('race_id')
    
    # Strategies Definitions
    strategies = [
        # ===== 馬連 =====
        ('Umaren', '1-2 一点', 'static', lambda r1, r2, top, diff: [sorted([r1, r2])]),
        ('Umaren', '1-2 (差<0.10)', 'cond', lambda r1, r2, top, diff: [sorted([r1, r2])] if diff < 0.10 else []),
        ('Umaren', 'BOX 1-2-3 (3点)', 'static', lambda r1, r2, top, diff: list(itertools.combinations(top[:3], 2))),
        
        # ===== ワイド =====
        ('Wide', 'BOX 1-2-3 (3点)', 'static', lambda r1, r2, top, diff: list(itertools.combinations(top[:3], 2))),
        
        # ===== 3連複 =====
        ('SanRenPuku', '1-2-3 一点', 'static', lambda r1, r2, top, diff: [tuple(sorted(top[:3]))]),
        ('SanRenPuku', '1-2軸 3-6流し (4点)', 'static', lambda r1, r2, top, diff: [tuple(sorted([r1, r2, top[i]])) for i in range(2, min(6, len(top)))]),
        
        # ===== 3連単 (既存) =====
        ('SanRenTan', '1→2→3 一点', 'static', lambda r1, r2, top, diff: [(r1, r2, top[2])] if len(top)>2 else []),
        ('SanRenTan', '1-2-3 マルチ (6点)', 'static', lambda r1, r2, top, diff: list(itertools.permutations(top[:3], 3))),
        ('SanRenTan', '1固定→2-6流し (20点)', 'static', lambda r1, r2, top, diff: [(r1, x, y) for x, y in itertools.permutations(top[1:6], 2)]),
        
        # ギャップ戦略 (BEST PERFORMER)
        ('SanRenTan', 'ギャップ 1→3-6→2-6 (16点)', 'static', lambda r1, r2, top, diff: gen_gap_strategy(top)),
        
        # ===== 3連単 追加戦略 (Rank 1,2 頭) =====
        ('SanRenTan', '2固定→1,3-6流し (20点)', 'static', lambda r1, r2, top, diff: [(r2, x, y) for x, y in itertools.permutations([r1]+top[2:6], 2)]),
        ('SanRenTan', '1,2→1,2,3→1,2,3-6 (Form 24点)', 'static', lambda r1, r2, top, diff: gen_formation_12_123_1236(top)),
        ('SanRenTan', '1,2→1,2→全 (Form 1,2-1,2-All)', 'static', lambda r1, r2, top, diff: gen_12_12_all(top)),
        
        # New: Broad 1,2 Head
        ('SanRenTan', '1,2→1-6→1-6 (40点)', 'static', lambda r1, r2, top, diff: gen_12_head_broad(top)),
        ('SanRenTan', '1,2→3-6→3-6 (24点)', 'static', lambda r1, r2, top, diff: gen_12_head_gap_broad(top)),
    ]
    
    results = []

    for st_type, st_name, st_mode, st_func in strategies:
        total_bet = 0
        total_ret = 0
        total_hit = 0
        race_count = 0
        
        for rid, group in races:
            if rid not in payout_map: continue
            
            sorted_g = group.sort_values('pred_rank')
            if len(sorted_g) < 7: continue
            
            top_nums = sorted_g['horse_number'].astype(int).tolist()
            r1 = top_nums[0]; r2 = top_nums[1]
            
            r1_score = sorted_g.iloc[0]['score']
            r2_score = sorted_g.iloc[1]['score']
            diff = r1_score - r2_score
            
            combos = []
            
            if st_mode == 'adaptive':
                # Not using simple adaptive here for simplicity
                pass
            else:
                try:
                    combos = st_func(r1, r2, top_nums, diff)
                except Exception as e:
                    continue
                 
            if not combos: continue
            
            race_bet = len(combos)
            race_ret = 0
            hit_flag = 0
            
            for c in combos:
                if st_type == 'Umaren':
                     c_s = sorted(c)
                     key = f"{c_s[0]:02}{c_s[1]:02}"
                     if key in payout_map[rid]['umaren']:
                         race_ret += payout_map[rid]['umaren'][key] / 100
                         hit_flag = 1
                         
                elif st_type == 'Wide':
                     c_s = sorted(c)
                     key = f"{c_s[0]:02}{c_s[1]:02}"
                     if key in payout_map[rid]['wide']:
                         race_ret += payout_map[rid]['wide'][key] / 100
                         hit_flag = 1
                         
                elif st_type == 'SanRenPuku':
                     c_s = sorted(c)
                     key = f"{c_s[0]:02}{c_s[1]:02}{c_s[2]:02}"
                     if key in payout_map[rid]['sanrenpuku']:
                         race_ret += payout_map[rid]['sanrenpuku'][key] / 100
                         hit_flag = 1
                         
                elif st_type == 'SanRenTan':
                     key = f"{c[0]:02}{c[1]:02}{c[2]:02}"
                     if key in payout_map[rid]['sanrentan']:
                         race_ret += payout_map[rid]['sanrentan'][key] / 100
                         hit_flag = 1
            
            total_bet += race_bet
            total_ret += race_ret
            if hit_flag: total_hit += 1
            race_count += 1
            
        roi = total_ret / total_bet * 100 if total_bet > 0 else 0
        hit_rate = total_hit / race_count * 100 if race_count > 0 else 0
        avg_c = total_bet / race_count if race_count > 0 else 0
        
        results.append({
            'Model': model_name,
            'type': st_type, 'name': st_name, 'avg_cost': avg_c,
            'hit_rate': hit_rate, 'total_ret': total_ret, 'roi': roi
        })
        
    return results

if __name__ == "__main__":
    models = [
        ('v5 (JRA-Only)', 'reports/predictions_v5_2025.csv'),
        ('v4 (Full)', 'reports/predictions_v4_2025.csv')
    ]
    
    all_results = []
    
    with open('reports/turf_comparison_final.txt', 'w', encoding='utf-8') as f:
        for m_name, m_path in models:
            if os.path.exists(m_path):
                print(f"Processing {m_name}...")
                df = load_predictions(m_path)
                res = simulate_strategies(df, m_name)
                all_results.extend(res)
                
                f.write("\n" + "="*100 + "\n")
                f.write(f"🌿 芝レース限定 馬券戦略シミュレーション結果 ({m_name})\n")
                f.write("="*100 + "\n")
                f.write(f"{'券種':<12} | {'戦略':<40} | {'点数':<6} | {'的中率':<8} | {'回収':<10} | {'ROI':<8}\n")
                f.write("-" * 100 + "\n")
                
                for r in res:
                     f.write(f"{r['type']:<12} | {r['name']:<40} | {r['avg_cost']:<6.1f} | {r['hit_rate']:>6.1f}% | {r['total_ret']:>10.0f} | {r['roi']:>6.1f}%\n")
            else:
                f.write(f"File not found: {m_path}\n")

        # Comparative Summary
        f.write("\n" + "="*100 + "\n")
        f.write("🏆 モデル比較 TOP 10 戦略 (ROI順)\n")
        f.write("="*100 + "\n")
        sorted_res = sorted(all_results, key=lambda x: x['roi'], reverse=True)
        for i, r in enumerate(sorted_res[:20]): # Show top 20
            f.write(f"{i+1}. [{r['Model']}] {r['type']} {r['name']}: ROI {r['roi']:.1f}% (的中率 {r['hit_rate']:.1f}%)\n")
            
    print("Done. Saved to reports/turf_comparison_final.txt")
