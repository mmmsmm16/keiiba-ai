"""
バランス型戦略シミュレーション
- 目標: ROI 150%程度を維持しつつベット数を増やす
- 条件を緩和して対象レース数を拡大
"""
import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import logging
from scipy.special import softmax
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_predictions_from_db(years=[2024, 2025]):
    data_path = 'data/processed/preprocessed_data.parquet'
    df = pd.read_parquet(data_path)
    df = df[df['year'].isin(years)].copy()
    logger.info(f"Loaded {len(df)} rows for years {years}")
    return df

def load_model_and_predict(df, model_name='ensemble', version='v4_2025'):
    sys.path.append('src')
    from model.ensemble import EnsembleModel
    
    model = EnsembleModel()
    model.load_model(f'models/ensemble_{version}.pkl')
    
    import pickle
    with open('data/processed/lgbm_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    
    X = df[feature_cols]
    scores = model.predict(X)
    df['score'] = scores
    
    return df

def load_payouts(years=[2024, 2025]):
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen IN ({years_str})")
    df = pd.read_sql(query, engine)
    
    df['race_id'] = (
        df['kaisai_nen'].astype(str) +
        df['keibajo_code'].astype(str) +
        df['kaisai_kai'].astype(str) +
        df['kaisai_nichime'].astype(str) +
        df['race_bango'].astype(str)
    )
    return df

def build_payout_map(pay_df):
    payout_map = defaultdict(lambda: {'tansho': {}, 'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}})
    
    for _, row in pay_df.iterrows():
        rid = row['race_id']
        
        for prefix, max_count in [('haraimodoshi_tansho', 3), ('haraimodoshi_umaren', 3), 
                                   ('haraimodoshi_wide', 7), ('haraimodoshi_sanrenpuku', 3), 
                                   ('haraimodoshi_sanrentan', 6)]:
            bet_type = prefix.split('_')[1]
            for i in range(1, max_count + 1):
                col_a = f'{prefix}_{i}a'
                col_b = f'{prefix}_{i}b'
                if col_a in row and row[col_a] and str(row[col_a]).strip():
                    try:
                        key = str(row[col_a]).strip()
                        val = int(float(str(row[col_b]).strip()))
                        payout_map[rid][bet_type][key] = val
                    except:
                        pass
    
    return dict(payout_map)

def preprocess_data(df):
    df = df.copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    df['prob'] = df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    df['ev'] = df['prob'] * df['odds'].fillna(0)
    
    return df

def simulate_balanced_strategy(df, payout_map):
    """
    バランス型戦略シミュレーション
    ROI 150%程度を維持しつつベット数を増やす
    """
    print("\n" + "="*80)
    print("📊 バランス型戦略シミュレーション (ROI 150%目標)")
    print("="*80)
    
    # レースごとのTop1と馬番リストを事前計算
    race_top1 = df[df['pred_rank'] == 1].set_index('race_id')[['prob', 'ev', 'odds', 'rank', 'horse_number']].to_dict('index')
    
    race_horses = {}
    for rid, grp in df.groupby('race_id'):
        sorted_g = grp.sort_values('score', ascending=False)
        race_horses[rid] = sorted_g['horse_number'].astype(int).tolist()
    
    total_races = len(race_top1)
    
    # 集計
    stats = {
        'sanrentan_mid': {'races': 0, 'cost': 0, 'return': 0, 'hits': 0, 'desc': '3連単(オッズ3-10)'},
        'tansho_longshot': {'races': 0, 'cost': 0, 'return': 0, 'hits': 0, 'desc': '単勝(オッズ10-50)'},
        'sanrenpuku_medium': {'races': 0, 'cost': 0, 'return': 0, 'hits': 0, 'desc': '3連複(オッズ10-20)'},
        'umaren_ev': {'races': 0, 'cost': 0, 'return': 0, 'hits': 0, 'desc': '馬連(EV≥1.2, オッズ10+)'},
        'tansho_all_longshot': {'races': 0, 'cost': 0, 'return': 0, 'hits': 0, 'desc': '単勝(オッズ10+全体)'},
        'skip': {'races': 0},
    }
    
    for rid, top1 in race_top1.items():
        if rid not in payout_map:
            stats['skip']['races'] += 1
            continue
        
        odds = top1['odds'] if not pd.isna(top1['odds']) else 0
        ev = top1['ev']
        actual_rank = top1['rank']
        h_nums = race_horses.get(rid, [])
        
        # 優先順位で戦略を判定 (バランス型: 条件を緩和)
        strategy = None
        
        # 条件1: オッズ3-10倍 & EV >= 1.0 (緩和) → 3連単ながし5頭
        if 3 <= odds < 10 and ev >= 1.0:
            strategy = 'sanrentan_mid'
            if len(h_nums) >= 6:
                axis = h_nums[0]
                opps = h_nums[1:6]
                tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
                race_cost = len(tickets) * 100
                race_ret = 0
                hit_flag = 0
                
                for t in tickets:
                    key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                    if key in payout_map[rid]['sanrentan']:
                        race_ret += payout_map[rid]['sanrentan'][key]
                        hit_flag = 1
                
                stats['sanrentan_mid']['races'] += 1
                stats['sanrentan_mid']['cost'] += race_cost
                stats['sanrentan_mid']['return'] += race_ret
                stats['sanrentan_mid']['hits'] += hit_flag
            else:
                stats['skip']['races'] += 1
                
        # 条件2: オッズ10-50倍 → 単勝
        elif 10 <= odds <= 50:
            strategy = 'tansho_all_longshot'
            race_cost = 100
            race_ret = odds * 100 if actual_rank == 1 else 0
            
            stats['tansho_all_longshot']['races'] += 1
            stats['tansho_all_longshot']['cost'] += race_cost
            stats['tansho_all_longshot']['return'] += race_ret
            stats['tansho_all_longshot']['hits'] += 1 if actual_rank == 1 else 0
            
        # 条件3: オッズ50倍以上 & EV >= 1.0 → 単勝（大穴）
        elif odds > 50 and ev >= 1.0:
            strategy = 'tansho_longshot'
            race_cost = 100
            race_ret = odds * 100 if actual_rank == 1 else 0
            
            stats['tansho_longshot']['races'] += 1
            stats['tansho_longshot']['cost'] += race_cost
            stats['tansho_longshot']['return'] += race_ret
            stats['tansho_longshot']['hits'] += 1 if actual_rank == 1 else 0
            
        else:
            # 見送り
            stats['skip']['races'] += 1
    
    # 結果表示
    print(f"\n【全 {total_races} レース中の内訳】\n")
    
    bet_races = 0
    total_cost = 0
    total_return = 0
    total_hits = 0
    
    for name, s in stats.items():
        if name == 'skip':
            continue
        
        if s['races'] > 0:
            roi = s['return'] / s['cost'] * 100 if s['cost'] > 0 else 0
            hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
            profit = s['return'] - s['cost']
            
            print(f"📍 {s['desc']}:")
            print(f"   対象レース: {s['races']} 件")
            print(f"   投資額: ¥{s['cost']:,}")
            print(f"   払戻: ¥{s['return']:,.0f}")
            print(f"   利益: ¥{profit:+,.0f}")
            print(f"   ROI: {roi:.1f}%, 的中率: {hit_rate:.1f}%")
            print()
            
            bet_races += s['races']
            total_cost += s['cost']
            total_return += s['return']
            total_hits += s['hits']
    
    skip_races = stats['skip']['races']
    
    print("-" * 60)
    print(f"\n🎯 【総合結果】")
    print(f"   全レース数: {total_races}")
    print(f"   ベット対象: {bet_races} レース ({bet_races/total_races*100:.1f}%)")
    print(f"   見送り: {skip_races} レース ({skip_races/total_races*100:.1f}%)")
    print()
    print(f"   💰 総投資額: ¥{total_cost:,}")
    print(f"   💰 総払戻: ¥{total_return:,.0f}")
    print(f"   💰 総利益: ¥{total_return - total_cost:+,.0f}")
    print()
    
    if total_cost > 0:
        combined_roi = total_return / total_cost * 100
        combined_hit_rate = total_hits / bet_races * 100 if bet_races > 0 else 0
        print(f"   📈 総合ROI: {combined_roi:.1f}%")
        print(f"   📈 総合的中率: {combined_hit_rate:.1f}%")
        
        # 週あたりの目安
        weeks = 104  # 約2年分
        bets_per_week = bet_races / weeks
        print(f"\n   📅 週あたりベット数(目安): {bets_per_week:.1f} レース")
        
        # 結果をファイルに保存
        with open('reports/balanced_strategy_result.txt', 'w', encoding='utf-8') as f:
            f.write("=== バランス型戦略シミュレーション結果 ===\n\n")
            f.write(f"全レース数: {total_races}\n")
            f.write(f"ベット対象: {bet_races} レース ({bet_races/total_races*100:.1f}%)\n")
            f.write(f"見送り: {skip_races} レース\n\n")
            f.write(f"総投資額: ¥{total_cost:,}\n")
            f.write(f"総払戻: ¥{total_return:,.0f}\n")
            f.write(f"総利益: ¥{total_return - total_cost:+,.0f}\n\n")
            f.write(f"総合ROI: {combined_roi:.1f}%\n")
            f.write(f"総合的中率: {combined_hit_rate:.1f}%\n")
            f.write(f"週あたりベット数: {bets_per_week:.1f} レース\n\n")
            
            f.write("--- 戦略別内訳 ---\n")
            for name, s in stats.items():
                if name != 'skip' and s['races'] > 0:
                    roi = s['return'] / s['cost'] * 100 if s['cost'] > 0 else 0
                    hit_rate = s['hits'] / s['races'] * 100
                    f.write(f"{s['desc']}: {s['races']}レース, ROI {roi:.1f}%, Hit {hit_rate:.1f}%\n")
        
        print("\n結果を reports/balanced_strategy_result.txt に保存しました")
    
    print("\n" + "="*80)

def main():
    print("\n" + "#"*80)
    print("# 📊 バランス型戦略 シミュレーション (2024+2025年)")
    print("# 目標: ROI 150%程度、ベット数増加")
    print("#"*80)
    
    years = [2024, 2025]
    
    df = load_predictions_from_db(years)
    if df is None:
        return
    
    df = load_model_and_predict(df, 'ensemble', 'v4_2025')
    df = preprocess_data(df)
    
    pay_df = load_payouts(years)
    payout_map = build_payout_map(pay_df)
    logger.info(f"Built payout map for {len(payout_map)} races")
    
    simulate_balanced_strategy(df, payout_map)
    
    print("\n✅ シミュレーション完了!")

if __name__ == "__main__":
    main()
