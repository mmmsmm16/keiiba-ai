"""
2025 Backtest with AI Rules (v23)

AIが発掘したルールセット(final_rules_v23.json)を用いて、
2025年のレースデータに対するバックテストを実行する。
モデルは v23 (Fold4) を使用する。
"""
import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, text
from tqdm import tqdm
from scipy.special import softmax

# パス設定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src')) # Fix import path

# 必要なモジュールをインポート
# ※ run_ytd_simulation.py と同様に inference モジュールを活用
from inference.loader import InferenceDataLoader
from inference.preprocessor import InferencePreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----- Helper Functions -----

def load_rules(rule_path):
    with open(rule_path, 'r') as f:
        return json.load(f)

def get_2025_dates(loader):
    """2025年の開催日を取得 (JRAのみ)"""
    try:
        # JRAの競馬場コード (01-10) に絞る
        query = text("SELECT DISTINCT kaisai_nen, kaisai_tsukihi FROM jvd_ra WHERE kaisai_nen = '2025' AND keibajo_code >= '01' AND keibajo_code <= '10' ORDER BY 1, 2")
        with loader.engine.connect() as conn:
            result = conn.execute(query)
            dates = []
            for row in result:
                nen = row[0]
                md = row[1]
                dt_str = f"{nen}-{md[:2]}-{md[2:]}"
                dates.append(dt_str)
            return dates
    except Exception as e:
        logger.error(f"Failed to fetch dates: {e}")
        return []

def load_payout_map_for_race(race_id, loader):
    """レースIDに対する払い戻し情報をDBから取得して辞書化"""
    query = text("SELECT * FROM jvd_hr WHERE kaisai_nen = :nen AND keibajo_code = :jo AND kaisai_kai = :kai AND kaisai_nichime = :nichi AND race_bango = :ban")
    # race_id parsing: 4(nen)+2(jo)+2(kai)+2(nichi)+2(ban)
    nen = race_id[:4]
    jo = race_id[4:6]
    kai = race_id[6:8]
    nichi = race_id[8:10]
    ban = race_id[10:12]
    
    with loader.engine.connect() as conn:
        result = conn.execute(query, {"nen": nen, "jo": jo, "kai": kai, "nichi": nichi, "ban": ban}).fetchone()
        
    if not result: return None
    
    # Row to dict mapping (simplified)
    # result is a Row object, can be accessed by column name
    row = result._mapping
    
    pm = {'tansho': {}, 'umaren': {}, 'umatan': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}}
    for k_base in ['tansho', 'umaren', 'umatan', 'wide', 'sanrenpuku', 'sanrentan']:
        for i in range(1, 8): # Limit to typical range
            k, v = f'haraimodoshi_{k_base}_{i}a', f'haraimodoshi_{k_base}_{i}b'
            if k in row and row[k]:
                try:
                    key = str(row[k])
                    if k_base == 'tansho': key = key.zfill(2)
                    pm[k_base][key] = int(row[v])
                except: pass
    return pm

# ベット候補生成 (共通化すべきだが独立動作のためコピー)
from itertools import combinations, permutations
def generate_bets(top_horses, n_horses):
    bets = {}
    bets['tansho_top1'] = ([f"{top_horses[0]:02}"], 100)
    if len(top_horses) >= 3:
        bets['umaren_box3'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:3], 2)], 300)
    if len(top_horses) >= 5:
        bets['umaren_box5'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:5], 2)], 1000)
        bets['umaren_nagashi'] = ([f"{min(top_horses[0],b):02}{max(top_horses[0],b):02}" for b in top_horses[1:5]], 400)
        hits = [f"{min(a,b):02}{max(a,b):02}" for a in top_horses[:2] for b in top_horses[2:5]]
        bets['umaren_form'] = (hits, len(hits)*100)
    if len(top_horses) >= 5:
        bets['wide_box5'] = ([f"{min(a,b):02}{max(a,b):02}" for a,b in combinations(top_horses[:5], 2)], 1000)
        bets['wide_nagashi'] = ([f"{min(top_horses[0],b):02}{max(top_horses[0],b):02}" for b in top_horses[1:5]], 400)
        hits = [f"{min(a,b):02}{max(a,b):02}" for a in top_horses[:2] for b in top_horses[2:5]]
        bets['wide_form'] = (hits, len(hits)*100)
    if len(top_horses) >= 5:
        bets['umatan_1st'] = ([f"{top_horses[0]:02}{b:02}" for b in top_horses[1:5]], 400)
        bets['umatan_2nd'] = ([f"{b:02}{top_horses[0]:02}" for b in top_horses[1:4]], 300)
        bets['umatan_box3'] = ([f"{a:02}{b:02}" for a,b in permutations(top_horses[:3], 2)], 600)
    if len(top_horses) >= 6:
        bets['sanrenpuku_box5'] = ([f"{''.join(sorted([f'{x:02}' for x in c]))}" for c in combinations(top_horses[:5], 3)], 1000)
        bets['sanrenpuku_nagashi'] = ([f"{''.join(sorted([f'{top_horses[0]:02}', f'{b:02}', f'{c:02}']))}" for b,c in combinations(top_horses[1:5], 2)], 600)
        hits = []
        for h in top_horses[2:6]:
            code = sorted([f"{top_horses[0]:02}", f"{top_horses[1]:02}", f"{h:02}"])
            hits.append("".join(code))
        bets['sanrenpuku_form'] = (hits, len(hits)*100)
    if len(top_horses) >= 5:
        bets['sanrentan_1st'] = ([f"{top_horses[0]:02}{b:02}{c:02}" for b,c in permutations(top_horses[1:5], 2)], 1200)
        bets['sanrentan_box3'] = ([f"{a:02}{b:02}{c:02}" for a,b,c in permutations(top_horses[:3], 3)], 600)
    return bets

def apply_rules(row, rules):
    bets_to_make = []
    for rule in rules:
        match = True
        for feat, op, thres in rule['conditions']:
            val = row[feat]
            if op == '<=':
                if not (val <= thres):
                    match = False
                    break
            else:
                if not (val > thres):
                    match = False
                    break
        if match:
            bets_to_make.append(rule['bet_name'])
    return list(set(bets_to_make))

# ----- Main Execution -----

def main():
    logger.info("Setting up 2025 Backtest...")
    
    # 1. Load Rules
    cv_dir = os.path.join(project_root, 'experiments', 'v23_regression_cv')
    rule_path = os.path.join(cv_dir, 'final_rules_v23.json')
    rules = load_rules(rule_path)
    logger.info(f"Loaded {len(rules)} rules.")
    
    # 2. Load Models (Fold4)
    model_dir = os.path.join(cv_dir, 'fold4')
    with open(os.path.join(model_dir, 'lgbm_v23.pkl'), 'rb') as f: lgbm = pickle.load(f)
    with open(os.path.join(model_dir, 'catboost_v23.pkl'), 'rb') as f: catboost = pickle.load(f)
    with open(os.path.join(model_dir, 'meta_v23.pkl'), 'rb') as f: meta = pickle.load(f)
    logger.info("Models loaded (v23 fold4).")
    
    # 3. Setup Data Loader
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    
    # Load history for preprocessing (Leakage-aware)
    history_path = os.path.join(project_root, 'data/processed/preprocessed_data_v10_leakfix.parquet')
    if os.path.exists(history_path):
        history_df = pd.read_parquet(history_path)
    else:
        logger.error("History data not found.")
        return
        
    dates = get_2025_dates(loader)
    logger.info(f"Processing {len(dates)} days in 2025 (Full JRA)...")
    # dates = dates[:5] # Limit removed
    
    # Load pickle to get feature names
    dataset_path = os.path.join(project_root, 'data/processed/lgbm_datasets_v10_leakfix.pkl')
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    feature_cols = datasets['train']['X'].columns.tolist()
    
    total_bet = 0
    total_ret = 0
    results = []
    
    for date_str in tqdm(dates):
        flat_date = date_str.replace('-', '')
        raw_df = loader.load(target_date=flat_date)
        if raw_df.empty: continue
            
        # Preprocess using CURRENT history
        # history_df is updated every loop
        try:
             # return_full_df=True to get dataframe for history update
             X, ids, full_processed_df = preprocessor.preprocess(raw_df, history_df=history_df, return_full_df=True)
        except Exception as e:
             logger.error(f"Preprocess error at {date_str}: {e}")
             continue
             
        if X.empty: continue
        
        # Ensure features
        proc_df = pd.concat([ids, X], axis=1)
        # Deduplicate
        proc_df = proc_df.loc[:, ~proc_df.columns.duplicated()]
        
        missing = set(feature_cols) - set(proc_df.columns)
        for c in missing: proc_df[c] = 0
        X_pred = proc_df[feature_cols]
        
        # Predict
        lgbm_pred = lgbm.predict(X_pred)
        cb_pred = catboost.predict(X_pred)
        meta_score = meta.predict(np.column_stack([lgbm_pred, cb_pred]))
        proc_df['score'] = meta_score
        
        # --- UPDATE HISTORY FOR NEXT DAY ---
        # 予測に使った full_processed_df (特徴量付き) に、実際の結果(rank, time等)をマージして履歴に追加する
        # raw_dfには結果が入っている
        
        # Restore result columns from raw_df
        # full_processed_df may have dropped 'rank', 'last_3f' etc.
        # We need to map them back from raw_df using race_id + horse_number
        
        # Create key
        raw_df['key'] = raw_df['race_id'].astype(str) + '_' + raw_df['horse_number'].astype(str)
        full_processed_df['key'] = full_processed_df['race_id'].astype(str) + '_' + full_processed_df['horse_number'].astype(str)
        
        # Columns to restore for history
        restore_cols = ['rank', 'last_3f', 'time', 'popularity', 'odds'] # Add others if needed by aggregators
        
        for col in restore_cols:
            if col in raw_df.columns:
                mapper = raw_df.set_index('key')[col].to_dict()
                full_processed_df[col] = full_processed_df['key'].map(mapper)
        
        full_processed_df.drop('key', axis=1, inplace=True)
        
        # Append to history_df
        # history_df columns may differ slightly (full_processed_df has extra features)
        # We should align columns or just concat (pandas handles NaN)
        # To avoid memory explosion, maybe drop unnecessary columns? 
        # But features are needed for lag calculation next time.
        
        history_df = pd.concat([history_df, full_processed_df], axis=0, ignore_index=True)
        
        # Memory management (optional): keep only last 1-2 years?
        # But aggregators might need long history.
        # For 1 year loop, it should be fine.
        
        # -----------------------------------
        
        # Race Loop using rules
        for race_id, group in proc_df.groupby('race_id'):
            # Feature Extraction for Rules
            sorted_group = group.nlargest(6, 'score')
            top_horses = sorted_group['horse_number'].astype(int).tolist()
            top_scores = sorted_group['score'].tolist()
            if len(top_horses) < 3: continue
            
            top3_odds = sorted_group['odds'].head(3).tolist() if 'odds' in sorted_group.columns else []
            avg_top3_odds = np.mean([o for o in top3_odds if not pd.isna(o)]) if top3_odds else 0
            all_scores = group['score'].tolist()
            score_conc = sum(top_scores[:3]) / sum(all_scores) if sum(all_scores) > 0 else 0
            
            # Metadata
            # Use proc_df (group) for available features, fallback to parsing race_id
            
            try:
                # venue_code: race_id[4:6] is safest
                # 'K6' などの不正ID (地方競馬?) が混じる場合があるため try-except
                venue_code = int(race_id[4:6])
            except ValueError:
                continue

            # Filter JRA only (01-10)
            # ユーザー要望によりNRA除外
            if not (1 <= venue_code <= 10):
                continue
            
            # distance
            try:
                dist = float(group['distance'].iloc[0]) if 'distance' in group.columns else 1600.0
            except:
                dist = 1600.0

            # surface
            surf = 0
            if 'surface' in group.columns:
                val = group['surface'].iloc[0]
                try:
                    val_int = int(val)
                    surf = val_int - 1 if val_int > 0 else 0
                except:
                    surf = 0 # Default to 0 if parse fails
            
            row_features = {
                'score_gap': top_scores[0] - top_scores[1] if len(top_scores) >= 2 else 0,
                'top1_odds': sorted_group['odds'].iloc[0] if 'odds' in sorted_group.columns else 0,
                'avg_top3_odds': avg_top3_odds,
                'score_conc': score_conc,
                'n_horses': len(group),
                'distance': dist,
                'surface': surf,
                'venue': venue_code - 1, # 01->0 ... 
                'month': pd.to_datetime(date_str).month,
            }
            
            target_bets = apply_rules(row_features, rules)
            if not target_bets: continue
            
            # Payout
            pm = load_payout_map_for_race(race_id, loader)
            if not pm: continue
            
            bets = generate_bets(top_horses, len(group))
            
            r_bet = 0
            r_ret = 0
            
            for bname in target_bets:
                if bname not in bets: continue
                codes, cost = bets[bname]
                ptype = bname.split('_')[0]
                
                win = 0
                if ptype in pm:
                    payouts = pm[ptype]
                    for c in codes:
                        if c in payouts: win += payouts[c]
                        
                r_bet += cost
                r_ret += win
                
            total_bet += r_bet
            total_ret += r_ret
            results.append({'date': date_str, 'race_id': race_id, 'bet': r_bet, 'return': r_ret})
            
    # Summary
    logger.info("=== 2025 Backtest Results ===")
    roi = total_ret / total_bet * 100 if total_bet > 0 else 0
    profit = total_ret - total_bet
    logger.info(f"Total ROI: {roi:.1f}%")
    logger.info(f"Total Profit: {profit:,}")
    logger.info(f"Total Bet: {total_bet:,}")
    
    # Save CSV
    pd.DataFrame(results).to_csv('reports/backtest/backtest_2025_v23_results.csv', index=False)
    logger.info("Saved detailed results to reports/backtest/backtest_2025_v23_results.csv")

if __name__ == "__main__":
    main()
