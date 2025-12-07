import sys
import os
import pickle
import pandas as pd
import numpy as np
import logging
import json
import argparse
from datetime import datetime
from scipy.special import softmax

from sqlalchemy import create_engine, text

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.ensemble import EnsembleModel
from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payout_data(year=2024):
    logger.info(f"払戻データ(jvd_hr)をロード中... Year={year}")
    engine = get_db_engine()
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen = '{year}'")
    try:
        df = pd.read_sql(query, engine)
        
        # race_id の構築 (jvd_raと同じロジック: YYYY+Place+Kai+Day+Race)
        # jvd_hr columns: kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango
        # values are usually strings like '2024', '05', '01', '01', '01'
        df['race_id'] = (
            df['kaisai_nen'].astype(str) +
            df['keibajo_code'].astype(str) +
            df['kaisai_kai'].astype(str) +
            df['kaisai_nichime'].astype(str) +
            df['race_bango'].astype(str)
        )
        logger.info(f"払戻データロード完了: {len(df)} 件")
        return df
    except Exception as e:
        logger.error(f"払戻データロードエラー: {e}")
        return pd.DataFrame()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description='モデル評価スクリプト')
    parser.add_argument('--model', type=str, default='ensemble', choices=['lgbm', 'catboost', 'tabnet', 'ensemble'], help='評価するモデル')
    parser.add_argument('--version', type=str, default='v1', help='モデルバージョン')
    args = parser.parse_args()

    # 1. データのロード (Parquetから元データを取得)
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/preprocessed_data.parquet')
    if not os.path.exists(data_path):
        logger.error(f"データファイルがありません: {data_path}")
        return

    # テストデータ(2024年)のみロード
    df = pd.read_parquet(data_path)
    test_df = df[df['year'] == 2024].copy()

    if test_df.empty:
        logger.error("テストデータ(2024年)がありません。")
        return

    # 2. モデルのロード
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    model = None
    
    logger.info(f"評価開始: Model={args.model}, Version={args.version}")

    try:
        if args.model == 'ensemble':
            model = EnsembleModel()
            path = os.path.join(model_dir, f'ensemble_{args.version}.pkl')
            if not os.path.exists(path):
                 # Fallback
                 if args.version == 'v1' and os.path.exists(os.path.join(model_dir, 'ensemble_model.pkl')):
                     path = os.path.join(model_dir, 'ensemble_model.pkl')
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
            model.load_model(path)

        elif args.model == 'lgbm':
            model = KeibaLGBM()
            path = os.path.join(model_dir, f'lgbm_{args.version}.pkl')
            if not os.path.exists(path):
                 if args.version == 'v1' and os.path.exists(os.path.join(model_dir, 'lgbm.pkl')):
                     path = os.path.join(model_dir, 'lgbm.pkl')
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
            model.load_model(path)

        elif args.model == 'catboost':
            model = KeibaCatBoost()
            path = os.path.join(model_dir, f'catboost_{args.version}.pkl')
            if not os.path.exists(path):
                 if args.version == 'v1' and os.path.exists(os.path.join(model_dir, 'catboost.pkl')):
                     path = os.path.join(model_dir, 'catboost.pkl')
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
            model.load_model(path)

        elif args.model == 'tabnet':
            model = KeibaTabNet()
            path = os.path.join(model_dir, f'tabnet_{args.version}.zip')
            if not os.path.exists(path):
                 if args.version == 'v1' and os.path.exists(os.path.join(model_dir, 'tabnet.zip')):
                     path = os.path.join(model_dir, 'tabnet.zip')
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
            model.load_model(path.replace('.zip', '.pkl'))
            
    except Exception as e:
        logger.error(f"モデルロードエラー: {e}")
        return

    # 3. 特徴量の整合性確認と予測
    feature_cols = None
    
    # モデルから特徴量リストを取得（互換性維持のため）
    if args.model == 'lgbm' and hasattr(model.model, 'feature_name'):
        feature_cols = model.model.feature_name()
    elif args.model == 'catboost' and hasattr(model.model, 'feature_names_'):
        feature_cols = model.model.feature_names_
        
    # モデルから取得できない場合は最新の学習データセット情報を使用
    if feature_cols is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
        with open(dataset_path, 'rb') as f:
            datasets = pickle.load(f)

        if datasets['train']['X'] is None:
            logger.error("学習データの特徴量情報がありません。")
            return

        feature_cols = datasets['train']['X'].columns.tolist()
    
    # テストデータにカラムが存在するか確認
    missing_cols = set(feature_cols) - set(test_df.columns)
    if missing_cols:
        logger.warning(f"不足しているカラムがあります（0で埋めます）: {missing_cols}")
        for c in missing_cols:
            test_df[c] = 0

    X_test = test_df[feature_cols]

    logger.info("予測を実行中...")
    scores = model.predict(X_test)
    test_df['score'] = scores

    # 確率と期待値の計算
    logger.info("勝率と期待値を計算中...")
    # レースごとにSoftmax
    test_df['prob'] = test_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    # 期待値 = 確率 * オッズ (欠損は0)
    test_df['expected_value'] = test_df['prob'] * test_df['odds'].fillna(0)

    # 4. シミュレーションと保存
    output_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
    os.makedirs(output_dir, exist_ok=True)
    
    simulation_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'version': args.version,
        'strategies': {},
        'roi_curve': []
    }

    # Strategy 1: レース内で最も期待値が高い馬を1点買い
    logger.info("--- Simulation: Max Expected Value (1点買い) ---")
    sim_ev = simulate_single_choice(test_df, 'expected_value')
    simulation_results['strategies']['max_ev'] = sim_ev
    logger.info(f"Max EV - ROI: {sim_ev['roi']:.2f}%, Hit: {sim_ev['accuracy']:.2%}")

    # Strategy 2: レース内で最もスコアが高い馬を1点買い
    logger.info("--- Simulation: Max Score (1点買い) ---")
    sim_score = simulate_single_choice(test_df, 'score')
    simulation_results['strategies']['max_score'] = sim_score
    logger.info(f"Max Score - ROI: {sim_score['roi']:.2f}%, Hit: {sim_score['accuracy']:.2%}")

    # Strategy 3: 期待値閾値ごとのROIカーブ
    # 期待値が X 以上の馬を全て買う（単勝）
    logger.info("--- Simulation: EV Thresholds (ROI Curve) ---")
    curve_data = simulate_threshold_curve(test_df)
    simulation_results['roi_curve'] = curve_data

    # Strategy 4: 複合馬券シミュレーション (Box 5)
    # n_races が多いので少し時間がかかる可能性あり
    payout_df = load_payout_data(year=2024)
    if not payout_df.empty:
        sim_complex = simulate_complex_betting(test_df, payout_df)
        simulation_results['strategies'].update(sim_complex)
    else:
        logger.warning("払戻データがロードできなかったため、複合馬券シミュレーションをスキップします。")

    # Save to JSON
    # バージョンごとにファイルを分けるか？
    # Dashboard reads 'latest_simulation.json'. Let's keep it that way for 'latest', but maybe save a copy.
    json_path = os.path.join(output_dir, 'latest_simulation.json')
    history_json_path = os.path.join(output_dir, f'simulation_{args.model}_{args.version}.json')
    
    with open(json_path, 'w') as f:
        json.dump(simulation_results, f, indent=4, cls=NpEncoder)
    with open(history_json_path, 'w') as f:
        json.dump(simulation_results, f, indent=4, cls=NpEncoder)
        
    logger.info(f"シミュレーション結果を保存しました: {json_path}")


def simulate_single_choice(df, target_col):
    results = []
    
    for race_id, group in df.groupby('race_id'):
        if group[target_col].isnull().all():
            continue

        best_idx = group[target_col].idxmax()
        best_horse = group.loc[best_idx]
        
        actual_rank = best_horse['rank']
        odds = best_horse['odds']
        
        bet = 100
        return_amount = odds * 100 if actual_rank == 1 else 0
        
        results.append({'bet': bet, 'return': return_amount, 'hit': 1 if actual_rank == 1 else 0})
        
    sim_df = pd.DataFrame(results)
    if sim_df.empty:
        return {'roi': 0, 'accuracy': 0, 'total_bet': 0, 'total_return': 0}
        
    total_bet = sim_df['bet'].sum()
    total_return = sim_df['return'].sum()
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    accuracy = sim_df['hit'].mean()
    
    return {'roi': roi, 'accuracy': accuracy, 'total_bet': total_bet, 'total_return': total_return}

def simulate_threshold_curve(df):
    target_col = 'expected_value'
    # 閾値を0.5刻みなどで設定
    # 期待値は prob * odds。prob約0.1 * odds10 = 1.0 (等倍)。
    # 0.5 (回収率50%期待) から 2.0 (回収率200%期待) くらいまでスキャン
    thresholds = np.arange(0.5, 3.0, 0.1)
    
    curve_data = []
    
    for th in thresholds:
        # 閾値を超える条件の行を抽出
        bets = df[df[target_col] >= th].copy()
        
        if bets.empty:
            curve_data.append({
                'threshold': th,
                'roi': 0,
                'bet_count': 0,
                'accuracy': 0
            })
            continue
            
        bets['bet_amount'] = 100
        bets['return_amount'] = bets.apply(lambda row: row['odds'] * 100 if row['rank'] == 1 else 0, axis=1)
        
        total_bet = bets['bet_amount'].sum()
        total_return = bets['return_amount'].sum()
        roi = total_return / total_bet * 100 if total_bet > 0 else 0
        bet_count = len(bets)
        accuracy = (bets['rank'] == 1).mean()
        
        curve_data.append({
            'threshold': round(th, 2),
            'roi': roi,
            'bet_count': bet_count,
            'accuracy': accuracy
        })
        
    return curve_data

def simulate_complex_betting(df, payout_df):
    from itertools import combinations, permutations
    
    logger.info("--- Simulation: Complex Betting (Box 5) ---")
    
    # Payoutデータを検索しやすいように辞書化
    # race_id -> { 'umaren': {comb: pay, ...}, 'sanrenpuku': ..., 'sanrentan': ... }
    payout_map = {}
    
    # 必要な列だけ抽出して反復
    # haraimodoshi_umaren_1a, 1b, 2a, 2b, 3a, 3b... max usually 3?
    # JRA-VAN doc says up to 3 entries for simple bets usually.
    # We will check 1 to 3.
    
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        # Umaren (1-3)
        for i in range(1, 4):
            k_comb = f'haraimodoshi_umaren_{i}a'
            k_pay = f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['umaren'][str(row[k_comb])] = pay
                except: pass

        # Sanrenpuku (1-3)
        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrenpuku_{i}a'
            k_pay = f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrenpuku'][str(row[k_comb])] = pay
                except: pass

        # Sanrentan (1-6? usually 3, but up to 6 in huge dead heat)
        for i in range(1, 7):
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrentan'][str(row[k_comb])] = pay
                except: pass

    # Strategies
    stats = {
        'umaren_box5': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'sanrenpuku_box5': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'sanrentan_box5': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
    }
    
    # 予測データでループ
    race_groups = df.groupby('race_id')
    for race_id, group in race_groups:
        if race_id not in payout_map:
            continue
            
        # Top 5 horses by score
        top5 = group.sort_values('score', ascending=False).head(5)
        if len(top5) < 2:
            continue
            
        horse_nums = top5['horse_number'].astype(int).tolist() # [1, 5, 2, ...]
        
        # --- Umaren Box 5 (10点) ---
        # 組み合わせ (順序なし) -> 小さい順に並べて文字列化 (0102)
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(horse_nums) >= 2:
            combos = list(combinations(horse_nums, 2)) # (1, 5), (1, 2)...
            bet_count = len(combos) # 10 if 5 horses
            
            for c in combos:
                # ソートして文字列化 (0205)
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                
                if comb_str in payout_map[race_id]['umaren']:
                    return_amount += payout_map[race_id]['umaren'][comb_str]
                    hit = 1
            
            stats['umaren_box5']['bet'] += bet_count * 100
            stats['umaren_box5']['return'] += return_amount
            stats['umaren_box5']['hit'] += hit
            stats['umaren_box5']['races'] += 1

        # --- Sanrenpuku Box 5 (10点) ---
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(horse_nums) >= 3:
            combos = list(combinations(horse_nums, 3))
            bet_count = len(combos) # 10
            
            for c in combos:
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                
                if comb_str in payout_map[race_id]['sanrenpuku']:
                    return_amount += payout_map[race_id]['sanrenpuku'][comb_str]
                    hit = 1
                    
            stats['sanrenpuku_box5']['bet'] += bet_count * 100
            stats['sanrenpuku_box5']['return'] += return_amount
            stats['sanrenpuku_box5']['hit'] += hit
            stats['sanrenpuku_box5']['races'] += 1

        # --- Sanrentan Box 5 (60点) ---
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(horse_nums) >= 3: # 3連単も3頭以上必要
            # Boxなので順列 (Permutations)
            perms = list(permutations(horse_nums, 3))
            bet_count = len(perms) # 60
            
            for p in perms:
                # 順序通り文字列化
                perm_str = f"{p[0]:02}{p[1]:02}{p[2]:02}"
                
                if perm_str in payout_map[race_id]['sanrentan']:
                    return_amount += payout_map[race_id]['sanrentan'][perm_str]
                    hit = 1
            
            stats['sanrentan_box5']['bet'] += bet_count * 100
            stats['sanrentan_box5']['return'] += return_amount
            stats['sanrentan_box5']['hit'] += hit
            stats['sanrentan_box5']['races'] += 1

    # 集計結果作成
    final_results = {}
    for k, v in stats.items():
        roi = v['return'] / v['bet'] * 100 if v['bet'] > 0 else 0
        accuracy = v['hit'] / v['races'] if v['races'] > 0 else 0
        final_results[k] = {
            'roi': roi, 
            'accuracy': accuracy, 
            'bet': v['bet'], 
            'return': v['return'],
            'races': v['races']
        }
        logger.info(f"{k} - ROI: {roi:.2f}%, Hit: {accuracy:.2%} ({v['races']} races)")
        
    return final_results

if __name__ == "__main__":
    main()
