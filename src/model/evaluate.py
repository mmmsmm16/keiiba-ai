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
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

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

def load_payout_data(years=[2024]):
    logger.info(f"払戻データ(jvd_hr)をロード中... Years={years}")
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen IN ({years_str})")
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
    parser.add_argument('--years', type=str, default='2024', help='評価対象年 (カンマ区切り, e.g. "2024,2025")')
    parser.add_argument('--dataset_suffix', type=str, default='', help='データセットのサフィックス (例: _v10_leakfix)')
    args = parser.parse_args()

    # 1. データのロード (Parquetから元データを取得)
    data_path = os.path.join(os.path.dirname(__file__), f'../../data/processed/preprocessed_data{args.dataset_suffix}.parquet')
    if not os.path.exists(data_path):
        logger.error(f"データファイルがありません: {data_path}")
        return

    # テストデータ(指定年)のみロード
    target_years = [int(y) for y in args.years.split(',')]
    df = pd.read_parquet(data_path)
    test_df = df[df['year'].isin(target_years)].copy()

    if test_df.empty:
        logger.error(f"テストデータ({target_years})がありません。")
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
        dataset_path = os.path.join(os.path.dirname(__file__), f'../../data/processed/lgbm_datasets{args.dataset_suffix}.pkl')
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
    # [v7] Clip probability to avoid log(0) in metrics
    test_df['prob'] = np.clip(test_df['prob'], 1e-15, 1.0 - 1e-15)
    
    # 期待値 = 確率 * オッズ (欠損は0)
    test_df['expected_value'] = test_df['prob'] * test_df['odds'].fillna(0)

    # --- [v7] Probability Metrics ---
    prob_metrics = calculate_probability_metrics(test_df)
    # ------------------------------

    output_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
    os.makedirs(output_dir, exist_ok=True)

    # Save raw predictions for analysis
    pred_path = os.path.join(output_dir, f'predictions_{args.model}_{args.version}.parquet')
    try:
        test_df.to_parquet(pred_path, index=False)
        logger.info(f"詳細予測データを保存しました: {pred_path}")
    except Exception as e:
        logger.warning(f"予測データの保存に失敗しました: {e}")

    # Save payout data if loaded
    payout_df = load_payout_data(years=target_years)
    if not payout_df.empty:
        payout_path = os.path.join(output_dir, 'payouts_2024.parquet')
        try:
             # Ensure types for parquet
             payout_df.to_parquet(payout_path, index=False)
             logger.info(f"払戻データを保存しました: {payout_path}")
        except Exception as e:
             logger.warning(f"払戻データの保存に失敗しました: {e}")
    
    simulation_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'version': args.version,
        'strategies': {},
        'roi_curve': [],
        'probability_metrics': prob_metrics
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
    payout_df = load_payout_data(years=target_years)
    if not payout_df.empty:
        sim_complex = simulate_complex_betting(test_df, payout_df)
        simulation_results['strategies'].update(sim_complex)
    else:
        logger.warning("払戻データがロードできなかったため、複合馬券シミュレーションをスキップします。")

    # Strategy 5: オッズフィルタリング (穴馬狙い)
    logger.info("--- Simulation: Max EV with Odds Filter ---")
    # 10倍〜50倍 (中穴)
    sim_ev_mid = simulate_single_choice(test_df, 'expected_value', min_odds=10.0, max_odds=50.0)
    simulation_results['strategies']['max_ev_odds_10_50'] = sim_ev_mid
    logger.info(f"Max EV (10-50x) - ROI: {sim_ev_mid['roi']:.2f}%, Hit: {sim_ev_mid['accuracy']:.2%} ({sim_ev_mid['bet_count']} bets)")

    # 50倍以上 (大穴)
    sim_ev_long = simulate_single_choice(test_df, 'expected_value', min_odds=50.0)
    simulation_results['strategies']['max_ev_odds_50plus'] = sim_ev_long
    logger.info(f"Max EV (50x+) - ROI: {sim_ev_long['roi']:.2f}%, Hit: {sim_ev_long['accuracy']:.2%} ({sim_ev_long['bet_count']} bets)")

    # Strategy 6: 流し (Formation) シミュレーション
    if not payout_df.empty:
        sim_formation = simulate_formation_betting(test_df, payout_df)
        simulation_results['strategies'].update(sim_formation)

    # Save to JSON
    json_path = os.path.join(output_dir, 'latest_simulation.json')
    history_json_path = os.path.join(output_dir, f'simulation_{args.model}_{args.version}.json')
    
    with open(json_path, 'w') as f:
        json.dump(simulation_results, f, indent=4, cls=NpEncoder)
    with open(history_json_path, 'w') as f:
        json.dump(simulation_results, f, indent=4, cls=NpEncoder)
        
    logger.info(f"シミュレーション結果を保存しました: {json_path}")

    # Save Raw Predictions to Parquet (for Dashboard)
    try:
        logger.info("予測結果(Parquet)を保存中...")
        race_info = load_race_info(target_years)
        if not race_info.empty:
            # Ensure race_id type matches
            if test_df['race_id'].dtype != race_info['race_id'].dtype:
                 test_df['race_id'] = test_df['race_id'].astype(str)
                 race_info['race_id'] = race_info['race_id'].astype(str)
            
            test_df = pd.merge(test_df, race_info, on='race_id', how='left')
        
        # Filter JRA Only (Venue 01-10)
        # venue column might be '01', '05' etc. Ensure it exists.
        if 'venue' in test_df.columns:
            # Cast to int for comparison if needed, but string comparison works '01' <= x <= '10'
            # Be careful with '10' vs '2'. '01'...'10'.
            # Safer to use list.
            jra_venues = [f"{i:02}" for i in range(1, 11)]
            before_len = len(test_df)
            test_df = test_df[test_df['venue'].isin(jra_venues)]
            logger.info(f"JRAレース(01-10)のみ抽出: {before_len} -> {len(test_df)}")
        
        # Select important columns to save space
        save_cols = [
            'race_id', 'date', 'year', 'month', 'day', 
            'venue', 'race_number', 'hondai', # Race Metadata
            'horse_number', 'horse_name', 
            'odds', 'popularity', 'rank', # Ground Truth
            'score', 'prob', 'expected_value' # Predictions
        ]
        # Only save columns that exist
        save_cols = [c for c in save_cols if c in test_df.columns]
        
        parquet_path = os.path.join(output_dir, f'predictions_{args.model}_{args.version}.parquet')
        test_df[save_cols].to_parquet(parquet_path)
        logger.info(f"予測データを保存(Parquet): {parquet_path}")
    except Exception as e:
        logger.error(f"Parquet保存エラー: {e}")


def simulate_single_choice(df, target_col, min_odds=None, max_odds=None):
    results = []
    
    for race_id, group in df.groupby('race_id'):
        if group[target_col].isnull().all():
            continue

        best_idx = group[target_col].idxmax()
        best_horse = group.loc[best_idx]
        
        odds = best_horse['odds']
        
        # オッズフィルタ
        if min_odds is not None and (pd.isna(odds) or odds < min_odds):
            continue
        if max_odds is not None and (pd.isna(odds) or odds > max_odds):
            continue
        
        actual_rank = best_horse['rank']
        
        bet = 100
        return_amount = odds * 100 if actual_rank == 1 else 0
        
        results.append({'bet': bet, 'return': return_amount, 'hit': 1 if actual_rank == 1 else 0})
        
    sim_df = pd.DataFrame(results)
    if sim_df.empty:
        return {'roi': 0, 'accuracy': 0, 'total_bet': 0, 'total_return': 0, 'bet_count': 0}
        
    total_bet = sim_df['bet'].sum()
    total_return = sim_df['return'].sum()
    roi = total_return / total_bet * 100 if total_bet > 0 else 0
    accuracy = sim_df['hit'].mean()
    
    return {'roi': roi, 'accuracy': accuracy, 'total_bet': total_bet, 'total_return': total_return, 'bet_count': len(sim_df)}

def simulate_threshold_curve(df):
    target_col = 'expected_value'
    thresholds = np.arange(0.5, 3.0, 0.1)
    curve_data = []
    
    for th in thresholds:
        bets = df[df[target_col] >= th].copy()
        
        if bets.empty:
            curve_data.append({'threshold': th, 'roi': 0, 'bet_count': 0, 'accuracy': 0})
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

def simulate_formation_betting(df, payout_df):
    from itertools import combinations, permutations
    
    logger.info("--- Simulation: Formation Betting (Nagashi) ---")
    
    # 既存の payout_map 作成ロジックを流用したいが、関数分割していないので再実装または関数化すべき
    # ここでは冗長だが再実装する
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        # Umaren
        for i in range(1, 4):
            k_comb = f'haraimodoshi_umaren_{i}a'
            k_pay = f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['umaren'][str(row[k_comb])] = pay
                except: pass

        # Sanrenpuku
        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrenpuku_{i}a'
            k_pay = f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrenpuku'][str(row[k_comb])] = pay
                except: pass

        # Sanrentan
        for i in range(1, 7):
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrentan'][str(row[k_comb])] = pay
                except: pass

    stats = {
        'umaren_nagashi': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'sanrenpuku_nagashi': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'sanrentan_nagashi': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
    }
    
    for race_id, group in df.groupby('race_id'):
        if race_id not in payout_map:
            continue
            
        # Top 6 horses by score (Axis:1, Opp:2-6)
        sorted_horses = group.sort_values('score', ascending=False)
        if len(sorted_horses) < 3:
            continue
            
        axis_horse = sorted_horses.iloc[0] # Axis
        opp_horses = sorted_horses.iloc[1:6] # Opponents (Max 5)
        
        axis_num = int(axis_horse['horse_number'])
        opp_nums = opp_horses['horse_number'].astype(int).tolist()
        
        if not opp_nums:
            continue

        # --- Umaren Nagashi (Axis - Opps) ---
        bet_count = 0
        return_amount = 0
        hit = 0
        
        combos = [(axis_num, opp) for opp in opp_nums]  # (1, 2), (1, 3)...
        bet_count = len(combos)
        
        for c in combos:
            c_sorted = sorted(c)
            comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}"
            if comb_str in payout_map[race_id]['umaren']:
                return_amount += payout_map[race_id]['umaren'][comb_str]
                hit = 1
        
        stats['umaren_nagashi']['bet'] += bet_count * 100
        stats['umaren_nagashi']['return'] += return_amount
        stats['umaren_nagashi']['hit'] += hit
        stats['umaren_nagashi']['races'] += 1

        # --- Sanrenpuku Nagashi (Axis 1 head - Opps Combos) ---
        # Axis固定、相手から2頭選ぶ
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(opp_nums) >= 2:
            opp_combos = list(combinations(opp_nums, 2)) # Opponents同士の組み合わせ
            bet_count = len(opp_combos) # 10 if 5 opps
            
            for oc in opp_combos:
                c = (axis_num, oc[0], oc[1])
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                
                if comb_str in payout_map[race_id]['sanrenpuku']:
                    return_amount += payout_map[race_id]['sanrenpuku'][comb_str]
                    hit = 1
        
            stats['sanrenpuku_nagashi']['bet'] += bet_count * 100
            stats['sanrenpuku_nagashi']['return'] += return_amount
            stats['sanrenpuku_nagashi']['hit'] += hit
            stats['sanrenpuku_nagashi']['races'] += 1

        # --- Sanrentan Nagashi (1-head Multi) ---
        # Axis 1着固定、相手が2,3着 (順列)
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(opp_nums) >= 2:
            opp_perms = list(permutations(opp_nums, 2)) # Opponents同士の順列
            bet_count = len(opp_perms) # 20 if 5 opps
            
            for op in opp_perms:
                # 1着: Axis, 2着: op[0], 3着: op[1]
                perm_str = f"{axis_num:02}{op[0]:02}{op[1]:02}"
                
                if perm_str in payout_map[race_id]['sanrentan']:
                    return_amount += payout_map[race_id]['sanrentan'][perm_str]
                    hit = 1
            
            stats['sanrentan_nagashi']['bet'] += bet_count * 100
            stats['sanrentan_nagashi']['return'] += return_amount
            stats['sanrentan_nagashi']['hit'] += hit
            stats['sanrentan_nagashi']['races'] += 1

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

def simulate_complex_betting(df, payout_df):
    from itertools import combinations, permutations
    
    logger.info("--- Simulation: Complex Betting (Box 5) ---")
    
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        payout_map[rid] = {'umaren': {}, 'sanrenpuku': {}, 'sanrentan': {}}
        
        for i in range(1, 4):
            k_comb = f'haraimodoshi_umaren_{i}a'
            k_pay = f'haraimodoshi_umaren_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['umaren'][str(row[k_comb])] = pay
                except: pass

        for i in range(1, 4):
            k_comb = f'haraimodoshi_sanrenpuku_{i}a'
            k_pay = f'haraimodoshi_sanrenpuku_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrenpuku'][str(row[k_comb])] = pay
                except: pass

        for i in range(1, 7):
            k_comb = f'haraimodoshi_sanrentan_{i}a'
            k_pay = f'haraimodoshi_sanrentan_{i}b'
            if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                try:
                    pay = int(row[k_pay])
                    payout_map[rid]['sanrentan'][str(row[k_comb])] = pay
                except: pass

    stats = {
        'umaren_box5': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'sanrenpuku_box5': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'sanrentan_box5': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
    }
    
    for race_id, group in df.groupby('race_id'):
        if race_id not in payout_map:
            continue
            
        top5 = group.sort_values('score', ascending=False).head(5)
        if len(top5) < 2:
            continue
            
        horse_nums = top5['horse_number'].astype(int).tolist()
        
        # --- Umaren Box 5 ---
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(horse_nums) >= 2:
            combos = list(combinations(horse_nums, 2))
            bet_count = len(combos)
            
            for c in combos:
                c_sorted = sorted(c)
                comb_str = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                if comb_str in payout_map[race_id]['umaren']:
                    return_amount += payout_map[race_id]['umaren'][comb_str]
                    hit = 1
            
            stats['umaren_box5']['bet'] += bet_count * 100
            stats['umaren_box5']['return'] += return_amount
            stats['umaren_box5']['hit'] += hit
            stats['umaren_box5']['races'] += 1

        # --- Sanrenpuku Box 5 ---
        bet_count = 0
        return_amount = 0
        hit = 0
        
        if len(horse_nums) >= 3:
            combos = list(combinations(horse_nums, 3))
            bet_count = len(combos)
            
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

        # --- Sanrentan Box 5 ---
        if len(horse_nums) >= 3:
            perms = list(permutations(horse_nums, 3))
            bet_count = len(perms)
            
            for p in perms:
                perm_str = f"{p[0]:02}{p[1]:02}{p[2]:02}"
                if perm_str in payout_map[race_id]['sanrentan']:
                    return_amount += payout_map[race_id]['sanrentan'][perm_str]
                    hit = 1
            
            stats['sanrentan_box5']['bet'] += bet_count * 100
            stats['sanrentan_box5']['return'] += return_amount
            stats['sanrentan_box5']['hit'] += hit
            stats['sanrentan_box5']['races'] += 1

        # --- Recommended Formation (3-Renta Axis 1 -> 6 Opponents) ---
        # Cond: Prob >= 0.20, Odds >= 3.0 (EV >= 0.6 check implied but explicit EV > 1.2 in leaderboard)
        # Note: Leaderboard says Prob>=0.2, Odds>=3.0, EV>=1.2.
        # We need prob/odds from the top horse.
        top_horse = top5.iloc[0]
        prob = top_horse['prob']
        odds = top_horse['odds'] if not pd.isna(top_horse['odds']) else 0
        ev = prob * odds
        
        # Check conditions
        if prob >= 0.20 and odds >= 3.0 and ev >= 1.2:
            # Need top 7 horses for opponents (Axis + 6 Opponents)
            # Re-fetch top 7
            top7 = group.sort_values('score', ascending=False).head(7)
            if len(top7) >= 7:
                axis = int(top_horse['horse_number'])
                opps = top7.iloc[1:7]['horse_number'].astype(int).tolist() # 2nd to 7th
                
                # 3-Renta Formation: Axis -> Opps -> Opps
                # Points: 1 * 6 * 5 = 30
                # Generate all (Axis, o1, o2) perms where o1, o2 in opps, o1 != o2
                
                f_bet_count = 0
                f_return = 0
                f_hit = 0
                
                opp_perms = list(permutations(opps, 2)) # (o1, o2)
                f_bet_count = len(opp_perms) # Should be 30
                
                for o1, o2 in opp_perms:
                    comb_str = f"{axis:02}{o1:02}{o2:02}"
                    if comb_str in payout_map[race_id]['sanrentan']:
                        f_return += payout_map[race_id]['sanrentan'][comb_str]
                        f_hit = 1
                
                if 'recommended_formation' not in stats:
                     stats['recommended_formation'] = {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
                
                stats['recommended_formation']['bet'] += f_bet_count * 100
                stats['recommended_formation']['return'] += f_return
                stats['recommended_formation']['hit'] += f_hit
                stats['recommended_formation']['races'] += 1

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

def load_race_info(years=[2024]):
    logger.info(f"レース情報(jvd_ra)をロード中... Years={years}")
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    # Fetch minimal columns: race keys and hondai (title)
    # Construct race_id from keys to match
    query = text(f"""
        SELECT 
            kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
            kyosomei_hondai AS hondai
        FROM jvd_ra 
        WHERE kaisai_nen IN ({years_str})
    """)
    try:
        df = pd.read_sql(query, engine)
        df['race_id'] = (
            df['kaisai_nen'].astype(str) +
            df['keibajo_code'].astype(str) +
            df['kaisai_kai'].astype(str) +
            df['kaisai_nichime'].astype(str) +
            df['race_bango'].astype(str)
        )
        return df[['race_id', 'hondai']]
    except Exception as e:
        logger.error(f"レース情報ロードエラー: {e}")
        return pd.DataFrame()

def calculate_probability_metrics(df):
    """
    [v7] 確率予測の精度評価指標を計算する
    """
    logger.info("--- Probability Evaluation ---")
    
    # Target: rank==1 (binary)
    if 'rank' not in df.columns or 'prob' not in df.columns:
        logger.warning("rank or prob column missing. Skipping probability metrics.")
        return {}
        
    y_true = (df['rank'] == 1).astype(int)
    y_prob = df['prob']
    
    # Check if prob is valid
    if y_prob.isnull().any():
        logger.warning("Probability contains NaNs. Filling with 0.")
        y_prob = y_prob.fillna(0)

    # Metrics
    try:
        bs = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        
        logger.info(f"  Brier Score Loss: {bs:.5f}")
        logger.info(f"  Log Loss: {ll:.5f}")
        
        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        logger.info(f"  Calibration Curve (Bins=10):")
        for true, pred in zip(prob_true, prob_pred):
            logger.info(f"    Pred: {pred:.4f} -> Actual: {true:.4f}")
            
        return {
            'brier_score': bs, 
            'log_loss': ll,
            'calibration_curve': {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error calculating probability metrics: {e}")
        return {}

if __name__ == "__main__":
    main()
