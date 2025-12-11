import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from scipy.special import softmax
from sqlalchemy import create_engine, text
from datetime import datetime

from src.pipeline.config import ExperimentConfig
from src.model.ensemble import EnsembleModel
from src.model.lgbm import KeibaLGBM
from src.model.catboost_model import KeibaCatBoost
from src.model.tabnet_model import KeibaTabNet
from src.model.roi_model import ROIModel

logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def load_payout_data(years):
    logger.info(f"Loading payout data for years: {years}")
    engine = get_db_engine()
    years_str = ",".join([f"'{y}'" for y in years])
    query = text(f"SELECT * FROM jvd_hr WHERE kaisai_nen IN ({years_str})")
    try:
        df = pd.read_sql(query, engine)
        df['race_id'] = (
            df['kaisai_nen'].astype(str) +
            df['keibajo_code'].astype(str) +
            df['kaisai_kai'].astype(str) +
            df['kaisai_nichime'].astype(str) +
            df['race_bango'].astype(str)
        )
        return df
    except Exception as e:
        logger.error(f"Failed to load payout data: {e}")
        return pd.DataFrame()

def load_models(config: ExperimentConfig, run_dir: str):
    model_type = config.model.type
    models_dir = os.path.join(run_dir, "models")
    loaded_models = {}

    try:
        if model_type == 'lgbm':
            model = KeibaLGBM()
            model.load_model(os.path.join(models_dir, 'lgbm.pkl'))
            loaded_models['lgbm'] = model
            
        elif model_type == 'catboost':
            model = KeibaCatBoost()
            model.load_model(os.path.join(models_dir, 'catboost.pkl'))
            loaded_models['catboost'] = model

        elif model_type == 'tabnet':
            model = KeibaTabNet()
            # Ensure CPU usage for inference to avoid docker issues
            model.load_model(os.path.join(models_dir, 'tabnet.zip').replace('.zip', '.pkl'), device_name='cpu')
            loaded_models['tabnet'] = model
            
        elif model_type == 'ensemble' or model_type == 'ensemble_only':
            model = EnsembleModel()
            # EnsembleModel loads sub-models internally. 
            # We need to ensure it uses CPU for TabNet if present.
            # Modified EnsembleModel.load_model takes device_name.
            model.load_model(os.path.join(models_dir, 'ensemble.pkl'), device_name='cpu')
            loaded_models['ensemble'] = model
        
        elif model_type == 'roi':
            # ROIモデルのロード
            model_path = os.path.join(models_dir, 'roi_model_best.pt')
            if os.path.exists(model_path):
                model = ROIModel()
                model.load(model_path)
                loaded_models['roi'] = model
            else:
                logger.warning(f"ROI model not found at {model_path}")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e
        
    return loaded_models

def evaluate_model(config: ExperimentConfig, run_dir: str):
    logger.info("Starting Evaluation...")
    
    # 1. Load Data
    # We need raw data (with odds, rank) for evaluation, not just features.
    # Try multiple sources: 1. preprocessed parquet, 2. lgbm_datasets.pkl
    
    data_dir = os.path.join(run_dir, "data")
    df = None
    
    # Option 1: preprocessed parquet files
    if os.path.exists(data_dir):
        parquet_files = [f for f in os.listdir(data_dir) if f.startswith("preprocessed_data") and f.endswith(".parquet")]
        if parquet_files:
            data_path = os.path.join(data_dir, parquet_files[0])
            logger.info(f"Loading test data from {data_path}")
            df = pd.read_parquet(data_path)
    
    # Option 2: lgbm_datasets.pkl (valid set contains evaluation data)
    if df is None:
        pkl_path = os.path.join(data_dir, "lgbm_datasets.pkl")
        if os.path.exists(pkl_path):
            logger.info(f"Loading evaluation data from {pkl_path}")
            with open(pkl_path, 'rb') as f:
                datasets = pickle.load(f)
            # Use valid set for evaluation
            if 'valid' in datasets and 'X' in datasets['valid']:
                X_valid = datasets['valid']['X']
                y_valid = datasets['valid']['y']
                # Reconstruct basic info needed for evaluation
                df = X_valid.copy()
                # y is target (0-3), we need rank. Convert back: target 3->rank 1, 2->2, 1->3, 0->4+
                df['target'] = y_valid
                # We need odds and rank from DB or cache
                logger.info(f"Loaded {len(df)} rows from valid set. Note: odds/rank may need DB lookup.")
            else:
                logger.warning("No valid set found in lgbm_datasets.pkl")
    
    if df is None or df.empty:
        raise FileNotFoundError(f"No preprocessed data found in {data_dir}")
    
    # Filter for validation/test year
    # Config has valid_year. We treat valid_year as the evaluation target.
    target_year = config.data.valid_year
    test_df = df[df['year'] == target_year].copy() if 'year' in df.columns else df.copy()
    
    if test_df.empty:
        logger.warning(f"No data found for year {target_year}. Using all available data.")
        test_df = df.copy()

    # JRAフィルター（評価時にNARを除外）
    if config.evaluation.jra_only and 'venue' in test_df.columns:
        # JRA venue codes: 01-10
        jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        test_df['venue_code'] = test_df['venue'].astype(str).str[:2]
        before_count = len(test_df)
        test_df = test_df[test_df['venue_code'].isin(jra_codes)].copy()
        logger.info(f"🏇 JRA Only Filter: {before_count} -> {len(test_df)} rows (NAR removed: {before_count - len(test_df)})")

    if test_df.empty:
         raise ValueError(f"No data available for evaluation (Year: {target_year})")

    # 2. Load Models
    models = load_models(config, run_dir)
    
    # 評価対象モデルリストの拡張
    # アンサンブルの場合は、その構成モデルも個別に評価する
    eval_targets = {}
    if 'ensemble' in models:
        ens = models['ensemble']
        eval_targets['Ensemble'] = ens
        if ens.has_lgbm:
            eval_targets['LightGBM (Single)'] = ens.lgbm
        if ens.has_catboost:
            eval_targets['CatBoost (Single)'] = ens.catboost
        if ens.has_tabnet:
            eval_targets['TabNet (Single)'] = ens.tabnet
    elif 'roi' in models:
        # ROIモデルの場合は特別な評価が必要
        eval_targets['ROI'] = models['roi']
    else:
        # 単体モデルの場合
        for name, model in models.items():
            eval_targets[name] = model
    
    # モデルが見つからない場合のチェック
    if not eval_targets:
        logger.warning("No models found for evaluation. Skipping evaluation.")
        return

    simulation_results = {
        'model_type': config.model.type,
        'target_year': target_year,
        'metrics': {}, # モデルごとの指標
        'strategies': {} # 主モデル(Ensemble等)の戦略結果
    }

    # 主モデル（推論結果保存用）
    main_model_name = list(eval_targets.keys())[0]
    main_model = eval_targets[main_model_name]

    # 特徴量カラムの特定 (Main Modelから)
    feature_cols = None
    if hasattr(main_model, 'feature_names') and main_model.feature_names:
         feature_cols = main_model.feature_names
    elif hasattr(main_model, 'model') and hasattr(main_model.model, 'feature_name'): # LGBM
         feature_cols = main_model.model.feature_name()
    elif hasattr(main_model, 'model') and hasattr(main_model.model, 'feature_names_'): # CatBoost
         feature_cols = main_model.model.feature_names_
    
    if feature_cols is None:
        ds_path = os.path.join(run_dir, "data/lgbm_datasets.pkl")
        if os.path.exists(ds_path):
            with open(ds_path, 'rb') as f:
                d = pickle.load(f)
                if hasattr(d['train']['X'], 'columns'):
                    feature_cols = d['train']['X'].columns.tolist()

    if feature_cols is None:
         raise RuntimeError("特徴量カラム名を特定できませんでした。")

    # 欠損カラムの補完
    missing = set(feature_cols) - set(test_df.columns)
    if missing:
        for c in missing:
            test_df[c] = 0
            
    X_test = test_df[feature_cols]

    # 各モデルの評価実行
    logger.info(f"評価対象モデル: {list(eval_targets.keys())}")
    
    for name, model in eval_targets.items():
        logger.info(f"Evaluating {name}...")
        try:

            if hasattr(model, 'model_type') and model.model_type in ['simple', 'attention']:
                # ROIModel用の予測ロジック (DataFrame -> 3D Tensor)
                # 数値カラムのみ抽出
                numeric_df = X_test.select_dtypes(include=[np.number])
                roi_features = numeric_df.columns.tolist()
                
                # input_dimチェック
                if hasattr(model, 'input_dim') and model.input_dim != len(roi_features):
                    # 足りない分をパディング（簡易対応）
                    diff = model.input_dim - len(roi_features)
                    if diff > 0:
                        logger.warning(f"ROI feature mismatch: model {model.input_dim} vs data {len(roi_features)}. Padding with 0.")
                        for i in range(diff):
                            numeric_df[f'_pad_{i}'] = 0
                        roi_features = numeric_df.columns.tolist()
                    elif diff < 0:
                         # 多い分は切り捨て
                         roi_features = roi_features[:model.input_dim]

                # レースIDに基づいて3D変換して予測
                # Evaluate用のdfはtest_df（X_testの元）にあるはず
                # X_testはfeature_colsのみなのでrace_idがない可能性がある
                # test_dfを使う
                
                # インデックスを保存して整列に使用
                temp_scores = []
                
                # test_dfにrace_idとhorse_numberがあることを前提
                # X_testはDataFrameなのでindexはtest_dfと同じはず
                # numeric_dfにindexを付与
                numeric_df['__idx__'] = numeric_df.index
                
                # race_idが必要なのでtest_dfから結合
                if 'race_id' not in numeric_df.columns:
                    numeric_df['race_id'] = test_df['race_id']
                if 'horse_number' not in numeric_df.columns:
                    numeric_df['horse_number'] = test_df['horse_number']
                
                for race_id, grp in numeric_df.groupby('race_id'):
                    grp = grp.sort_values('horse_number')
                    indices = grp['__idx__'].values
                    
                    # 特徴量 (roi_featuresを使用)
                    X_grp = grp[roi_features[:model.input_dim]].values.astype(np.float32)
                    X_grp = np.nan_to_num(X_grp, nan=0.0)
                    
                    n_horses = len(grp)
                    max_horses = 18 # モデルの想定最大数
                    
                    # パディング
                    X_padded = np.zeros((1, max_horses, model.input_dim), dtype=np.float32)
                    mask = np.zeros((1, max_horses), dtype=np.float32)
                    
                    n = min(n_horses, max_horses)
                    X_padded[0, :n, :] = X_grp[:n]
                    mask[0, :n] = 1.0
                    
                    # 予測
                    p = model.predict(X_padded, mask)
                    
                    # 結果取得
                    batch_scores = p[0, :n]
                    
                    for i, idx in enumerate(indices[:n]):
                        temp_scores.append({'idx': idx, 'score': batch_scores[i]})
                
                # 元の順序に戻す
                score_map = {item['idx']: item['score'] for item in temp_scores}
                scores = np.array([score_map.get(i, -999.0) for i in X_test.index])
                
            else:
                # 通常のモデル (LightGBM, CatBoost, TabNet, Ensemble)
                scores = model.predict(X_test)
            
            # 一時的にスコアをDataFrameに入れる（計算用）
            temp_df = test_df.copy()
            temp_df['score'] = scores
            temp_df['prob'] = temp_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
            temp_df['expected_value'] = temp_df['prob'] * temp_df['odds'].fillna(0)
            
            # 基本指標 (Accuracy, ROI @ Max Score)
            res = simulate_single_choice(temp_df, 'score')
            simulation_results['metrics'][name] = res
            logger.info(f"[{name}] ROI: {res['roi']:.2f}%, Accuracy: {res['accuracy']:.2%}")
            
            # Main Modelの場合は結果をtest_dfに保存（後の戦略最適化用）
            if name == main_model_name:
                test_df['score'] = scores
                test_df['prob'] = temp_df['prob']
                test_df['expected_value'] = temp_df['expected_value']
                
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")

    # 4. ROI Simulation (for Main Model)
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Single Choice Strategies (Main Model)
    logger.info(f"Simulating Single Choice Strategies for {main_model_name}...")
    for strategy in ['max_score', 'max_ev']:
        target = 'score' if strategy == 'max_score' else 'expected_value'
        res = simulate_single_choice(test_df, target)
        simulation_results['strategies'][strategy] = res
        
    # Save Metrics
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(simulation_results, f, indent=4, cls=NpEncoder)
        
    # Save Predictions (Main Model)
    pred_path = os.path.join(reports_dir, "predictions.parquet")
    test_df.to_parquet(pred_path, index=False)
    
    logger.info(f"Evaluation completed. Reports saved to {reports_dir}")

def simulate_single_choice(df, target_col):
    results = []
    for race_id, group in df.groupby('race_id'):
        if group[target_col].isnull().all(): continue
        
        best = group.loc[group[target_col].idxmax()]
        bet = 100
        # rankカラムチェック
        if 'rank' not in best:
            # targetがあれば復元を試みる (target: 3->1, 2->2, 1->3, 0->other)
            if 'target' in best:
                t = best['target']
                if t == 3: best['rank'] = 1
                elif t == 2: best['rank'] = 2
                elif t == 1: best['rank'] = 3
                else: best['rank'] = 99
            else:
                # ランク情報なし
                continue
                
        ret = best['odds'] * 100 if best['rank'] == 1 else 0
        hit = 1 if best['rank'] == 1 else 0
        place_hit = 1 if best['rank'] <= 3 else 0
        results.append({'bet': bet, 'return': ret, 'hit': hit, 'place_hit': place_hit})
        
    df_res = pd.DataFrame(results)
    if df_res.empty: return {'roi': 0, 'accuracy': 0, 'place_rate': 0, 'bets': 0}
    
    total_bet = df_res['bet'].sum()
    total_ret = df_res['return'].sum()
    roi = (total_ret / total_bet) * 100 if total_bet > 0 else 0
    acc = df_res['hit'].mean()
    place_rate = df_res['place_hit'].mean()
    
    return {'roi': roi, 'accuracy': acc, 'place_rate': place_rate, 'bets': len(df_res), 'total_return': total_ret}
