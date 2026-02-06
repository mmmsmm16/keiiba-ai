import argparse
import yaml
import pandas as pd
import numpy as np
# lightgbm moved down
import os
import sys
import logging
import pickle
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score, average_precision_score, brier_score_loss
from sklearn.model_selection import KFold
import mlflow

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.utils.leak_detector import check_data_leakage
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.dataset import DatasetSplitter
from src.config.validator import ConfigValidator
import lightgbm as lgb

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイル(YAML)を読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_experiment(config_path: str, strict: bool = False):
    """実験を実行するメイン関数"""
    # 1. 設定の読み込み
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {config_path}. エラー: {e}")
        return

    # [Enhanced] Config Guardrail Check
    # Check if strict mode is enabled in config or args
    config_strict = config.get('strict', False)
    is_strict = strict or config_strict
    
    try:
        ConfigValidator.validate(config, config_path=config_path, strict=is_strict)
    except ValueError as e:
        logger.error(f"⛔ Config Validation Failed: {e}")
        # Stop experiment immediately
        return

    exp_name = config.get('experiment_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    feature_blocks = config.get('features', [])
    model_params = config.get('model_params', {})
    dataset_cfg = config.get('dataset', {})
    calibration_cfg = config.get('calibration', {})
    
    # 成果物ディレクトリの作成
    artifact_dir = f"models/experiments/{exp_name}"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 設定のコピーを保存
    with open(os.path.join(artifact_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # [NEW] Git Metadata retrieval
    git_hash = "unknown"
    is_dirty = False
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        is_dirty = bool(status)
    except Exception:
        pass
        
    # [NEW] Save Metadata for Leaderboard/Reproducibility
    metadata = {
        'experiment_name': exp_name,
        'model_type': model_params.get('model_type', 'lightgbm'),
        'objective': model_params.get('objective', ''),
        'target_col': dataset_cfg.get('target_col', ''),
        'binary_target': dataset_cfg.get('binary_target', ''),
        'metrics': model_params.get('metric', []),
        'time_decay_enabled': config.get('sample_weight', {}).get('enabled', False),
        'time_decay_strategy': config.get('sample_weight', {}).get('strategy', 'none'),
        'valid_year': dataset_cfg.get('valid_year', 2024),
        'train_end_date': dataset_cfg.get('train_end_date', ''),
        'feature_count': len(feature_blocks),
        'calibration_enabled': calibration_cfg.get('enabled', False),
        'timestamp': datetime.now().isoformat(),
        'git': {
            'commit_hash': git_hash,
            'is_dirty': is_dirty
        },
        'strict_mode': is_strict
    }
    with open(os.path.join(artifact_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"🚀 実験を開始します: {exp_name}")
    logger.info(f"使用特徴量ブロック: {feature_blocks}")

    # [Optim] Data Loading OUTSIDE MLFlow context to prevent deadlock
    loader = JraVanDataLoader()
    start_date = dataset_cfg.get('train_start_date', '2015-01-01')
    end_date = dataset_cfg.get('test_end_date', '2025-12-31')
    jra_only = dataset_cfg.get('jra_only', False)
    skip_odds = dataset_cfg.get('drop_market_data', False)
    
    logger.info(f"データをロード中 ({start_date} ~ {end_date})...")
    # [Optim] skip_training=True because no features use jvd_hc (prevents OOM)
    raw_df = loader.load(history_start_date=start_date, end_date=end_date, jra_only=jra_only, skip_odds=skip_odds, skip_training=True)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)

    with mlflow.start_run(run_name=exp_name):
        logger.info(f"✨ MLFlow Run Started: {exp_name}")
        
        # 3. 特徴量パイプライン
        logger.info("FeaturePipelineを初期化中...")
        try:
            pipeline = FeaturePipeline(cache_dir="data/features")
            logger.info(f"FeaturePipeline初期化完了. ブロック数: {len(feature_blocks)}")
            sys.stdout.flush()
            
            logger.info("FeaturePipelineを使用してデータセットを構築中 (load_features)...")
            df = pipeline.load_features(clean_df, feature_blocks)
            logger.info(f"特徴量作成完了: Shape={df.shape}")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"❌ FeaturePipelineでエラーが発生しました: {e}", exc_info=True)
            return
        
        # 4. リーク検知 & ターゲット作成
        logger.info("ターゲット作成とリーク検知を実行中...")
        if 'rank' in clean_df.columns and 'target' not in df.columns:
            if 'rank' not in df.columns:
                target_source = clean_df[['race_id', 'horse_number', 'rank', 'odds']]
                df = pd.merge(df, target_source, on=['race_id', 'horse_number'], how='left')

            def create_ranking_target(rank):
                if pd.isna(rank): return 0
                if rank == 1: return 3
                elif rank == 2: return 2
                elif rank == 3: return 1
                else: return 0
            df['target'] = df['rank'].apply(create_ranking_target)
            
            if model_params.get('objective') == 'regression':
                logger.info("🎯 回帰モード: Target = FinalOdds * (Rank==1) を作成します。")
                df['target'] = df.apply(lambda row: row['odds'] if row['rank'] == 1 else 0.0, axis=1).fillna(0.0)
                
        try:
            check_data_leakage(df, target_col='target')
        except ValueError as e:
            logger.error(f"⛔ リークが検出されました: {e}")
            return

        # 5. データ分割
        splitter = DatasetSplitter()
        valid_year = dataset_cfg.get('valid_year', 2024)
        train_end_str = dataset_cfg.get('train_end_date', '2023-12-31')
        train_end_dt = pd.to_datetime(train_end_str)
        train_end_year = train_end_dt.year
        logger.info(f"データセット分割を実行中 (検証年: {valid_year}, Train End: {train_end_year})...")
        
        key_cols = ['race_id', 'date', 'horse_id'] 
        for k in key_cols:
            if k not in df.columns and k in clean_df.columns:
                df[k] = clean_df[k]
                
        if 'year' not in df.columns and 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        elif 'year' not in df.columns and 'year' in clean_df.columns:
            df['year'] = clean_df['year']

        datasets = splitter.split_and_create_dataset(df, valid_year=valid_year)
    train_set, valid_set = datasets['train'], datasets['valid']
    
    # [NEW] Time-Decay Weighting
    sample_weight_cfg = config.get('sample_weight', {})
    if sample_weight_cfg.get('enabled', False) and 'train' in datasets:
        strategy = sample_weight_cfg.get('strategy', 'time_decay')
        normalize = sample_weight_cfg.get('normalize', True)
        logger.info(f"⚖️ Sample Weighting enabled. Strategy: {strategy}, Normalize: {normalize}")
        
        X_train = train_set['X']
        if 'date' in X_train.columns:
            dates = pd.to_datetime(X_train['date'])
        elif 'date' in df.columns:
            dates = pd.to_datetime(df.loc[X_train.index, 'date'])
        else:
            dates = None

        if dates is not None:
            years = dates.dt.year
            weights = np.ones(len(dates))
            
            if strategy == 'exponential':
                decay_rate = sample_weight_cfg.get('decay_rate', 0.001)
                # days_old = (train_end - race_date).days
                # 未来の日付が含まれる場合 (設定ミス等) は0 (重み1.0) にする
                days_old = (train_end_dt - dates).dt.days
                days_old = np.maximum(days_old, 0) 
                weights = np.exp(-decay_rate * days_old)
                
            elif strategy == 'piecewise':
                # delta = train_end_year - race_year
                # delta=0 (Last Year) -> 1.0
                # delta=1 (Prev Year) -> 0.7
                # delta=2 -> 0.5
                # else -> 0.3
                
                # Configから相対マップを取得 (なければデフォルト)
                # config format expectation:
                # year_weights: {0: 1.0, 1: 0.7, 2: 0.5, "default": 0.3}
                # 互換性のため、既存の年指定(2024: 1.0)がある場合の変換ロジックも入れる？
                # いや、M4からは新しい相対ロジックで行く。Config側も合わせる必要があるが、
                # M3のConfigは年指定だった。
                # ここでは「キーが2000以上の場合は絶対年、それ以外は相対年」と判定して互換性を維持する。
                
                yw_cfg = sample_weight_cfg.get('year_weights', {})
                default_w = yw_cfg.get('default', 0.3)
                
                delta_years = train_end_year - years
                
                # ベクトル化適用
                # まずデフォルト値で初期化
                weights = np.full(len(X_train), default_w)
                
                # キーごとに適用
                for k, w in yw_cfg.items():
                    if k == 'default': continue
                    try:
                        k_int = int(k)
                        # 2000以上なら絶対年、それ以外なら相対年
                        if k_int > 1900:
                            # Absolute Year Mode (Legacy Support)
                            mask = (years == k_int)
                        else:
                            # Relative Year Mode (Delta)
                            mask = (delta_years == k_int)
                            
                        weights[mask] = w
                    except:
                        continue
                        
            if normalize:
                weights = weights / weights.mean()
                
            # Log stats
            w_series = pd.Series(weights)
            logger.info(f"  Weights stats: min={w_series.min():.4f}, max={w_series.max():.4f}, mean={w_series.mean():.4f}")
            
            train_set['weight'] = weights
            logger.info(f"  Weights: Min={weights.min():.4f}, Max={weights.max():.4f}, Mean={weights.mean():.4f}")

    # 特徴量フィルタリング
    if skip_odds:
        logger.info("🚫 市場データ (odds, popularity) を特徴量から除外します。")
        market_cols = [c for c in train_set['X'].columns if any(m in c for m in ['odds', 'popularity'])]
        if market_cols:
            train_set['X'] = train_set['X'].drop(columns=market_cols)
            valid_set['X'] = valid_set['X'].drop(columns=market_cols)

    # [FIX] Exclude Features defined in config
    exclude_cols = dataset_cfg.get('exclude_features', [])
    if exclude_cols:
        logger.info(f"🚫 指定された特徴量を除外します: {len(exclude_cols)} items")
        # 実際に存在するカラムのみ
        drop_target = [c for c in exclude_cols if c in train_set['X'].columns]
        if drop_target:
            train_set['X'] = train_set['X'].drop(columns=drop_target)
            valid_set['X'] = valid_set['X'].drop(columns=drop_target)

    # 6. モデル学習
    model_type = model_params.get('model_type', 'lightgbm')
    objective = model_params.get('objective', 'lambdarank')
    do_calibration = calibration_cfg.get('enabled', False) and objective == 'binary'
    
    cat_features = dataset_cfg.get('categorical_features', [])
    auto_cat = [c for c in train_set['X'].columns if train_set['X'][c].dtype == 'object']
    cat_features = list(set(cat_features + auto_cat))
    cat_features = [c for c in cat_features if c not in ['race_id', 'date', 'horse_id', 'target', 'year', 'y', 'rank', 'odds', 'target_win', 'target_top3']]
    if cat_features:
        logger.info(f"Categorical features detected: {cat_features}")

    train_y, valid_y = train_set['y'], valid_set['y']
    valid_y_relevance = valid_y.copy()
    
    if objective == 'binary':
        binary_target = dataset_cfg.get('binary_target', 'top3')
        if binary_target == 'win':
            train_y, valid_y = (train_y == 3).astype(int), (valid_y == 3).astype(int)
        elif binary_target == 'top2':
            train_y, valid_y = (train_y >= 2).astype(int), (valid_y >= 2).astype(int)
        else:
            train_y, valid_y = (train_y > 0).astype(int), (valid_y > 0).astype(int)

    # 特徴量セットの確定
    drop_cols = ['race_id', 'horse_id', 'date', 'target', 'year', 'y', 'rank', 'odds', 'target_win', 'target_top3', 'is_win', 'is_top3']
    feature_cols = [c for c in train_set['X'].columns if c not in drop_cols]
    cat_features = [c for c in cat_features if c in feature_cols]
    
    def prepare_df(df_input):
        df_out = df_input[feature_cols].copy()
        if model_type == 'catboost':
            for col in cat_features: df_out[col] = df_out[col].fillna("missing").astype(str)
        else:
            for col in cat_features: df_out[col] = df_out[col].astype('category')
        return df_out

    # 全体データの事前加工 (高速化のため)
    logger.info("⏳ 特徴量の事前加工中...")
    X_train_processed = prepare_df(train_set['X'])
    X_valid_processed = prepare_df(valid_set['X'])

    def train_model(X_train_pre, t_y, t_group, valid_pts_pre, override_params=None, weight=None):
        params = override_params if override_params else model_params
        
        if model_type == 'catboost':
            import catboost as cb
            cb_params = params.copy()
            cb_params.pop('model_type', None); cb_params.pop('objective', None); cb_params.pop('early_stopping_rounds', None)
            loss_fn = 'Logloss' if objective == 'binary' else objective
            if objective == 'lambdarank': loss_fn = 'YetiRank'
            elif objective == 'regression': loss_fn = 'RMSE'
            
            t_pool = cb.Pool(data=X_train_pre, label=t_y, weight=weight, group_id=np.repeat(np.arange(len(t_group)), t_group) if objective == 'lambdarank' else None, cat_features=cat_features)
            eval_sets = []
            for vX, vy, vg in valid_pts_pre:
                v_pool = cb.Pool(data=vX, label=vy, group_id=np.repeat(np.arange(len(vg)), vg) if objective == 'lambdarank' else None, cat_features=cat_features)
                eval_sets.append(v_pool)
            
            fit_m = cb.CatBoostClassifier(loss_function=loss_fn, **cb_params) if objective == 'binary' else \
                    (cb.CatBoostRanker(loss_function=loss_fn, **cb_params) if objective == 'lambdarank' else \
                     cb.CatBoostRegressor(loss_function=loss_fn, **cb_params))
            fit_m.fit(t_pool, eval_set=eval_sets if eval_sets else None, early_stopping_rounds=params.get('early_stopping_rounds', 50), verbose=False)
            return fit_m
        else:
            lgb_train = lgb.Dataset(X_train_pre, label=t_y, categorical_feature=cat_features, free_raw_data=False, weight=weight)
            if objective == 'lambdarank': lgb_train.set_group(t_group)
            
            v_sets, v_names = [lgb_train], ['train']
            for i, (vX, vy, vg) in enumerate(valid_pts_pre):
                v_ds = lgb.Dataset(vX, label=vy, reference=lgb_train, categorical_feature=cat_features, free_raw_data=False)
                if objective == 'lambdarank': v_ds.set_group(vg)
                v_sets.append(v_ds)
                v_names.append(f'valid_{i}')
            
            l_params = params.copy(); l_params.pop('model_type', None)
            n_rounds = params.get('num_boost_round', params.get('n_estimators', 100))
            callbacks = [lgb.log_evaluation(period=0)]
            if valid_pts_pre: callbacks.append(lgb.early_stopping(stopping_rounds=50))
            return lgb.train(l_params, lgb_train, num_boost_round=n_rounds, valid_sets=v_sets, valid_names=v_names, callbacks=callbacks)

    # 5.5 Dry Run / Sanity Check
    logger.info("🛠️ 学習前のデータ整合性チェック (Dry Run) を実行中...")
    try:
        _ = train_model(X_train_processed.head(100), train_y[:100], [100], [], override_params={'n_estimators': 1})
        logger.info("✅ データ整合性チェック合格。")
    except Exception as e:
        logger.error(f"❌ データ整合性チェックでエラーが発生しました: {e}")
        return

    # メインモデル学習
    model_save_path_pkl = os.path.join(artifact_dir, 'model.pkl')
    model_save_path_cbm = os.path.join(artifact_dir, 'model.cbm')
    model = None

    if model_type == 'catboost' and os.path.exists(model_save_path_cbm):
        import catboost as cb
        logger.info(f"📦 既存のモデルをロードしています: {model_save_path_cbm}")
        model = cb.CatBoostClassifier() if objective == 'binary' else \
                (cb.CatBoostRanker() if objective == 'lambdarank' else cb.CatBoostRegressor())
        model.load_model(model_save_path_cbm)
    elif model_type != 'catboost' and os.path.exists(model_save_path_pkl):
        logger.info(f"📦 既存のモデルをロードしています: {model_save_path_pkl}")
        with open(model_save_path_pkl, 'rb') as f: model = pickle.load(f)

    if model is None:
        logger.info(f"{model_type.upper()}の学習を開始 (目的関数: {objective})...")
        model = train_model(X_train_processed, train_y, train_set['group'], [(X_valid_processed, valid_y, valid_set['group'])], weight=train_set.get('weight'))
        # 中間保存
        if model_type == 'catboost': model.save_model(model_save_path_cbm)
        else:
            with open(model_save_path_pkl, 'wb') as f: pickle.dump(model, f)
        logger.info(f"✅ メインモデルを保存しました。")
    
    # 確率校正器の学習 (OOF)
    calibrator = None
    calibrator_path = os.path.join(artifact_dir, 'calibrator.pkl')
    oof_path = os.path.join(artifact_dir, 'oof_probs.npy')
    
    if do_calibration:
        from src.models.calibration import ProbabilityCalibrator
        if os.path.exists(calibrator_path):
            logger.info(f"📦 既存の確率校正器をロードしています: {calibrator_path}")
            with open(calibrator_path, 'rb') as f: calibrator = pickle.load(f)
        else:
            n_folds, method = calibration_cfg.get('n_folds', 5), calibration_cfg.get('method', 'platt')
            
            if os.path.exists(oof_path):
                logger.info(f"💾 既存のOOF予測をロードしています: {oof_path}")
                oof_probs = np.load(oof_path)
            else:
                logger.info(f"🔮 確率校正器の学習を開始 (Method: {method}, OOF Folds: {n_folds})...")
                
                # OOF分割用にrace_idを確保
                if 'race_id' not in train_set['X'].columns:
                    train_set['X']['race_id'] = df.loc[train_set['X'].index, 'race_id'].values
                
                unique_races = train_set['X']['race_id'].unique()
                kf, oof_probs = KFold(n_splits=n_folds, shuffle=True, random_state=42), np.zeros(len(train_y))
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(unique_races)):
                    logger.info(f"  Calibration OOF Fold {fold+1}/{n_folds} training...")
                    mask_t = train_set['X']['race_id'].isin(unique_races[train_idx])
                    mask_v = train_set['X']['race_id'].isin(unique_races[val_idx])
                    
                    X_t_pre, y_t = X_train_processed[mask_t], train_y[mask_t]
                    X_v_pre, y_v = X_train_processed[mask_v], train_y[mask_v]
                    
                    def get_groups(mask): return train_set['X'][mask].groupby('race_id', sort=False).size().values
                    
                    oof_params = model_params.copy()
                    if oof_params.get('n_estimators', 0) > 500: oof_params['n_estimators'] = 500
                    # OOF学習では early_stopping を無効化（高速化のため）
                    oof_params.pop('early_stopping_rounds', None)
                    
                    try:
                        m_oof = train_model(X_t_pre, y_t, get_groups(mask_t), [], override_params=oof_params)
                        logger.info(f"  Calibration OOF Fold {fold+1}/{n_folds} predicting...")
                        probs = m_oof.predict(X_v_pre)
                        # Nan Clean
                        probs = np.nan_to_num(probs, nan=0.0)
                        oof_probs[mask_v] = probs
                    except Exception as e:
                        logger.error(f"  Fold {fold+1} failed: {e}")
                        # Fallback: fill with mean
                        oof_probs[mask_v] = y_t.mean()
                
                np.save(oof_path, oof_probs)
                
            calibrator = ProbabilityCalibrator(method=method)
            try:
                # Pre-clean OOF
                oof_clean = np.nan_to_num(oof_probs, nan=0.0)
                oof_clean = np.clip(oof_clean, 0.0, 1.0)
                calibrator.fit(train_y.values, oof_clean)
                logger.info("✅ 確率校正器の学習が完了しました。")
            except Exception as e:
                logger.warning(f"⚠️ Isotonic Calibration failed: {e}. Falling back to Sigmoid (Platt).")
                try:
                    calibrator = ProbabilityCalibrator(method='sigmoid')
                    calibrator.fit(train_y.values, oof_clean)
                    logger.info("✅ Sigmoid校正器の学習が完了しました (Fallback)。")
                except Exception as e2:
                     logger.error(f"❌ Calibration 致命的エラー: {e2}. 校正をスキップします。")
                     calibrator = None

            if calibrator:
                with open(calibrator_path, 'wb') as f: pickle.dump(calibrator, f)
    
    # 7. 評価
    logger.info("モデル評価中...")
    preds = model.predict(prepare_df(valid_set['X']))
    if calibrator:
        logger.info("  Applying probability calibration to predictions...")
        preds = calibrator.predict(preds)
        
    binary_y = (valid_y > 0).astype(int) 
    if objective == 'regression':
        from sklearn.metrics import mean_squared_error
        auc_score, ll_score, bs_score, ap_score = 0.0, np.sqrt(mean_squared_error(valid_y, preds)), 0.0, 0.0
        logger.info(f"Calculated RMSE: {ll_score:.4f}")
    else:
        auc_score = roc_auc_score(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
        ll_score = log_loss(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
        ap_score = average_precision_score(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
        bs_score = brier_score_loss(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
    
    # 7. 評価
    logger.info("モデル評価中...")
    X_valid = valid_set['X']
    
    # 評価用メタデータを取得 (Distance, N_Horses)
    # df or clean_df から取得。Indexは一致している前提
    # X_validのindexを使ってclean_dfから取得する
    meta_df = clean_df.loc[X_valid.index, ['race_id', 'distance']].copy()
    
    # 頭数(n_horses)を計算 (race_idごとのレコード数)
    # ※ clean_dfには全頭いるはず
    race_counts = meta_df.groupby('race_id')['race_id'].transform('count')
    meta_df['n_horses'] = race_counts

    preds = model.predict(prepare_df(X_valid))
    if calibrator:
        logger.info("  Applying probability calibration to predictions...")
        preds = calibrator.predict(preds)
        
    binary_y = (valid_y > 0).astype(int) 
    
    # Metrics Calculation Helper
    def calc_ranking_metrics(y_true_bin, y_score, groups):
        ndcg_list, recall_list = [], []
        curr = 0
        for size in groups:
            # sizeが一致しない場合はスキップ (念のため)
            if curr + size > len(y_true_bin): break
            
            y_t_bin = y_true_bin[curr : curr + size]
            y_s = y_score[curr : curr + size]
            
            if size > 1 and np.sum(y_t_bin) > 0:
                # NDCG (ranking quality)
                # binary target for NDCG implies relevance 1 or 0
                ndcg_list.append(ndcg_score([y_t_bin], [y_s], k=5))
                
                # Recall@5 (Race-Hit@5: Top3馬が予測Top5に1頭でも入れば1)
                top_k_idx = np.argsort(y_s)[::-1][:5]
                # Hit count in Top 5
                hits = np.sum(y_t_bin[top_k_idx])
                # Recall per race definition: User definition seems to be "Any Hit" (based on previous chat) 
                # OR standard Recall (Hits / Total Positives).
                # User prompted: "Race-Hit@5 (Top5にTop3馬が1頭でも含まれるか)" -> This is Hit Rate @ 5.
                # However, M4-A Audit used standard Recall formula? 
                # Wait, User said: "Verify Recall@5 ... (0.9635)". 0.96 is extremely high for "All Top3 found in Top5".
                # It MUST be "Any Hit".
                # Let's calculate both: "Race-Hit" (Any) and "Recall" (Coverage).
                # But for the report, User specifically asked for "Race-Hit@5" AND "NDCG".
                # I will calculate "RaceHit@5" (Variable name recall_at_5 in legacy code likely meant this).
                
                is_hit = 1.0 if hits > 0 else 0.0
                recall_list.append(is_hit)
                
            curr += size
        
        return np.mean(ndcg_list) if ndcg_list else 0.0, np.mean(recall_list) if recall_list else 0.0

    # Overall Metrics
    groups_valid = valid_set['group']
    ndcg_all, hit_all = calc_ranking_metrics(binary_y.values, preds, groups_valid)
    
    if objective == 'regression':
         from sklearn.metrics import mean_squared_error
         auc_score, ll_score, bs_score, ap_score = 0.0, np.sqrt(mean_squared_error(valid_y, preds)), 0.0, 0.0
         logger.info(f"Calculated RMSE: {ll_score:.4f}")
    else:
         auc_score = roc_auc_score(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
         ll_score = log_loss(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
         ap_score = average_precision_score(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0
         bs_score = brier_score_loss(binary_y, preds) if len(np.unique(binary_y)) > 1 else 0.0

    logger.info(f"Overall - NDCG@5: {ndcg_all:.4f}, Race-Hit@5: {hit_all:.4f}")

    # Segment Evaluation
    # Re-construct dataframe for segmentation
    eval_df = pd.DataFrame({
        'race_id': meta_df['race_id'].values,
        'distance': meta_df['distance'].values,
        'n_horses': meta_df['n_horses'].values,
        'y_bin': binary_y.values,
        'y_score': preds
    })
    
    # Segment: Small Field (<= 10)
    # race_id単位でgroupして判定する必要があるが、eval_dfは既に展開されている。
    # race_idごとに集約してmetrics計算するのは重いので、groupsとmaskを使ってフィルタリングする
    
    # 1. Race Level Attributes Map
    race_attrs = eval_df.groupby('race_id')[['distance', 'n_horses']].first()
    
    # 2. Filter Race IDs
    small_races = race_attrs[race_attrs['n_horses'] <= 10].index
    mile_races = race_attrs[(race_attrs['distance'] >= 1400) & (race_attrs['distance'] <= 1800)].index
    
    def eval_subset(subset_races, label):
        # subsetに含まれる行のみ抽出
        # Note: group構造を維持する必要がある。
        # 単純にmaskするとgroupが壊れる。
        # Race単位でループして、そのRaceがsubsetに含まれるか判定するのが確実。
        
        ndcg_list, hit_list = [], []
        curr = 0
        for size in groups_valid:
            # 現在のチャンクのrace_idを取得 (先頭1行で十分)
            if curr >= len(eval_df): break
            rid = eval_df.iloc[curr]['race_id']
            
            if rid in subset_races:
                chunk = eval_df.iloc[curr : curr + size]
                y_t_bin = chunk['y_bin'].values
                y_s = chunk['y_score'].values
                
                if size > 1 and np.sum(y_t_bin) > 0:
                    ndcg_list.append(ndcg_score([y_t_bin], [y_s], k=5))
                    top_k = np.argsort(y_s)[::-1][:5]
                    hits = np.sum(y_t_bin[top_k])
                    hit_list.append(1.0 if hits > 0 else 0.0)
            
            curr += size
        
        val_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
        val_hit = np.mean(hit_list) if hit_list else 0.0
        logger.info(f"Segment [{label}] - NDCG@5: {val_ndcg:.4f}, Race-Hit@5: {val_hit:.4f}")
        return val_ndcg, val_hit

    ndcg_small, hit_small = eval_subset(small_races, "SmallField<=10")
    ndcg_mile, hit_mile = eval_subset(mile_races, "Mile1400-1800")
    
    # Metrics JSON Save
    metrics_summary = {
        'overall': {
            'auc': auc_score, 'logloss': ll_score, 'ndcg_5': ndcg_all, 'race_hit_5': hit_all
        },
        'segments': {
            'small_field': {'ndcg_5': ndcg_small, 'race_hit_5': hit_small},
            'mile': {'ndcg_5': ndcg_mile, 'race_hit_5': hit_mile}
        }
    }
    with open(os.path.join(artifact_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)


    # 8. リーダーボード記録
    leaderboard_path = "reports/experiment_leaderboard.md"
    if not os.path.exists(leaderboard_path):
        with open(leaderboard_path, 'w', encoding='utf-8') as f:
            f.write("| Exp ID | Features | Model | Year | AUC | LogLoss | Brier | PR-AUC | NDCG | Recall@5 | ROI | Desc |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    row = f"| {exp_name} | {len(feature_blocks)} features | {objective} | {valid_year} | {auc_score:.4f} | {ll_score:.4f} | {bs_score:.4f} | {ap_score:.4f} | {ndcg_all:.4f} | {hit_all:.4f} | 0.0% | {dataset_cfg.get('description', '')} |\n"
    with open(leaderboard_path, 'a', encoding='utf-8') as f: f.write(row)
    logger.info("✅ 実験が正常に完了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment with Config Guardrails")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--strict', action='store_true', help='Enable strict config validation (Warnings -> Errors)')
    args = parser.parse_args()
    
    run_experiment(args.config, strict=args.strict)
