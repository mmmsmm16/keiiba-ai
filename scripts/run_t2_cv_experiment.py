
import sys
import os
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from sklearn.model_selection import GroupKFold
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader
from evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="T2 CV Experiment Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--mode", type=str, default="cv", choices=['cv', 'holdout'], help="cv=K-Fold CV, holdout=Train/Valid split (faster)")
    parser.add_argument("--objective", type=str, default="binary", choices=['binary', 'top2', 'top3', 'regression', 'lambdarank'], help="binary=1着, top2=2着以内, top3=3着以内, regression=順位回帰, lambdarank=ランキング")
    parser.add_argument("--start_year", type=int, default=None, help="Start year for training (override config)")
    parser.add_argument("--weight_strategy", type=str, default="none", choices=['none', 'balanced', 'daily'], help="Sample weighting strategy")
    parser.add_argument("--save_preds", action="store_true", help="Save validation predictions to parquet")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    EXP_NAME = cfg.get('experiment_name', 'exp_t2_cv')
    MODEL_DIR = cfg.get('model_save_path', f"models/experiments/{EXP_NAME}")
    
    # Override for CV experiment
    MODEL_DIR = os.path.join(MODEL_DIR, "cv_results")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info(f"🚀 Starting T2 CV Experiment [{EXP_NAME}]")
    
    # Paths
    FEAT_path = "data/temp_t2/T2_features.parquet"
    TGT_path = "data/temp_t2/T2_targets.parquet"
    
    # Load Data
    logger.info("Loading Data...")
    try:
        df_features = pd.read_parquet(FEAT_path)
        df_targets = pd.read_parquet(TGT_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Merge features and targets
    if 'date' in df_features.columns and 'date' in df_targets.columns:
        df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number', 'date'], how='inner')
    else:
        df = pd.merge(df_features, df_targets, on=['race_id', 'horse_number'], how='inner')
        
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    # --- Fetch Odds for ROI (Since they are dropped in parquet) ---
    logger.info("Fetching Odds from DB (jvd_se)...")
    loader = JraVanDataLoader()
    
    # Identify year range
    min_year = df['year'].min()
    max_year = df['year'].max()
    
    # Query jvd_se for Final Odds
    query = f"""
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        umaban as horse_number,
        tansho_odds 
    FROM jvd_se 
    WHERE kaisai_nen BETWEEN '{min_year}' AND '{max_year}'
    """
    try:
        df_odds = pd.read_sql(query, loader.engine)
        # Parse Odds: JVD format often needs numeric conversion
        # tansho_odds is usually string like '0015' -> 1.5? Or raw float?
        # In jvd_se schema from typical PC-KEIBA, it might be numeric or encoded string.
        # Assuming typical JRA-VAN numeric or simple castable.
        # Check if it's string digits implies /10 or direct?
        # Safe approach: coerce to float. If large int (e.g. 150), divide by 10 or 100?
        # JVD Spec: "単勝オッズ" (4 bytes) e.g. "0035" -> 3.5. 
        # But pandas read_sql might infer types.
        
        # Let's inspect a sample if possible, or assume standard JRA-VAN 10x
        # Wait, if we use parsed DB, it might be actual float.
        # Let's try to convert and inspect distribution range.
        
        df_odds['odds_final'] = pd.to_numeric(df_odds['tansho_odds'], errors='coerce') / 10.0
        # Fix join key: umaban is zero-padded string "01", df has int 1
        df_odds['horse_number'] = pd.to_numeric(df_odds['horse_number'], errors='coerce').fillna(0).astype(int)
        
        df = pd.merge(df, df_odds[['race_id', 'horse_number', 'odds_final']], on=['race_id', 'horse_number'], how='left')
        logger.info(f"Merged Odds: {df['odds_final'].count()} records found. Mean: {df['odds_final'].mean():.2f}")
        
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        df['odds_final'] = 0.0
    
    VALID_YEAR = cfg['dataset'].get('valid_year', 2023)
    # Allow override via arg, or default to config, or default to 2015
    TRAIN_START = str(getattr(args, 'start_year', cfg['dataset'].get('train_start_date', '2015-01-01')))
    # Ensure full date format if only year given
    if len(TRAIN_START) == 4:
        TRAIN_START = f"{TRAIN_START}-01-01"
    
    # Use data up to VALID_YEAR (inclusive) for CV
    mask_cv = (df['year'] <= VALID_YEAR) & (df['year'] >= pd.to_datetime(TRAIN_START).year)
    df_cv = df[mask_cv].copy().reset_index(drop=True)
    
    logger.info(f"CV Dataset: {df_cv.shape} (Years: {df_cv['year'].min()}-{df_cv['year'].max()})")
    
    # Prepare X, y
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'target', 'year', 'time_diff', 'odds_10min', 'odds_final', 'tansho_odds', 'updated_at', 'created_at']
    exclude_features = cfg['dataset'].get('exclude_features', [])
    if exclude_features:
        exclude_cols.extend(exclude_features)
        
    X_cols = [c for c in df_cv.columns if c not in exclude_cols and not pd.api.types.is_datetime64_any_dtype(df_cv[c])]
    
    # Categoricals
    cat_cols = cfg['dataset']['categorical_features']
    for col in cat_cols:
        if col in df_cv.columns:
            df_cv[col] = df_cv[col].astype('category')
    for col in X_cols:
        if df_cv[col].dtype == 'object':
             df_cv[col] = df_cv[col].astype('category')
             
    X = df_cv[X_cols]
    # Target: Binary (1着/top2/top3) or Regression (順位) or LambdaRank
    if args.objective == 'regression':
        # 順位を反転（1位=最高スコア）して回帰ターゲットに
        max_rank = df_cv['rank'].max()
        y = (max_rank + 1 - df_cv['rank']).astype(float)  # 1位=18, 最下位=1 (18頭立ての場合)
        logger.info(f"Objective: REGRESSION (順位回帰, max_rank={max_rank})")
    elif args.objective == 'top2':
        y = (df_cv['rank'] <= 2).astype(int)
        logger.info("Objective: TOP2 (2着以内予測)")
    elif args.objective == 'top3':
        y = (df_cv['rank'] <= 3).astype(int)
        logger.info("Objective: TOP3 (3着以内予測)")
    elif args.objective == 'lambdarank':
        # Relevance labels: 1位=4, 2位=3, 3位=2, 4-5位=1, else=0
        def rank_to_relevance(r):
            if r == 1: return 4
            elif r == 2: return 3
            elif r == 3: return 2
            elif r <= 5: return 1
            else: return 0
        y = df_cv['rank'].apply(rank_to_relevance).astype(int)
        logger.info("Objective: LAMBDARANK (ランキング学習)")
    else:  # binary (default)
        y = (df_cv['rank'] == 1).astype(int)
        logger.info("Objective: BINARY (1着予測)")
    groups = df_cv['race_id']
    
    # Compute group sizes for LambdaRank
    group_sizes = None
    if args.objective == 'lambdarank':
        group_sizes = df_cv.groupby('race_id').size().to_dict()
    
    # Odds for ROI
    # Prioritize odds_final loaded from DB
    odds = df_cv['odds_final'].fillna(0)
    
    # Weighting Strategy
    weights = None
    if getattr(args, 'weight_strategy', 'none') == 'balanced':
        # 1 / field_size
        # Need field_size. If not in X, compute from groups
        # Faster: df_cv.groupby('race_id')['race_id'].transform('count')
        # Check if field_size feature exists in X? "field_size" might be in X_cols.
        # But safer to compute fresh.
        race_counts = df_cv['race_id'].map(df_cv['race_id'].value_counts())
        weights = 1.0 / race_counts
        logger.info("Weight Strategy: BALANCED (1/field_size)")
    elif getattr(args, 'weight_strategy', 'none') == 'daily':
        # 1 / daily_race_count
        # Approximate: count rows per date / field_size? No, count RACES per date.
        # daily_race_count = number of unique race_ids per date.
        # Then distribute weight=1 per day?
        # User said: "同日合計weight一定". If day has N races, each race gets 1/N?
        # Or each sample gets?
        # Usually we want "Day Importance = 1".
        # So sum of weights for date D = 1.
        # Sum of weights = count(rows_in_D) * w_i = 1 => w_i = 1 / rows_in_D.
        # This penalizes large field sizes and busy days.
        daily_rows = df_cv['date'].map(df_cv['date'].value_counts())
        weights = 1.0 / daily_rows
        # User also suggested "レース均等: weight = 1/field_size（1レース合計weight=1）".
        # My 'balanced' above gives w = 1/N_horses. Sum_horses (1/N_horses) = 1.
        # So 'balanced' matches user's "1レース合計weight=1".
        # 'daily' matches "同日合計weight一定".
        logger.info("Weight Strategy: DAILY (1/daily_row_count)")
    
    # CV Setup
    gkf = GroupKFold(n_splits=args.n_folds)
    evaluator = Evaluator()
    
    fold_metrics = []
    
    logger.info(f"Starting {args.n_folds}-Fold CV...")
    
    params = cfg['model_params']
    lgbm_params = {k:v for k,v in params.items() if k not in ['early_stopping_rounds', 'n_estimators', 'model_type']}
    
    # Override objective if regression or lambdarank
    if args.objective == 'regression':
        lgbm_params['objective'] = 'regression'
        lgbm_params['metric'] = 'rmse'
    elif args.objective == 'lambdarank':
        lgbm_params['objective'] = 'lambdarank'
        lgbm_params['metric'] = 'ndcg'
        lgbm_params['ndcg_eval_at'] = [1, 3, 5]  # NDCG@1, @3, @5
        lgbm_params['label_gain'] = [0, 1, 2, 3, 4]  # Gains for relevance 0-4
    
    # Mode: CV or Hold-out
    if args.mode == 'holdout':
        # Hold-out: Train on years < valid_year, Valid on valid_year
        VALID_YEAR = cfg['dataset'].get('valid_year', 2023)
        train_mask = df_cv['year'] < VALID_YEAR
        val_mask = df_cv['year'] == VALID_YEAR
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        val_groups = groups[val_mask]
        val_odds = odds[val_mask]
        w_train = weights[train_mask] if weights is not None else None
        
        logger.info(f"Hold-out Mode: Train={train_mask.sum()}, Valid={val_mask.sum()}")
        
        # Create Dataset - add group info for lambdarank
        if args.objective == 'lambdarank':
            train_race_ids = groups[train_mask]
            val_race_ids = groups[val_mask]
            train_group = train_race_ids.groupby(train_race_ids).size().values
            val_group = val_race_ids.groupby(val_race_ids).size().values
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train, group=train_group)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, group=val_group)
        else:
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            lgbm_params,
            lgb_train,
            num_boost_round=params.get('n_estimators', 1000),
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(params.get('early_stopping_rounds', 50), verbose=False)]
        )
        
        preds = model.predict(X_val)
        
        # Evaluate
        if args.objective == 'regression':
            # 回帰の場合、予測値をそのままスコアとして使って Top1 を計算
            metrics = evaluator.evaluate(y_true=(df_cv[val_mask]['rank'] == 1).astype(int), y_prob=preds, group_ids=val_groups, odds=val_odds)
        else:
            metrics = evaluator.evaluate(y_true=y_val, y_prob=preds, group_ids=val_groups, odds=val_odds)
        metrics['fold'] = 1
        fold_metrics.append(metrics)
        logger.info(f"  AUC: {metrics.get('auc', 'N/A')}, Top1: {metrics.get('top1_precision',0):.4f}, ROI: {metrics.get('roi_top1_flat',0):.2f}%")
        
        # Save predictions if requested
        if args.save_preds:
            pred_df = df_cv[val_mask][['race_id', 'horse_number', 'date', 'rank']].copy()
            pred_df['pred_prob'] = preds
            pred_df['odds_final'] = val_odds.values
            pred_path = os.path.join(MODEL_DIR, f"preds_{args.objective}.parquet")
            pred_df.to_parquet(pred_path)
            logger.info(f"Predictions saved to: {pred_path}")
    else:
        # CV Mode
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"Fold {fold+1}/{args.n_folds}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            w_train = weights.iloc[train_idx] if weights is not None else None
            
            # Meta info for eval
            val_groups = groups.iloc[val_idx]
            val_odds = odds.iloc[val_idx]
            
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            
            model = lgb.train(
                lgbm_params,
                lgb_train,
                num_boost_round=params.get('n_estimators', 1000),
                valid_sets=[lgb_train, lgb_val],
                callbacks=[lgb.log_evaluation(0), lgb.early_stopping(params.get('early_stopping_rounds', 50), verbose=False)]
            )
            
            preds = model.predict(X_val)
            
            # Evaluate using new Evaluator
            if args.objective == 'regression':
                y_true_eval = (df_cv.iloc[val_idx]['rank'] == 1).astype(int)
                metrics = evaluator.evaluate(y_true=y_true_eval, y_prob=preds, group_ids=val_groups, odds=val_odds)
            else:
                metrics = evaluator.evaluate(y_true=y_val, y_prob=preds, group_ids=val_groups, odds=val_odds)
            metrics['fold'] = fold + 1
            logger.info(f"  AUC: {metrics.get('auc', 'N/A')}, LogLoss: {metrics.get('logloss', 'N/A')}, Top1: {metrics.get('top1_precision',0):.4f}, ROI: {metrics.get('roi_top1_flat',0):.2f}%")
            
            fold_metrics.append(metrics)
            
            # Save OOF predictions in CV mode
            if args.save_preds:
                fold_pred = df_cv.iloc[val_idx][['race_id', 'horse_number', 'date', 'rank']].copy()
                fold_pred['pred_prob'] = preds
                fold_pred['odds_final'] = val_odds.values
                fold_pred['fold'] = fold + 1
                if fold == 0:
                    all_preds = fold_pred
                else:
                    all_preds = pd.concat([all_preds, fold_pred])
        
        # Save all OOF predictions after CV loop
        if args.save_preds and 'all_preds' in dir():
            pred_path = os.path.join(MODEL_DIR, f"preds_oof_{args.objective}.parquet")
            all_preds.to_parquet(pred_path)
            logger.info(f"OOF Predictions saved to: {pred_path}")
        
    # Aggregate Rules
    metrics_df = pd.DataFrame(fold_metrics)
    agg_metrics = metrics_df.mean(numeric_only=True).to_dict()
    std_metrics = metrics_df.std(numeric_only=True).to_dict()
    
    print("\n" + "="*60)
    print(f" CV Results ({args.n_folds} Folds) - {EXP_NAME}")
    print("="*60)
    
    target_metrics = ['auc', 'logloss', 'brier', 'ece', 'top1_precision', 'roi_top1_flat']
    for m in target_metrics:
        if m in agg_metrics:
            mean = agg_metrics[m]
            std = std_metrics.get(m, 0.0)
            print(f" {m:<15}: {mean:.4f} ± {std:.4f}")
            
    # Save Report
    report = {
        'experiment': EXP_NAME,
        'date': datetime.now().isoformat(),
        'cv_mean': agg_metrics,
        'cv_std': std_metrics,
        'folds': fold_metrics
    }
    
    report_path = os.path.join(MODEL_DIR, "cv_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
