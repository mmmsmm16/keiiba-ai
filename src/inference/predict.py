
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet
from model.ensemble import EnsembleModel
from inference.preprocessor import InferencePreprocessor
from inference.loader import InferenceDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('--date', type=str, required=True, help='Target Date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='ensemble', choices=['lgbm', 'catboost', 'tabnet', 'ensemble'])
    parser.add_argument('--version', type=str, default='v5')
    args = parser.parse_args()
    
    # 1. データのロード
    loader = InferenceDataLoader()
    # 開催日を判定 (指定日が開催日でない場合、直近の過去開催日を取得したりするロジックが必要だが、ここでは指定日必須とする)
    target_date = args.date.replace('-', '') # YYYYMMDD
    
    logger.info(f"データロード中: {target_date}")
    raw_df = loader.load(target_date=target_date)
    
    if raw_df.empty:
        logger.warning(f"{target_date} のレースデータが見つかりません。")
        return

    # 2. 前処理
    logger.info("前処理を実行中...")
    preprocessor = InferencePreprocessor()
    # 過去データ(学習用)を使って特徴量エンジニアリング
    X, ids = preprocessor.preprocess(raw_df)
    processed_df = pd.concat([ids, X], axis=1)
    
    if processed_df.empty:
        logger.error("前処理に失敗しました (データが空です)。")
        return

    # 3. モデルのロード
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    model = None
    
    logger.info(f"モデルをロード中: {args.model} ({args.version})")
    try:
        if args.model == 'ensemble':
            model = EnsembleModel()
            path = os.path.join(model_dir, f'ensemble_{args.version}.pkl')
            if not os.path.exists(path): path = os.path.join(model_dir, 'ensemble_model.pkl')
            model.load_model(path)
        elif args.model == 'lgbm':
            model = KeibaLGBM()
            path = os.path.join(model_dir, f'lgbm_{args.version}.pkl')
            if not os.path.exists(path): path = os.path.join(model_dir, 'lgbm.pkl')
            model.load_model(path)
        elif args.model == 'catboost':
            model = KeibaCatBoost()
            path = os.path.join(model_dir, f'catboost_{args.version}.pkl')
            if not os.path.exists(path): path = os.path.join(model_dir, 'catboost.pkl')
            model.load_model(path)
        elif args.model == 'tabnet':
            model = KeibaTabNet()
            path = os.path.join(model_dir, f'tabnet_{args.version}.zip')
            if not os.path.exists(path): path = os.path.join(model_dir, 'tabnet.zip')
            model.load_model(path.replace('.zip', '.pkl'))
    except Exception as e:
        logger.error(f"モデルロード失敗: {e}")
        return

    # 4. Calibrator のロード (New for Phase 13)
    logger.info("確率補正器 (Calibrator) をロード中...")
    calibrator = None
    calib_path = os.path.join(model_dir, 'calibrator.pkl')
    if os.path.exists(calib_path):
        from model.calibration import ProbabilityCalibrator
        calibrator = ProbabilityCalibrator()
        try:
            calibrator.load(calib_path)
        except Exception as e:
            logger.warning(f"Calibratorのロードに失敗: {e}")
            calibrator = None
    else:
        logger.warning("Calibratorが見つかりません。生の確率を使用します。")

    # 5. 予測実行
    logger.info("予測を実行中...")
    # Feature selection
    feature_cols = None
    
    if args.model == 'lgbm':
        # Check various attributes
        bst = model.model
        logger.info(f"モデル内部タイプ: {type(bst)}")
        
        if hasattr(bst, 'feature_name'):
            # Booster
            try:
                feature_cols = bst.feature_name()
            except:
                feature_cols = bst.feature_name
        elif hasattr(bst, 'feature_name_'):
            # Sklearn API
            feature_cols = bst.feature_name_
        elif hasattr(bst, 'booster_'):
            # Sklearn API
            feature_cols = bst.booster_.feature_name()
            
    elif args.model == 'catboost' and hasattr(model.model, 'feature_names_'):
        feature_cols = model.model.feature_names_
    
    # Fallback to dataset metadata
    if feature_cols is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                datasets = pickle.load(f)
            if datasets['train']['X'] is not None:
                feature_cols = datasets['train']['X'].columns.tolist()

    if feature_cols:
        logger.info(f"モデル定義の特徴量数: {len(feature_cols)}")
        # Deduplicate processed_df columns just in case
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        
        missing = set(feature_cols) - set(processed_df.columns)
        if missing: logger.warning(f"入力に欠損している特徴量: {missing}")
        for c in missing: processed_df[c] = 0
        X_pred = processed_df[feature_cols]
    else:
        logger.warning("モデルの特徴量名が見つかりません。全ての数値カラムを使用します。")
        # Fallback risky
        X_pred = processed_df.select_dtypes(include=[np.number])
    
    logger.info(f"{X_pred.shape[1]} 個の特徴量をモデルに入力します。")
    scores = model.predict(X_pred)
    processed_df['score'] = scores
    
    # Softmax Prob
    processed_df['prob'] = processed_df.groupby('race_id')['score'].transform(lambda x: softmax(x))
    
    # Calibrated Prob
    if calibrator:
        processed_df['calibrated_prob'] = calibrator.predict(processed_df['prob'].values)
        ev_prob = processed_df['calibrated_prob']
    else:
        processed_df['calibrated_prob'] = processed_df['prob']
        ev_prob = processed_df['prob']
        
    # Expected Value
    if 'odds' in processed_df.columns:
        processed_df['expected_value'] = ev_prob * processed_df['odds'].fillna(0)
    else:
        processed_df['expected_value'] = 0

    # Save Results
    output_dir = os.path.join(os.path.dirname(__file__), '../../experiments/predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{target_date}_{args.model}.csv")
    
    # Output columns
    out_cols = [
        'race_id', 'horse_number', 'horse_name', 
        'score', 'prob', 'calibrated_prob', 'expected_value', 
        'odds', 'popularity', 'rank', # rank might be missing for future
        'title', 'venue', 'race_number' # Metadata for reporting
    ]
    # Filter existing
    out_cols = [c for c in out_cols if c in processed_df.columns]
    
    processed_df[out_cols].to_csv(save_path, index=False)
    logger.info(f"予測結果を保存しました: {save_path}")
    
    # Also print top picks
    print("\n--- 推奨馬 (EV Top 5) ---")
    picks = processed_df.sort_values('expected_value', ascending=False).head(5)
    print(picks[['race_id', 'horse_number', 'horse_name', 'calibrated_prob', 'odds', 'expected_value']])
    
    # 最適戦略による買い目推奨
    print("\n" + "="*60)
    print("🎯 最適戦略による買い目推奨")
    print("="*60)
    
    try:
        from inference.optimal_strategy import OptimalStrategy
        strategy = OptimalStrategy()
        
        for race_id, group in processed_df.groupby('race_id'):
            sorted_g = group.sort_values('score', ascending=False)
            
            horse_numbers = sorted_g['horse_number'].astype(int).tolist()
            scores = sorted_g['score'].tolist()
            popularities = sorted_g['popularity'].fillna(99).astype(int).tolist()
            odds = sorted_g['odds'].fillna(0).tolist()
            
            if len(horse_numbers) < 6:
                continue
            
            race_info = {
                'venue': sorted_g.iloc[0].get('venue', ''),
                'race_number': sorted_g.iloc[0].get('race_number', ''),
                'title': sorted_g.iloc[0].get('title', '')
            }
            
            rec = strategy.analyze_race(horse_numbers, scores, popularities, odds)
            
            # 推奨がある場合のみ表示
            if rec.bet_type != "skip" or rec.confidence == "warning":
                print(f"\n{strategy.format_notification(rec, race_info)}")
    except Exception as e:
        logger.warning(f"最適戦略の計算に失敗: {e}")

if __name__ == "__main__":
    main()
