import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from typing import List, Dict

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.utils.payout_loader import PayoutLoader
from src.models.calibration import ProbabilityCalibrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_predictions(years: List[int], output_path: str):
    all_results = []
    
    loader = JraVanDataLoader()
    payout_loader = PayoutLoader()
    cleanser = DataCleanser()
    pipeline = FeaturePipeline()
    
    # 全期間の払戻情報を取得
    payout_map = payout_loader.load_payout_map(years)
    
    for year in years:
        logger.info(f"--- Exporting {year} predictions ---")
        config_path = f"config/experiments/exp_v13_e1_calib_{year}.yaml"
        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path}")
            continue
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        artifact_dir = f"models/experiments/v13_e1_calib_{year}"
        model_path = os.path.join(artifact_dir, 'model.pkl')
        calibrator_path = os.path.join(artifact_dir, 'calibrator.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(calibrator_path):
            logger.warning(f"Model or Calibrator not found in {artifact_dir}")
            continue
            
        # 1. データのロード (10分前オッズを含む)
        # バリデーションデータの対象期間を特定
        valid_year = config['dataset']['valid_year']
        df = loader.load(
            jra_only=config['dataset'].get('jra_only', True),
            history_start_date=f"{valid_year}-01-01",
            end_date=f"{valid_year}-12-31",
            skip_odds=False # 10分前オッズを取得
        )
        
        # クレンジング
        df = cleanser.cleanse(df)
        
        # 特徴量生成 (JRA用)
        # NOTE: 必要な特徴量ブロックのみを生成する
        feature_blocks = config['features']
        feature_df = pipeline.load_features(df, feature_blocks)
        # 必要なメタデータカラムを保持しつつ特徴量を結合
        df = pd.merge(df, feature_df, on=['race_id', 'horse_number'], how='inner', suffixes=('', '_dup'))
        # 重複カラム（horse_id等）を整理
        df = df[[c for c in df.columns if not c.endswith('_dup')]]
        
        # 特徴量セットの確定 (run_experiment.py と全く同じロジックを再現)
        drop_cols = ['race_id', 'horse_id', 'date', 'target', 'year', 'y', 'rank', 'odds', 'target_win', 'target_top3', 'is_win', 'is_top3']
        # features list からこれらを除外
        feature_cols = [c for c in df.columns if c not in drop_cols and c not in ['rank_str', 'raw_time', 'start_time_str', 'horse_name']]
        
        # モデルロード
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
            
        # 予測用データ準備
        X = df[feature_cols].copy()
        
        # run_experiment.py:138-141 のカテゴリカル検知ロジックを再現
        cat_features_cfg = config['dataset'].get('categorical_features', [])
        auto_cat = [c for c in X.columns if X[c].dtype == 'object']
        cat_features = list(set(cat_features_cfg + auto_cat))
        # フィルタリング
        cat_features = [c for c in cat_features if c in feature_cols]
        cat_features = [c for c in cat_features if c not in ['race_id', 'date', 'horse_id', 'target', 'year', 'y', 'rank', 'odds', 'target_win', 'target_top3']]
        
        logger.info(f"Replicated {len(cat_features)} categorical features.")

        # 型の統一
        for col in X.columns:
            if col in cat_features:
                X[col] = X[col].astype('category')
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # モデルの期待するカラム順序と一致することを確認 (Booster.predict は名前で合わせることもあるが、念の為)
        model_features = model.feature_name()
        if set(model_features) != set(X.columns):
            logger.warning(f"Feature mismatch! Model expects {len(model_features)}, X has {len(X.columns)}")
            missing = set(model_features) - set(X.columns)
            if missing: logger.warning(f"Missing from X: {missing}")
            extra = set(X.columns) - set(model_features)
            if extra: logger.warning(f"Extra in X: {extra}")
            
        # Xをモデルの順序に再配置 (足りないものはNaN)
        X_final = pd.DataFrame(index=X.index)
        for col in model_features:
            if col in X.columns:
                X_final[col] = X[col]
            else:
                X_final[col] = np.nan
            
        # 2. 推論
        try:
            p_raw = model.predict(X_final)
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            raise e
            
        p_cal = calibrator.predict(p_raw)
        
        # 3. カラム統合
        # DataCleanser で odds_10min_str -> odds_10min に変換されているのでそれを使用
        # また、JraVanDataLoader で race_number が取得されている
        available_cols = ['race_id', 'horse_id', 'date', 'venue', 'race_number', 'horse_number', 'rank', 'odds_10min',
                          'odds_place_min', 'odds_place_max', 'odds_umaren_str']
        res_cols = [c for c in available_cols if c in df.columns]
        res = df[res_cols].copy()
        
        # 名前の一致（ユーザー指定）
        if 'race_number' in res.columns: res.rename(columns={'race_number': 'race_no'}, inplace=True)
        if 'rank' in res.columns: res.rename(columns={'rank': 'finish_pos'}, inplace=True)
        if 'odds_10min' in res.columns: res.rename(columns={'odds_10min': 'odds_win_pre'}, inplace=True)
        if 'odds_place_min' in res.columns: res.rename(columns={'odds_place_min': 'odds_place_pre_min'}, inplace=True)
        if 'odds_place_max' in res.columns: res.rename(columns={'odds_place_max': 'odds_place_pre_max'}, inplace=True)
        # Use average place odds for single scalar? Or keep min/max?
        # User asked for `odds_place_pre`. Let's calculate average or keep min/max.
        # User request: `odds_place_pre` (10分前 複勝オッズ).
        # Let's add an average column `odds_place_pre` for simplicity in simple EV calc, but keep min/max.
        if 'odds_place_pre_min' in res.columns and 'odds_place_pre_max' in res.columns:
             res['odds_place_pre'] = (res['odds_place_pre_min'] + res['odds_place_pre_max']) / 2.0
             
        if 'odds_umaren_str' in res.columns: res.rename(columns={'odds_umaren_str': 'odds_umaren_pre'}, inplace=True)
        
        res['year_valid'] = year
        res['p_raw'] = p_raw
        res['p_cal'] = p_cal
        
        # 4. レース内順位 (rank_pred)
        res['rank_pred'] = res.groupby('race_id')['p_cal'].rank(ascending=False, method='min')
        
        # 5. 払戻金の結合
        def get_win_payout(row):
            rid = row['race_id']
            hno = str(row['horse_number']).zfill(2)
            if rid in payout_map:
                return payout_map[rid]['tansho'].get(hno, 0)
            return 0
            
        def get_place_payout(row):
            rid = row['race_id']
            hno = str(row['horse_number']).zfill(2)
            if rid in payout_map:
                # 複勝は複数のあたり馬番がある
                return payout_map[rid]['fukusho'].get(hno, 0)
            return 0
            
        res['payout_win'] = res.apply(get_win_payout, axis=1)
        res['payout_place'] = res.apply(get_place_payout, axis=1)
        
        all_results.append(res)
        
    if not all_results:
        logger.error("No results to export.")
        return
        
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 重複削除や型整理
    final_df.to_parquet(output_path, index=False)
    logger.info(f"✅ Exported {len(final_df)} records to {output_path}")
    
    # Verification summary
    logger.info("Verification Summary:")
    for year in years:
        y_df = final_df[final_df['year_valid'] == year]
        if len(y_df) == 0: continue
        
        # binary target for recall calculation
        y_df['is_top3'] = (y_df['finish_pos'] <= 3).astype(int)
        
        # Recall@5 by race
        def calc_recall5_race(group):
            if group['is_top3'].sum() == 0: return np.nan
            top5 = group.sort_values('p_cal', ascending=False).head(5)
            return top5['is_top3'].sum() / min(3, group['is_top3'].sum()) # Normalize by min(3, actual_winners)
            
        # Simplified Recall (any top-3 horse in top 5 pred)
        recalls = y_df.groupby('race_id').apply(lambda g: g.sort_values('p_cal', ascending=False).head(5)['is_top3'].sum() / g['is_top3'].sum() if g['is_top3'].sum() > 0 else np.nan)
        logger.info(f"  Year {year}: Recall@5 (Simple Mean) = {recalls.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/simulations/v13_e1_predictions_2022_2024.parquet")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_predictions(args.years, args.output)
