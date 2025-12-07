import argparse
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# srcパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.loader import InferenceDataLoader
from inference.preprocessor import InferencePreprocessor
from model.ensemble import EnsembleModel
from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='競馬予測推論スクリプト')
    parser.add_argument('--date', type=str, help='開催日 (YYYYMMDD)', required=False)
    parser.add_argument('--race_ids', type=str, nargs='+', help='レースIDリスト', required=False)
    parser.add_argument('--model', type=str, default='ensemble', choices=['lgbm', 'catboost', 'tabnet', 'ensemble'], help='使用するモデル')
    parser.add_argument('--version', type=str, default='v1', help='モデルバージョン')
    parser.add_argument('--model_dir', type=str, default='models', help='モデルディレクトリ')
    parser.add_argument('--output_dir', type=str, default='data/predictions', help='出力ディレクトリ')
    
    args = parser.parse_args()

    if not args.date and not args.race_ids:
        parser.error("--date または --race_ids のいずれかを指定してください。")

    try:
        # 1. データのロード
        logger.info("Step 1: データロード")
        loader = InferenceDataLoader()
        new_df = loader.load(target_date=args.date, race_ids=args.race_ids)

        if new_df.empty:
            logger.error("データが見つかりませんでした。終了します。")
            return

        # 2. 前処理
        logger.info("Step 2: 前処理 (過去データとの結合・特徴量生成)")
        preprocessor = InferencePreprocessor()
        X, ids = preprocessor.preprocess(new_df)

        if X.empty:
            logger.error("前処理後のデータが空です。終了します。")
            return

        # 3. モデルロード
        logger.info(f"Step 3: モデルロード ({args.model}, Version: {args.version})")
        
        model = None
        
        if args.model == 'ensemble':
            model = EnsembleModel()
            model_path = os.path.join(args.model_dir, f'ensemble_{args.version}.pkl')
            if not os.path.exists(model_path):
                 # Fallback to non-versioned if v1 not found? Or enforce?
                 # Try default name if versioned not found
                 default_path = os.path.join(args.model_dir, 'ensemble_model.pkl')
                 if os.path.exists(default_path):
                     logger.warning(f"指定バージョンのモデルが見つからないため、デフォルトモデルを使用します: {default_path}")
                     model_path = default_path
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            model.load_model(model_path)
            
        elif args.model == 'lgbm':
            model = KeibaLGBM()
            model_path = os.path.join(args.model_dir, f'lgbm_{args.version}.pkl')
            if not os.path.exists(model_path):
                 # Fallback
                 default_path = os.path.join(args.model_dir, 'lgbm.pkl')
                 if os.path.exists(default_path):
                     logger.warning(f"指定バージョンのモデルが見つからないため、デフォルトモデルを使用します: {default_path}")
                     model_path = default_path
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            model.load_model(model_path)
            
        elif args.model == 'catboost':
            model = KeibaCatBoost()
            model_path = os.path.join(args.model_dir, f'catboost_{args.version}.pkl')
            if not os.path.exists(model_path):
                 default_path = os.path.join(args.model_dir, 'catboost.pkl')
                 if os.path.exists(default_path):
                     logger.warning(f"指定バージョンのモデルが見つからないため、デフォルトモデルを使用します: {default_path}")
                     model_path = default_path
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            model.load_model(model_path)
            
        elif args.model == 'tabnet':
            model = KeibaTabNet()
            # TabNet saves as zip usually
            model_path = os.path.join(args.model_dir, f'tabnet_{args.version}.zip')
            
            if not os.path.exists(model_path):
                 default_path = os.path.join(args.model_dir, 'tabnet.zip')
                 if os.path.exists(default_path):
                     logger.warning(f"指定バージョンのモデルが見つからないため、デフォルトモデルを使用します: {default_path}")
                     model_path = default_path
                 else:
                     raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            model.load_model(model_path.replace('.zip', '.pkl'))
            
        
        # 4. 推論実行
        logger.info("Step 4: 推論実行")
        
        # 特徴量のフィルタリング (モデルが要求するものだけに絞る)
        if hasattr(model, 'model') and hasattr(model.model, 'feature_name'): # LightGBM
            required_features = model.model.feature_name()
            missing = set(required_features) - set(X.columns)
            if not missing:
                X = X[required_features]
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_names_'): # CatBoost
            required_features = model.model.feature_names_
            missing = set(required_features) - set(X.columns)
            if not missing:
                X = X[required_features]
                
        preds = model.predict(X)
        
        # 結果の整形
        results = ids.copy()
        results['score'] = preds
        
        # スコアに基づいて順位付け (Descending sort -> Rank 1)
        results['pred_rank'] = results.groupby('race_id')['score'].rank(ascending=False, method='min')

        # 5. 保存
        os.makedirs(args.output_dir, exist_ok=True)
        
        # ファイル名決定 (モデル名を含める)
        if args.date:
            filename = f"{args.date}_{args.model}_{args.version}.csv"
        else:
            filename = f"predictions_{args.model}_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = os.path.join(args.output_dir, filename)
        
        # 出力カラム整理
        out_cols = [
            'race_id', 'date', 'venue', 'race_number', 
            'pred_rank', 'horse_number', 'horse_name', 
            'score', 'jockey_id'
        ]
        results = results.sort_values(['race_id', 'pred_rank'])
        
        results[out_cols].to_csv(output_path, index=False)
        logger.info(f"予測結果を保存しました: {output_path}")
        
        # 簡易表示
        print("\n--- 予測結果サンプル (Top 3) ---")
        print(results[results['pred_rank'] <= 3].head(15))

    except Exception as e:
        logger.error(f"推論実行中にエラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    main()
