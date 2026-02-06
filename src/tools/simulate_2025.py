import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import logging

# プロジェクトルートへのパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # v05_sireをデフォルトとしつつ、引数でも指定可能に
    default_config = "config/experiments/exp_v05_sire.yaml"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get('experiment_name')
    feature_blocks = config.get('features', [])
    
    logger.info(f"🚀 Simulation 2025 for {exp_name}")
    logger.info(f"Features: {feature_blocks}")

    # 1. Load Data (2025 Full Year)
    loader = JraVanDataLoader()
    start_date = '2025-01-01'
    end_date = '2025-12-31'
    
    logger.info(f"Loading data ({start_date} ~ {end_date})...")
    # history_start_dateを2025-01-01にすると、過去走集計用のデータが足りなくなる可能性がある
    # FeaturePipelineは内部で過去データを必要とする場合があるが、
    # loaderは指定期間のデータしか返さない。
    # しかし FeaturePipeline の実装を見ると、渡された DF 内でのみ集計を行っている (shift, rolling)。
    # つまり、2025年のデータだけを渡すと、1月のレースの「過去5走」は欠損する。
    # 正確なシミュレーションのためには、2024年以前のデータも含めてロードし、
    # 特徴量生成後に 2025年分のみをフィルタリングする必要がある。
    
    # 余裕を持って1年前からロード
    load_start = '2024-01-01'
    logger.info(f"  Fetching data from {load_start} (for history context)...")
    raw_df = loader.load(history_start_date=load_start, end_date=end_date, jra_only=True)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Generate Features
    pipeline = FeaturePipeline(cache_dir="data/features")
    # force=Falseでキャッシュがあれば使う
    df_features = pipeline.load_features(clean_df, feature_blocks)
    
    # 3. Merge Metadata (Odds, Result, Date) for Simulation
    # df_features は feature columns + keys のみ
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'odds', 'horse_name']
    # unique keyでマージ
    df_sim = pd.merge(
        df_features, 
        clean_df[meta_cols], 
        on=['race_id', 'horse_number'], 
        how='inner'
    )
    
    # 4. Filter for 2025
    df_sim['date'] = pd.to_datetime(df_sim['date'])
    df_2025 = df_sim[(df_sim['date'] >= start_date) & (df_sim['date'] <= end_date)].copy()
    
    logger.info(f"Simulation Targets: {len(df_2025)} rows (2025)")
    
    # 5. Load Model & Predict
    model_path = f"models/experiments/{exp_name}/model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 特徴量カラムのみ抽出 (df_featuresのカラム - keys)
    # df_featuresの要素順序が変わっていると危険なので、
    # load_featuresの戻り値のカラムを使うのが安全（meta_colsは除外）
    # feature_cols = [c for c in df_features.columns if c not in ['race_id', 'horse_number', 'horse_id']]
    # しかし df_2025 には meta_cols が混ざっている。
    # model.predict に渡す X は、学習時と同じカラム構成でなければならない。
    
    # Feature Alignment
    model_features = model.feature_name()
    logger.info(f"Model expects {len(model_features)} features: {model_features}")
    
    # 型変換 (ageは数値のはずだがobjectになる場合があるため強制変換)
    if 'age' in df_2025.columns:
        df_2025['age'] = pd.to_numeric(df_2025['age'], errors='coerce')

    # カラム存在チェックと並べ替え
    X = pd.DataFrame(index=df_2025.index)
    for feat in model_features:
        if feat in df_2025.columns:
            X[feat] = df_2025[feat]
        else:
            logger.warning(f"Feature {feat} is missing in data. Filling with 0.")
            X[feat] = 0
            
    # 余分なカラムは自動的に除外される (Xは作成時空なので)
    
    logger.info("Predicting...")
    # Binaryモデルなので確率が出力されるはず (predict vs predict_proba check)
    # LightGBM sklearn APIなら predict_probaだが、Native API (train) なら predict が確率
    # run_experiment.py では lgb.train を使用 -> predict returns probability
    probs = model.predict(X)
    df_2025['pred_prob'] = probs
    
    # 6. Simulation (Flat Betting, EV >= 1.0)
    # EV = prob * odds
    # odds は 単勝オッズ
    df_2025['ev'] = df_2025['pred_prob'] * df_2025['odds']
    
    # 購入条件
    # odds > 1.0 (元返し除外), EV >= 1.0 (期待値1以上)
    # ※ 実運用では人気薄すぎると荒れるので足切りすることもあるが、今回は純粋な性能を見る
    bets = df_2025[
        (df_2025['ev'] >= 1.0) & 
        (df_2025['odds'].notna()) & 
        (df_2025['rank'] > 0) # 結果があるもの
    ].copy()
    
    logger.info(f"Bet Candidates: {len(bets)} / {len(df_2025)}")
    
    # 月次集計
    bets['month'] = bets['date'].dt.strftime('%Y-%m')
    bets['cost'] = 100
    bets['return'] = np.where(bets['rank'] == 1, bets['odds'] * 100, 0)
    
    monthly_stats = bets.groupby('month').agg({
        'race_id': 'count', # Bet Count
        'cost': 'sum',
        'return': 'sum'
    }).rename(columns={'race_id': 'bets'})
    
    monthly_stats['net'] = monthly_stats['return'] - monthly_stats['cost']
    monthly_stats['roi'] = (monthly_stats['return'] / monthly_stats['cost']) * 100
    
    # Total Stats
    total_bets = monthly_stats['bets'].sum()
    total_cost = monthly_stats['cost'].sum()
    total_return = monthly_stats['return'].sum()
    total_net = total_return - total_cost
    total_roi = (total_return / total_cost) * 100 if total_cost > 0 else 0
    
    # 結果表示
    print("\nXXX Simulation Result 2025 (Flat Betting, EV>=1.0) XXX")
    print(f"Model: {exp_name}")
    print(monthly_stats[['bets', 'return', 'net', 'roi']])
    print("-" * 50)
    print(f"Yearly Total:")
    print(f"  Bets: {total_bets}")
    print(f"  Cost: {total_cost:,.0f} JPY")
    print(f"  Return: {total_return:,.0f} JPY")
    print(f"  Net: {total_net:,.0f} JPY")
    print(f"  ROI: {total_roi:.2f}%")
    print("-" * 50)
    
    # ログ保存
    out_path = f"reports/simulation_{exp_name}_2025.csv"
    bets.to_csv(out_path, index=False)
    logger.info(f"Detailed simulation log saved to {out_path}")

if __name__ == "__main__":
    main()
