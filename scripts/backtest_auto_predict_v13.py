"""
Auto Predict V13 バックテスト (parquet版)
2025年の三連複BOX4戦略のROIを計算

preprocessed_data_v11.parquet を使用 (paper_trade_run.pyと同じ)

Usage:
    docker compose exec app python scripts/backtest_auto_predict_v13.py
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from itertools import combinations

import lightgbm as lgb
from scipy.special import expit
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定数
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/v13_market_residual')
PARQUET_PATH = os.path.join(os.path.dirname(__file__), '../data/processed/preprocessed_data_v11.parquet')

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


def load_models():
    models = []
    for fold in ['2022', '2023', '2024']:
        path = os.path.join(MODEL_DIR, f'v13_fold_{fold}.txt')
        if os.path.exists(path):
            models.append(lgb.Booster(model_file=path))
    return models


def load_payouts(engine, year='2025'):
    """三連複の払戻データを取得"""
    query = f"""
    SELECT 
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
        haraimodoshi_sanrenpuku_1a,
        haraimodoshi_sanrenpuku_1b,
        haraimodoshi_sanrenpuku_2a,
        haraimodoshi_sanrenpuku_2b,
        haraimodoshi_sanrenpuku_3a,
        haraimodoshi_sanrenpuku_3b
    FROM jvd_hr
    WHERE kaisai_nen = '{year}'
      AND keibajo_code BETWEEN '01' AND '10'
    """
    df = pd.read_sql(query, engine)
    
    payouts = {}
    for _, row in df.iterrows():
        race_id = row['race_id']
        payouts[race_id] = []
        
        for i in range(1, 4):
            try:
                combo_str = str(row[f'haraimodoshi_sanrenpuku_{i}a'] or '')
                payout_str = str(row[f'haraimodoshi_sanrenpuku_{i}b'] or '')
                
                if not combo_str or len(combo_str) < 6:
                    continue
                
                h1 = int(combo_str[0:2])
                h2 = int(combo_str[2:4])
                h3 = int(combo_str[4:6])
                payout = int(payout_str) if payout_str.isdigit() else 0
                
                if h1 > 0 and h2 > 0 and h3 > 0 and payout > 0:
                    combo = tuple(sorted([h1, h2, h3]))
                    payouts[race_id].append((combo, payout))
            except:
                continue
    
    return payouts


def run_backtest():
    logger.info("=== Auto Predict V13 Backtest (parquet版) ===")
    
    engine = get_db_engine()
    models = load_models()
    logger.info(f"Loaded {len(models)} models")
    
    # parquet からデータロード
    logger.info(f"Loading parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Parquet rows: {len(df)}")
    
    # 2025年のデータをフィルタ
    df['race_id_str'] = df['race_id'].astype(str)
    df_2025 = df[df['race_id_str'].str.startswith('2025')].copy()
    logger.info(f"2025 rows: {len(df_2025)}, races: {df_2025['race_id'].nunique()}")
    
    if df_2025.empty:
        logger.error("2025年のデータがありません")
        return
    
    # 払戻データロード
    payouts = load_payouts(engine, '2025')
    logger.info(f"Payouts: {len(payouts)} races")
    
    # モデル特徴量カラム
    feature_cols = models[0].feature_name()
    
    # 特徴量準備
    for c in feature_cols:
        if c not in df_2025.columns:
            df_2025[c] = 0
    
    X = df_2025[feature_cols].fillna(0)
    
    # Ensemble prediction
    logger.info("Predicting...")
    preds = []
    for model in models:
        preds.append(model.predict(X))
    avg_pred = np.mean(preds, axis=0)
    
    # paper_trade_run.py と同じ: expit → softmax
    df_2025['prob_raw'] = expit(avg_pred)
    
    def softmax_race(group):
        exp_vals = np.exp(group - group.max())
        return exp_vals / exp_vals.sum()
    
    df_2025['prob'] = df_2025.groupby('race_id')['prob_raw'].transform(softmax_race)
    
    # BOX4 チケット生成と的中判定
    total_stake = 0
    total_payout = 0
    total_races = 0
    total_hits = 0
    
    logger.info("Generating tickets and checking hits...")
    for race_id in df_2025['race_id'].unique():
        race_df = df_2025[df_2025['race_id'] == race_id].copy()
        
        if len(race_df) < 4:
            continue
        
        # Top4馬
        top4 = race_df.nlargest(4, 'prob')['horse_number'].astype(int).tolist()
        
        # BOX4チケット (4点)
        tickets = list(combinations(top4, 3))
        stake = len(tickets) * 100
        total_stake += stake
        total_races += 1
        
        # 的中チェック
        race_id_str = str(race_id)
        if race_id_str in payouts:
            for ticket in tickets:
                combo = tuple(sorted(ticket))
                for winning_combo, payout in payouts[race_id_str]:
                    if combo == winning_combo:
                        total_payout += payout
                        total_hits += 1
                        break
    
    # 結果
    roi = (total_payout / total_stake * 100) if total_stake > 0 else 0
    hit_rate = (total_hits / total_races * 100) if total_races > 0 else 0
    
    print("\n" + "=" * 50)
    print("=== Auto Predict V13 Backtest Results (parquet版) ===")
    print("=" * 50)
    print(f"Total Races: {total_races}")
    print(f"Total Stake: ¥{total_stake:,}")
    print(f"Total Payout: ¥{total_payout:,}")
    print(f"Profit: ¥{total_payout - total_stake:,}")
    print(f"ROI: {roi:.1f}%")
    print(f"Hit Rate: {hit_rate:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_backtest()
