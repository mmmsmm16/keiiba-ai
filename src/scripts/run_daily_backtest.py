
import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reporting.html_generator import HTMLReportGenerator
from model.evaluate import load_payout_data
from inference.strategy import WeeklyBettingStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_bets(df_bets, date_str):
    """
    実際のレース結果をロードし、損益を計算します。
    """
    year = date_str.split('-')[0]
    logger.info(f"{year}年の払戻データをロード中...")
    
    # 払戻データをロード (全データをロードするが問題ない)
    payout_df = load_payout_data(years=[int(year)])
    
    # BettingOptimizer を使用して払戻データをパース
    # BettingOptimizer には全てのレースIDを含むダミーの DataFrame が必要
    # df_bets has 'race_id'.
    from model.betting_strategy import BettingOptimizer
    dummy_df = df_bets[['race_id']].drop_duplicates().copy()
    
    optimizer = BettingOptimizer(dummy_df, payout_df)
    
    # 評価実行
    results = []
    total_cost = 0
    total_return = 0
    
    for _, row in df_bets.iterrows():
        rid = str(row['race_id'])
        payouts = optimizer.payout_map.get(rid, {})
        
        bet_type = row['bet_type'] # '馬連' -> 'umaren'?
        # We need to map '馬連' to 'umaren' key in map.
        # run_weekly.py sets 'bet_type' = '馬連'.
        # optimizer map keys: 'umaren', 'sanrenpuku', etc.
        
        btype_key = 'umaren' # Hardcoded as we only support umaren for now
        
        my_return = 0
        hit_eyes = []
        
        # Parse Eyes?
        # row['eyes'] is HTML string. We need raw data.
        # Strategy returns 'opponents' and 'axis_horse_num'.
        # We can reconstruct tickets.
        
        axis = int(row['axis_horse_num'])
        opps = row['opponents'] # list of ints
        unit = row['unit_bet']
        
        race_tickets = []
        for o in opps:
            comb = sorted([axis, o])
            key = f"{comb[0]:02}{comb[1]:02}"
            race_tickets.append(key)
            
        # Check Hit
        if btype_key in payouts:
            pmap = payouts[btype_key] # {key: return}
            for t in race_tickets:
                if t in pmap:
                    ret = int(pmap[t])
                    # Return is per 100 yen.
                    # profit = (unit / 100) * ret
                    pay = (unit / 100) * ret
                    my_return += pay
                    hit_eyes.append(t)
                    
        total_cost += row['cost']
        total_return += my_return
        
        results.append({
            'return': int(my_return),
            'hit': (my_return > 0),
            'profit': int(my_return - row['cost'])
        })
        
    df_results = pd.DataFrame(results)
    return pd.concat([df_bets, df_results], axis=1), total_cost, total_return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--bankroll', type=int, default=None)
    args = parser.parse_args()
    
    date_str = args.date
    config = load_config()
    if args.bankroll:
        config['betting']['initial_bankroll'] = args.bankroll
        
    # 1. Run Inference (or Load)
    # Check if prediction exists
    model_type = config['betting'].get('model_type', 'lgbm')
    model_version = config['betting'].get('model_version', 'v4_emb') # Use default or specific?
    # For backtest, we might want to use "weekly_update" if it exists, or just v4_emb.
    # Let's assume v4_emb for stable backtest unless specified.
    
    pred_path = f"experiments/predictions/{date_str.replace('-','')}_{model_type}.csv"
    
    if not os.path.exists(pred_path):
        logger.info(f"予測データが見つかりません。{date_str} の推論を実行します...")
        cmd = f"python src/inference/predict.py --date {date_str} --model {model_type} --version {model_version}"
        if os.system(cmd) != 0:
            logger.error("推論に失敗しました。")
            return

    df_pred = pd.read_csv(pred_path)
    if df_pred.empty:
        logger.error("予測データが空です。")
        return
        
    # 2. Strategy
    logger.info("賭け戦略を適用中...")
    # Need to instantiate strategy with config
    strategy = WeeklyBettingStrategy(config)
    df_bets = strategy.apply(df_pred)
    
    if df_bets.empty:
        logger.info("賭け対象が生成されませんでした。")
        return
        
    # Merge metadata from df_pred to df_bets if available
    # df_bets cols: race_id, axis, opps...
    # df_pred cols: race_id, title, venue...
    if 'title' in df_pred.columns:
        meta_df = df_pred[['race_id', 'title', 'venue', 'race_number']].drop_duplicates()
        df_bets = pd.merge(df_bets, meta_df, on='race_id', how='left')

    # 3. Simulation / Evaluation
    logger.info("結果を評価中...")
    df_evaluated, t_cost, t_ret = evaluate_bets(df_bets, date_str)
    
    t_profit = t_ret - t_cost
    roi = (t_ret / t_cost * 100) if t_cost > 0 else 0
    
    logger.info(f"投資額: {t_cost:,} | 払戻: {t_ret:,.0f} | 利益: {t_profit:,.0f} | 回収率: {roi:.1f}%")
    
    # 4. Report
    # We need to update HTML Generator to accept evaluated DF
    # Hack: Add 'result_str' to 'eyes' or similar?
    # Better: Update HTMLGenerator to look for 'return', 'profit' cols.
    
    # Or just modify 'eyes' column in df_evaluated to include result string.
    for idx, row in df_evaluated.iterrows():
        res_html = ""
        if row['hit']:
            res_html = f"<br><span style='color:red; font-weight:bold;'>大的中! 払戻: ¥{row['return']:,} (+{row['profit']:,})</span>"
        else:
            res_html = f"<br><span style='color:blue;'>ハズレ (-{row['cost']:,})</span>"
            
        df_evaluated.at[idx, 'eyes'] = row['eyes'] + res_html

    # Generate
    genes = HTMLReportGenerator(output_dir='reports/backtest')
    # Pass race_data (df_pred) for Entry Table
    save_path = genes.generate(df_evaluated, config['betting']['initial_bankroll'], race_data=df_pred, date_str=date_str)
    
    # Rename file to include verification tag? 
    # HTMLGenerator uses date. Let's rename it manually if needed or update HTMLGenerator.
    # Default is report_{date}.html.
    
    logger.info(f"検証レポートを保存しました: {save_path}")

if __name__ == "__main__":
    main()
