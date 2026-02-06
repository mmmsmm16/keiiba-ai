import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from scipy.special import expit, logit

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.scripts.auto_predict_v13 import get_db_engine

def load_data(pred_path: str, year: str) -> pd.DataFrame:
    logger.info(f"Loading predictions from {pred_path}...")
    preds = pd.read_parquet(pred_path)
    
    logger.info("Loading actual results from DB...")
    engine = get_db_engine()
    query = f"""
    SELECT 
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
        umaban as horse_number,
        kakutei_chakujun as actual_rank,
        tansho_odds as final_odds_raw,
        tansho_ninkijun as final_popularity
    FROM jvd_se
    WHERE kaisai_nen = '{year}'
    """
    results = pd.read_sql(query, engine)
    
    # Preprocess Results
    results['horse_number'] = pd.to_numeric(results['horse_number'], errors='coerce')
    results['final_odds'] = pd.to_numeric(results['final_odds_raw'], errors='coerce') / 10.0
    
    # Merge
    preds['race_id'] = preds['race_id'].astype(str)
    preds['horse_number'] = preds['horse_number'].astype(int)
    results['race_id'] = results['race_id'].astype(str)
    results['horse_number'] = results['horse_number'].astype(int)
    
    df = pd.merge(preds, results, on=['race_id', 'horse_number'], how='inner')
    
    # Target
    def get_target(rank):
        try: return 1 if int(rank) == 1 else 0
        except: return 0
    df['target'] = df['actual_rank'].apply(get_target)
    
    return df

def analyze_performance(df: pd.DataFrame):
    """
    A. Prediction Performance (Market vs Model)
    """
    # 1. Market Prob (T-10m)
    # Using p_market_tminus10m from parquet
    if 'p_market_tminus10m' not in df.columns:
        # Recalculate if missing
        df['p_market_tminus10m'] = 1.0 / df['odds_tminus10m']
        # Normalize per race
        sums = df.groupby('race_id')['p_market_tminus10m'].transform('sum')
        df['p_market_tminus10m'] = df['p_market_tminus10m'] / sums
        
    y_true = df['target'].values
    y_market = df['p_market_tminus10m'].values
    
    col_pred = 'prob_residual_softmax' if 'prob_residual_softmax' in df.columns else 'pred_prob'
    y_model = df[col_pred].values
    
    metrics = {}
    for name, y_pred in [('Market (T-10m)', y_market), ('Model (Residual)', y_model)]:
        # Drop NaN
        mask = ~np.isnan(y_pred)
        metrics[name] = {
            'LogLoss': log_loss(y_true[mask], y_pred[mask]),
            'Brier': brier_score_loss(y_true[mask], y_pred[mask]),
            'AUC': roc_auc_score(y_true[mask], y_pred[mask])
        }
    
    return metrics, df

def analyze_clv(df: pd.DataFrame):
    # B. CLV Analysis (Bought Horses Only)
    col_pred = 'prob_residual_softmax' if 'prob_residual_softmax' in df.columns else 'pred_prob'
    df['ev'] = df[col_pred] * df['odds_tminus10m']
    
    # Filter bets with threshold > 1.0 (Standard)
    # Note: Using > 1.0 as a sample strategy
    bets = df[df['ev'] > 1.0].copy()
    
    if bets.empty:
        return {}, pd.DataFrame()
        
    bets['clv'] = bets['final_odds'] / bets['odds_tminus10m']
    
    stats = {
        'count': len(bets),
        'clv_mean': bets['clv'].mean(),
        'clv_median': bets['clv'].median(),
        'clv_p5': bets['clv'].quantile(0.05),
        'clv_p95': bets['clv'].quantile(0.95),
        'roi': (bets[bets['target']==1]['final_odds'].sum() / len(bets)) * 100
    }
    
    return stats, bets[['race_id', 'horse_number', 'odds_tminus10m', 'final_odds', 'clv', 'target']]

def sweep_thresholds(df: pd.DataFrame, year: str):
    """
    C. EV Threshold Sweep (Split Year)
    Split: Jan-Aug (Dev), Sep-Dec (Eval)
    """
    # Assuming 'post_time' is datetime (parquet usually preserves)
    # Ensure datetime if exists
    if 'post_time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['post_time']):
            df['post_time'] = pd.to_datetime(df['post_time'])
            
        split_date = pd.Timestamp(f"{year}-09-01")
        # dev_df = df[df['post_time'] < split_date].copy()
        eval_df = df[df['post_time'] >= split_date].copy()
    else:
        # If no date, use all data (Assume input is the target set)
        eval_df = df.copy()
        
    thresholds = np.arange(0.9, 1.35, 0.05)
    results = []
    
    bet_unit = 100
    
    for th in thresholds:
        th = round(th, 2)
        # Eval on EVAL Set
        bets = eval_df[eval_df['ev'] >= th].copy()
        
        if bets.empty:
            roi, profit, n_bets, max_dd = 0, 0, 0, 0
        else:
            n_bets = len(bets)
            total_bet = n_bets * bet_unit
            
            # Return
            # Clean target: 1 if win
            returns = bets[bets['target']==1]['final_odds'].sum() * bet_unit
            roi = (returns / total_bet) * 100
            profit = returns - total_bet
            
            # Max DD
            if 'post_time' in bets.columns:
                bets = bets.sort_values('post_time')
            else:
                bets = bets.sort_values('race_id')
            
            bets['return'] = bets.apply(lambda x: x['final_odds'] * bet_unit if x['target']==1 else 0, axis=1)
            bets['profit'] = bets['return'] - bet_unit
            bets['cum_profit'] = bets['profit'].cumsum()
            bets['running_max'] = bets['cum_profit'].cummax()
            bets['dd'] = bets['cum_profit'] - bets['running_max']
            max_dd = bets['dd'].min()
            
        results.append({
            'threshold': th,
            'bets': int(n_bets),
            'roi': roi,
            'profit': int(profit),
            'max_dd': int(max_dd)
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--year', default='2025')
    parser.add_argument('--out_report', required=True)
    args = parser.parse_args()
    
    # Load
    df = load_data(args.predictions, args.year)
    
    # 1. Performance
    metrics, _ = analyze_performance(df)
    
    # 2. CLV
    clv_stats, clv_bets = analyze_clv(df)
    
    # 3. Sweep
    # Calc EV first for full DF
    col_pred = 'prob_residual_softmax' if 'prob_residual_softmax' in df.columns else 'pred_prob'
    df['ev'] = df[col_pred] * df['odds_tminus10m']
    sweep_results = sweep_thresholds(df, args.year)
    
    # Report Generation
    report = f"# Phase 9 Diagnostics (No Leak {args.year})\n\n"
    
    report += "## A. Prediction Performance\n"
    report += "| Model | AUC | LogLoss | Brier |\n|---|---|---|---|\n"
    for name, m in metrics.items():
        report += f"| {name} | {m['AUC']:.4f} | {m['LogLoss']:.4f} | {m['Brier']:.4f} |\n"
        
    report += "\n## B. CLV Analysis (Bought Bets EV > 1.0)\n"
    report += f"- **Count**: {clv_stats.get('count',0):,}\n"
    report += f"- **ROI**: {clv_stats.get('roi',0):.2f}%\n"
    report += f"- **CLV Median**: {clv_stats.get('clv_median',0):.3f}\n"
    report += f"- **CLV Mean**: {clv_stats.get('clv_mean',0):.3f}\n"
    report += f"- **CLV Range (p5 - p95)**: {clv_stats.get('clv_p5',0):.3f} - {clv_stats.get('clv_p95',0):.3f}\n"
    if clv_stats.get('clv_mean', 0) < 1.0:
        report += "\n> [!WARNING]\n> CLV < 1.0 indicates that odds generally worsen (drop) after betting time, meaning we are betting on 'overvalued' horses that the market later corrected.\n"
        
    report += "\n## C. EV Threshold Sweep (Sep-Dec 2025)\n"
    report += sweep_results.to_markdown(index=False)
    
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    with open(args.out_report, 'w', encoding='utf-8') as f:
        f.write(report)
        
    logger.info(f"Diagnostics saved to {args.out_report}")

if __name__ == "__main__":
    main()
