import pandas as pd
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_indicators(input_path: str, output_path: str):
    logger.info(f"Loading predictions from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 1. p1, p2 (Top1/Top2 based on p_cal)
    # 既に rank_pred があるのでそれを利用
    logger.info("Computing p1, p2 and margin...")
    
    # race_idごとのTop1, Top2確率を取得
    top_probs = df[df['rank_pred'].isin([1, 2])].pivot(index='race_id', columns='rank_pred', values='p_cal')
    top_probs.columns = [f'p{int(col)}' for col in top_probs.columns]
    
    # 欠損対応 (1頭しかいないレースなど)
    if 'p1' not in top_probs.columns: top_probs['p1'] = np.nan
    if 'p2' not in top_probs.columns: top_probs['p2'] = np.nan
    
    top_probs['margin'] = top_probs['p1'] - top_probs['p2']
    
    # 2. sum_top5
    logger.info("Computing sum_top5...")
    sum_top5 = df[df['rank_pred'] <= 5].groupby('race_id')['p_cal'].sum().rename('sum_top5')
    
    # 3. field_size
    logger.info("Computing field_size...")
    field_size = df.groupby('race_id').size().rename('field_size')
    
    # 4. entropy
    logger.info("Computing entropy...")
    def compute_entropy(p):
        # p_calが負や0の場合を考慮
        p = p[p > 0]
        if len(p) == 0: return 0
        return -np.sum(p * np.log(p))
    
    entropy = df.groupby('race_id')['p_cal'].apply(compute_entropy).rename('entropy')
    
    # 5. Ad-hoc Win Calibration (p_win)
    # Model target is Top3, so p_cal is Top3 Prob. We need Win Prob for EV/Umaren.
    # We calibrate p_cal (Top3 Prob) -> Win Prob using 2022-2023 data.
    logger.info("Calibrating p_win (from p_cal)...")
    train_mask = df['year_valid'].isin([2022, 2023])
    if train_mask.sum() > 0:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        # Target: 1 if rank==1 else 0
        y_win = (df.loc[train_mask, 'finish_pos'] == 1).astype(int)
        
        # Fit on p_cal (Top3 Prob)
        iso.fit(df.loc[train_mask, 'p_cal'], y_win)
        
        # Predict for all
        df['p_win'] = iso.predict(df['p_cal'])
        logger.info(f"p_win calibrated. Mean p_win: {df['p_win'].mean():.4f}")
    else:
        logger.warning("No training data for p_win calibration. setting p_win = p_cal / 3 (heuristic)")
        df['p_win'] = df['p_cal'] / 3.0

    # Rename p_cal to p_place for clarity if needed, but allow p_cal to remain
    df['p_place'] = df['p_cal']
    
    # 指標をマージ
    indicators = pd.concat([top_probs, sum_top5, field_size, entropy], axis=1).reset_index()
    
    df = pd.merge(df, indicators, on='race_id', how='left')
    
    logger.info(f"Saving enriched dataset to {output_path}...")
    df.to_parquet(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/simulations/v13_e1_predictions_2022_2024.parquet")
    parser.add_argument("--output", default="reports/simulations/v13_e1_enriched_2022_2024.parquet")
    args = parser.parse_args()
    
    add_indicators(args.input, args.output)
