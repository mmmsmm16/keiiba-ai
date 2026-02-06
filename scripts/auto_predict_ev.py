"""
Operational Strategy Script (EV Model)
Rankerモデルとリアルタイムオッズ (単勝・馬連) を組み合わせて期待値 (EV) を算出し、
投資価値のある買い目を推奨・通知する。
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# プロジェクトルートを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.feature_engineering import FeatureEngineer
from src.data.realtime_loader import RealTimeDataLoader
from src.preprocessing.aggregators import HistoryAggregator
from src.preprocessing.advanced_features import AdvancedFeatureEngineer
from src.preprocessing.incremental_aggregators import IncrementalCategoryAggregator
# Import other engineers as needed per auto_predict_no_odds.py
from src.preprocessing.experience_features import ExperienceFeatureEngineer
from src.preprocessing.relative_features import RelativeFeatureEngineer
from src.preprocessing.opposition_features import OppositionFeatureEngineer
from src.preprocessing.rating_features import RatingFeatureEngineer
from src.utils.discord import NotificationManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 定数
RANKER_MODEL_PATH = "models/eval/ranker_eval_v19.pkl"
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
REPORT_DIR = "reports/jra/daily"
VENUE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

    def __init__(self, model_path: str, threshold=1.0, min_prob=0.0, min_odds=0.0, kelly_fraction=0.1, max_bet=5000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading Ranker model from {model_path}...")
        payload = joblib.load(model_path)
        self.model = payload['model']
        self.feature_cols = payload['feature_cols']
        
        self.threshold = threshold
        self.min_prob = min_prob
        self.min_odds = min_odds
        self.kelly_fraction = kelly_fraction
        self.max_bet = max_bet
        
        # Initialize components
        self.loader = JraVanDataLoader()
        self.rt_loader = RealTimeDataLoader()
        self.cleanser = DataCleanser()
        self.engineer = FeatureEngineer()
        self.hist_agg = HistoryAggregator()
        self.adv_eng = AdvancedFeatureEngineer()
        self.exp_eng = ExperienceFeatureEngineer()
        self.rel_eng = RelativeFeatureEngineer()
        self.opp_eng = OppositionFeatureEngineer()
        self.rating_eng = RatingFeatureEngineer()

    def _kelly_bet(self, prob: float, odds: float) -> int:
        """
        Calculate Kelly Bet Amount.
        """
        if prob <= 0 or odds <= 1.0:
            return 0
            
        b = odds - 1.0
        q = 1.0 - prob
        f = (b * prob - q) / b # Full Kelly Fraction
        
        # Apply fractional kelly
        f = f * self.kelly_fraction
        
        if f <= 0:
            return 0
            
        # Suggested Bet
        bet_amount = 100000 * f # Assuming fixed bankroll 100k for Paper Trading per race context? 
        # Actually paper trading should track bankroll, but for now we assume 100k base for calculation if we don't have state.
        # Or just use the passed max_bet as a cap and base calculation on 100k?
        # Let's align with simulation: Initial Bankroll 100,000.
        
        bet_amount = min(bet_amount, self.max_bet)
        bet_amount = int(bet_amount // 100) * 100
        return bet_amount

    def _calc_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranker Score -> Softmax Probabilities
        """
        def softmax_group(group):
            s = group['score'].values
            # Temperature scaling could be tuned; 1.5 used in prev script
            if len(s) > 1 and s.std() > 0:
                z = (s - s.mean()) / s.std()
            else:
                z = s - s.mean()
            
            # Temperature scaling (Removed: T=1.5 -> T=1.0 based on calibration check)
            # exp_s = np.exp(z * 1.5)
            exp_s = np.exp(z)
            group['win_prob'] = exp_s / exp_s.sum()
            return group

        return df.groupby('race_id', group_keys=False).apply(softmax_group)

    def _calc_umaren_probs(self, race_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Umaren (Quinella) probabilities using Harville's formula approximation.
        P(i, j) = P(i) * P(j | not i) + P(j) * P(i | not j)
                = P(i)*P(j)/(1-P(i)) + P(j)*P(i)/(1-P(j))
        """
        probs = {}
        runners = race_df[['horse_number', 'win_prob']].values # [[num, prob], ...]
        
        for i in range(len(runners)):
            h1_num, p1 = runners[i]
            for j in range(i + 1, len(runners)):
                h2_num, p2 = runners[j]
                
                # Avoid division by zero
                denom1 = 1.0 - p1
                denom2 = 1.0 - p2
                
                if denom1 <= 0 or denom2 <= 0:
                    pair_prob = 0.0
                else:
                    pair_prob = (p1 * p2 / denom1) + (p2 * p1 / denom2)
                
                # Key format: "01-02" (sorted)
                k1 = int(h1_num)
                k2 = int(h2_num)
                key = f"{min(k1,k2):02d}-{max(k1,k2):02d}"
                probs[key] = pair_prob
                
        return probs

    def predict_ev(self, target_date: str, paper_trade: bool = False):
        logger.info(f"=== EV Prediction for {target_date} (PaperTrade={paper_trade}) ===")
        
        # 1. Load Data
        df = self.loader.load(history_start_date=target_date, end_date=target_date, jra_only=True)
        if len(df) == 0:
            logger.warning("No data found.")
            return None, []

        # 2. Preprocess (Same as Ranker)
        df = self.cleanser.cleanse(df)
        df = self.engineer.add_features(df)
        
        logger.info("Loading cache for incremental aggregation...")
        try:
            master_df = pd.read_parquet(CACHE_PATH)
            master_df['date'] = pd.to_datetime(master_df['date'])
            inc_cat_agg = IncrementalCategoryAggregator()
            inc_cat_agg.fit(master_df[master_df['date'] < target_date])
            df = inc_cat_agg.transform_update(df)
            
            # Context specific features
            ctx_date = (pd.to_datetime(target_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            context_df = master_df[(master_df['date'] >= ctx_date) & (master_df['date'] < target_date)].copy()
            proc_df = pd.concat([context_df, df], ignore_index=True).sort_values(['date', 'race_id'])
            
            # Feature generations
            proc_df = self.hist_agg.aggregate(proc_df)
            proc_df = self.adv_eng.add_features(proc_df)
            proc_df = self.exp_eng.add_features(proc_df)
            proc_df = self.rel_eng.add_features(proc_df)
            proc_df = self.opp_eng.add_features(proc_df)
            proc_df = self.rating_eng.add_features(proc_df)
            
            test_df = proc_df[proc_df['date'] == target_date].copy()
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None, []

        if test_df.empty: return None, []

        # 3. Predict Score & Win Prob
        X = test_df[self.feature_cols].copy()
        # Handle duplicates
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        
        test_df['score'] = self.model.predict(X)
        test_df = self._calc_probs(test_df)
        
        # 4. Fetch Real-time Odds
        race_ids = test_df['race_id'].unique().tolist()
        logger.info(f"Fetching odds for {len(race_ids)} races...")
        
        win_odds_map = self.rt_loader.get_latest_odds(race_ids, 'win')
        uma_odds_map = self.rt_loader.get_latest_odds(race_ids, 'umaren')
        
        # 5. Calculate EV
        reports = []
        discord_buffer = []
        paper_trades = []

        for race_id, race_df in test_df.groupby('race_id'):
            race_rec = []
            
            # --- Win EV ---
            r_win_odds = win_odds_map.get(race_id, {})
            # Map odds to dataframe
            race_df['odds'] = race_df['horse_number'].apply(lambda x: r_win_odds.get(str(int(x)), 0.0))
            race_df['ev'] = race_df['win_prob'] * race_df['odds']
            
            # Recommendations
            recs = []
            for _, row in race_df.iterrows():
                # Filter by Cutoffs
                prob = row['win_prob']
                odds = row['odds']
                if prob >= self.min_prob and odds >= self.min_odds:
                    if row['ev'] >= self.threshold:
                        # Calc bet amount
                        bet = self._kelly_bet(prob, odds)
                        row['bet_amount'] = bet
                        if bet > 0:
                            recs.append(row)
            
            recs = pd.DataFrame(recs)
            if not recs.empty:
                recs = recs.sort_values('ev', ascending=False)
                
                if paper_trade:
                    for _, row in recs.iterrows():
                        paper_trades.append({
                            'date': target_date,
                            'race_id': race_id,
                            'type': 'WIN',
                            'target': int(row['horse_number']),
                            'prob': row['win_prob'],
                            'odds': row['odds'],
                            'ev': row['ev'],
                            'bet_amount': row['bet_amount'],
                            'fraction': self.kelly_fraction,
                            'status': 'PENDING'
                        })
            
            # --- Umaren EV ---
            r_uma_probs = self._calc_umaren_probs(race_df)
            r_uma_odds = uma_odds_map.get(race_id, {})
            
            uma_recs = []
            for combo, prob in r_uma_probs.items():
                odds = r_uma_odds.get(combo, 0.0)
                ev = prob * odds
                
                # Filter by Cutoffs
                if prob >= self.min_prob and odds >= self.min_odds:
                    if ev >= self.threshold:
                        bet = self._kelly_bet(prob, odds)
                        if bet > 0:
                            uma_recs.append({
                                'combo': combo,
                                'prob': prob,
                                'odds': odds,
                                'ev': ev,
                                'bet_amount': bet
                            })
                            
                            if paper_trade:
                                paper_trades.append({
                                    'date': target_date,
                                    'race_id': race_id,
                                    'type': 'UMA',
                                    'target': combo,
                                    'prob': prob,
                                    'odds': odds,
                                    'ev': ev,
                                    'bet_amount': bet,
                                    'fraction': self.kelly_fraction,
                                    'status': 'PENDING'
                                })

            uma_recs.sort(key=lambda x: x['ev'], reverse=True)
            
            # --- Format Report ---
            v_code = race_df['venue'].iloc[0]
            v_name = VENUE_MAP.get(v_code, v_code)
            r_num = int(race_id[-2:])
            r_name = race_df['title'].iloc[0] if 'title' in race_df.columns else ""
            
            race_header = f"### {v_name} {r_num}R {r_name}"
            race_rec.append(race_header)
            
            # Win Table
            race_rec.append("**単勝 (Win) 推奨**")
            if not recs.empty:
                race_rec.append("| 馬番 | 馬名 | 勝率% | オッズ | EV | Bet |")
                race_rec.append("| :--- | :--- | :--- | :--- | :--- | :---|")
                for _, row in recs.iterrows():
                    h_name = row['horse_name']
                    # Mark if high EV
                    mark = "🔥" if row['ev'] > 1.2 else ""
                    race_rec.append(f"| {int(row['horse_number']):02d} | {h_name} | {row['win_prob']:.1%} | {row['odds']:.1f} | **{row['ev']:.2f}** {mark}| ¥{row['bet_amount']} |")
            else:
                race_rec.append("推奨なし (No bets > Threshold)")
                
            # Umaren Table
            race_rec.append("\n**馬連 (Umaren) 推奨**")
            if uma_recs:
                race_rec.append("| 組番 | 確率% | オッズ | EV | Bet |")
                race_rec.append("| :--- | :--- | :--- | :--- | :--- |")
                for r in uma_recs[:5]: # Top 5 only to save space
                    mark = "🔥" if r['ev'] > 1.5 else ""
                    race_rec.append(f"| {r['combo']} | {r['prob']:.1%} | {r['odds']:.1f} | **{r['ev']:.2f}** {mark}| ¥{r['bet_amount']} |")
            else:
                race_rec.append("推奨なし")
                
            reports.append("\n".join(race_rec))
            
            # Discord Summary (Concise)
            # Only if high EV exists
            top_win = recs.iloc[0] if not recs.empty else None
            top_uma = uma_recs[0] if uma_recs else None
            
            has_rec = False
            msg = f"**{v_name}{r_num}R**"
            
            if top_win is not None:
                msg += f"\n🏆 単: {int(top_win['horse_number']):02d} {top_win['horse_name']} (EV {top_win['ev']:.2f}) [¥{top_win['bet_amount']}]"
                has_rec = True
            if top_uma:
                msg += f"\n🔗 連: {top_uma['combo']} (EV {top_uma['ev']:.2f}) [¥{top_uma['bet_amount']}]"
                has_rec = True
                
            if has_rec:
                discord_buffer.append(msg)
                
        # Final Output
        full_report = f"# JRA EV運用レポート ({target_date})\n" + "\n\n".join(reports)
        
        # Save Paper Trades including failures/skips logic? 
        # Here we only record placed bets
        if paper_trade and paper_trades:
            log_dir = "data/paper_trades"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "trade_log.csv")
            
            pt_df = pd.DataFrame(paper_trades)
            # Check if file exists to append
            if os.path.exists(log_path):
                pt_df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                pt_df.to_csv(log_path, index=False)
            logger.info(f"Recorded {len(paper_trades)} paper trades to {log_path}")

        return full_report, discord_buffer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--paper-trade", action="store_true", help="Enable Paper Trading mode (Log bets to CSV)")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--fraction", type=float, default=0.1, help="Kelly fraction")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Minimum probability cutoff")
    parser.add_argument("--min_odds", type=float, default=0.0, help="Minimum odds cutoff")
    args = parser.parse_args()

    try:
        predictor = EVPreditcor(
            RANKER_MODEL_PATH, 
            threshold=args.threshold, 
            min_prob=args.min_prob, 
            min_odds=args.min_odds,
            kelly_fraction=args.fraction
        )
        report, summary_list = predictor.predict_ev(args.date, paper_trade=args.paper_trade)
        
        if report:
            os.makedirs(REPORT_DIR, exist_ok=True)
            report_path = os.path.join(REPORT_DIR, f"{args.date}_ev_strategy.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
            
            if args.notify and summary_list:
                webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
                if webhook_url:
                    nm = NotificationManager(webhook_url)
                    header = f"【JRA EV投資推奨】{args.date}"
                    if args.paper_trade:
                        header += " (PaperTrade)"
                    nm.send_text(header + "\n" + "\n".join(summary_list))
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
