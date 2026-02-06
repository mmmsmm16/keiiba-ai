# ... (Imports remain mostly the same, ensuring all needed are present)
import os
import sys
import json
import time
import requests
import argparse
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Optional
from scipy.special import softmax

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.preprocessor import InferencePreprocessor
from src.inference.loader import InferenceDataLoader
from src.utils.discord import NotificationManager
from src.betting.strategies import UnifiedStrategy, BettingConfig
from src.betting.odds_db import JraDbOddsProvider

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../../logs/auto_predict.log'))
    ]
)
logger = logging.getLogger(__name__)

# .env 手動読み込み
def load_env_manual():
    try:
        env_path = os.path.join(os.path.dirname(__file__), '../../.env')
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()
    except Exception as e:
        logger.warning(f".env reading failed: {e}")

# 定数
STATE_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../data/state/notified_races.json')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../experiments/v23_regression_cv/fold4')

class AutoPredictor:
    def __init__(self, dry_run: bool = False, target_date: str = None):
        self.dry_run = dry_run
        self.target_date = target_date
        self.state_file = STATE_FILE_PATH
        self.notified_races = self._load_state()
        
        # Models
        self.loader = InferenceDataLoader()
        self.preprocessor = InferencePreprocessor()
        self.lgbm, self.catboost, self.meta = self._load_v23_models()
        
        # Strategy
        self.odds_provider = JraDbOddsProvider()
        self.config = BettingConfig(
            bet_types=['win', 'place', 'umaren', 'wide', 'sanrenpuku', 'umatan'],
            ev_threshold=0.2,
            edge_threshold=1.2,
            budget_per_race=10000,
            top_n_horses=10
        )
        self.strategy = UnifiedStrategy(self.config, self.odds_provider)
        
        # Notification
        load_env_manual()
        webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            logger.error("❌ DISCORD_WEBHOOK_URL is not set.")
        self.notifier = NotificationManager(webhook_url)
        
    def _load_state(self) -> set:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f: return set(json.load(f))
            except: return set()
        return set()

    def _save_state(self):
        if self.dry_run: return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f: json.dump(list(self.notified_races), f)

    def _load_v23_models(self):
        logger.info(f"Loading v23 models from {MODEL_DIR}...")
        try:
            with open(os.path.join(MODEL_DIR, 'lgbm_v23.pkl'), 'rb') as f: lgbm = pickle.load(f)
            with open(os.path.join(MODEL_DIR, 'catboost_v23.pkl'), 'rb') as f: catboost = pickle.load(f)
            with open(os.path.join(MODEL_DIR, 'meta_v23.pkl'), 'rb') as f: meta = pickle.load(f)
            return lgbm, catboost, meta
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            sys.exit(1)

    def run(self):
        logger.info("AutoPredict Unified Process Start...")
        
        now = datetime.now()
        today_str = self.target_date.replace('-', '') if self.target_date else now.strftime('%Y%m%d')

        # 1. Race List
        try:
            race_list_df = self.loader.load_race_list(today_str)
        except Exception as e:
            logger.error(f"Race list load failed: {e}")
            return
            
        venue_map = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
            '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
        }
        race_list_df['venue_name'] = race_list_df['venue'].map(venue_map).fillna(race_list_df['venue'])

        if race_list_df.empty:
            logger.info("No races today.")
            return

        # 2. Filter (15-35 mins before)
        targets = []
        for _, row in race_list_df.iterrows():
            race_id = row['race_id']
            if race_id in self.notified_races: continue
                
            start_time_str = str(row['start_time']).replace(':', '')
            try:
                race_dt = datetime.strptime(f"{today_str}{start_time_str}", "%Y%m%d%H%M")
            except: continue

            if self.target_date:
                targets.append(row)
            else:
                diff = race_dt - now
                minutes = diff.total_seconds() / 60
                if 15 <= minutes <= 35:
                     targets.append(row)
        
        if not targets:
            logger.info("No target races for notification.")
            return

        logger.info(f"Targets: {len(targets)} races")

        # 3. Load Data & Predict
        target_ids = [r['race_id'] for r in targets]
        raw_df = self.loader.load(target_date=today_str, race_ids=target_ids)
        if raw_df.empty: return

        X, ids = self.preprocessor.preprocess(raw_df)
        
        # Feature Alignment
        expected_cols = self.lgbm.feature_name()
        for col in expected_cols:
            if col not in X.columns: X[col] = 0.0
        X = X[expected_cols]

        try:
            p1 = self.lgbm.predict(X)
            p2 = self.catboost.predict(X)
            scores = self.meta.predict(np.column_stack([p1, p2]))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return

        result_df = ids.copy()
        result_df['score'] = scores
        result_df['prob'] = result_df.groupby('race_id')['score'].transform(lambda x: softmax(x))

        # 4. Generate Tickets & Notify
        for race_meta in targets:
            race_id = race_meta['race_id']
            race_df = result_df[result_df['race_id'] == race_id].copy()
            if race_df.empty: continue
            
            # Prepare Model Probs for Strategy
            probs = dict(zip(race_df['horse_number'], race_df['prob']))
            
            # Generate Tickets
            # asof=None means use Latest Odds (Realtime)
            tickets = self.strategy.generate_tickets(race_id, probs, asof=None)
            
            # Save Report
            self._save_report(race_meta, race_df, tickets, today_str)

            # Notification
            if not self.dry_run:
                success = self.notifier.send_tickets(race_meta, race_df, tickets)
                if success: self.notified_races.add(race_id)
                time.sleep(1.0)
            else:
                logger.info(f"[DRY-RUN] {race_meta.get('title')}")
                print(race_df.sort_values('prob', ascending=False).head())
                print(f"Tickets Generated: {len(tickets)}")
                for t in tickets:
                    print(t)

        self._save_state()

    def _save_report(self, meta, df, tickets, date_str):
        """Save prediction and betting report to Markdown."""
        report_dir = os.path.join(os.path.dirname(__file__), f'../../reports/jra/daily/{date_str}')
        os.makedirs(report_dir, exist_ok=True)
        
        rid = meta['race_id']
        path = os.path.join(report_dir, f"{rid}.md")
        
        lines = []
        lines.append(f"# {meta.get('venue_name','')}{meta.get('race_number','')}R {meta.get('title','')}")
        lines.append(f"- Date: {meta.get('date', date_str)}")
        lines.append(f"- Start: {meta.get('start_time','')}")
        lines.append("")
        
        lines.append("## Predictions (Top 5)")
        lines.append("| Horse | Name | Prob | Score | Odds |")
        lines.append("|---|---|---|---|---|")
        
        top = df.sort_values('prob', ascending=False).head(5)
        for _, row in top.iterrows():
            h = str(int(row['horse_number'])).zfill(2)
            n = row.get('horse_name', '-')
            p = f"{row.get('prob', 0)*100:.1f}%"
            s = f"{row.get('score', 0):.2f}"
            o = f"{row.get('odds', 0):.1f}"
            lines.append(f"| {h} | {n} | {p} | {s} | {o} |")
        lines.append("")
        
        lines.append("## Tickets")
        if not tickets:
            lines.append("No tickets generated.")
        else:
            lines.append("| Type | Selection | Odds | Stake | EV/Score | Type |")
            lines.append("|---|---|---|---|---|---|")
            for t in tickets:
                btype = t.bet_type
                sel = t.combination_str
                odds = f"{t.odds:.1f}" if t.odds else "N/A"
                stake = t.stake
                val = f"{t.expected_value:.2f}" if t.odds_type == 'real' else f"{t.selection_score:.2f}"
                otype = t.odds_type
                lines.append(f"| {btype} | {sel} | {odds} | {stake} | {val} | {otype} |")
                
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description='Automated Prediction & Notification (Unified)')
    parser.add_argument('--dry-run', action='store_true', help='No Discord send')
    parser.add_argument('--date', type=str, help='Target date YYYY-MM-DD')
    args = parser.parse_args()
    
    predictor = AutoPredictor(dry_run=args.dry_run, target_date=args.date)
    predictor.run()

if __name__ == "__main__":
    main()
```
