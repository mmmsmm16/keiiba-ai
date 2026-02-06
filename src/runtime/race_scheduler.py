import time
import logging
import yaml
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import schedule
from sqlalchemy import create_engine, text
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Project modules
from src.runtime.odds_fetcher import OddsFetcher
from src.runtime.strategy_engine import StrategyEngine
from src.runtime.discord_notifier import DiscordNotifier
from src.runtime.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)

class RaceScheduler:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.db_url = self.config['system']['db_connection']
        self.engine = create_engine(self.db_url)
        
        self.odds_fetcher = OddsFetcher(self.db_url)
        self.strategy_engine = StrategyEngine(config_path)
        self.notifier = DiscordNotifier(os.environ.get(self.config['system']['discord_webhook_url_env']))
        
        # State Persistence
        self.state_path = self.config['system']['state_db_path'].replace('.sqlite','.json') # Use JSON for simplicity
        self.state = self._load_state()
        
        # Determine today
        self.today_str = datetime.now().strftime('%Y%m%d')
        if self.state.get('last_run_date') != self.today_str:
            # Reset daily counters if date changed
            self.state['last_run_date'] = self.today_str
            self.state['processed_races'] = []
            self.state['daily_spent'] = 0
            self._save_state()
        
        # Load Real Model
        self.model = ModelWrapper()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {'last_run_date': '', 'processed_races': [], 'daily_spent': 0}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def get_todays_races(self):
        # ... (Same as before) ...
        today_str = datetime.now().strftime('%Y%m%d') 
        yr = today_str[:4]
        md = today_str[4:]
        
        query = text("""
            SELECT 
                CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id,
                kaisai_nen, kaisai_tsukihi, hasso_jikoku,
                keibajo_code, race_bango,
                kyosomei_hondai as title
            FROM jvd_ra
            WHERE kaisai_nen = :yr 
              AND kaisai_tsukihi = :md
              AND data_kubun = '7' 
        """)
        
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(query, {"yr": yr, "md": md}).fetchall()
            
            races = []
            for r in rows:
                st_str = str(r.hasso_jikoku).zfill(4)
                start_dt = datetime.strptime(f"{yr}{md}{st_str}", "%Y%m%d%H%M")
                venue_map = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
                v_name = venue_map.get(r.keibajo_code, 'Unknown')
                
                races.append({
                    'race_id': r.race_id,
                    'start_time': start_dt,
                    'venue': v_name,
                    'race_number': int(r.race_bango),
                    'title': r.title or '',
                    'start_time_str': f"{st_str[:2]}:{st_str[2:]}"
                })
            return races
        except Exception as e:
            logger.error(f"Error fetching races: {e}")
            return []

    def process_race(self, race_id: str, race_meta: Dict):
        if race_id in self.state['processed_races']:
            return # Skip duplicate in memory check (though loop handles it too)

        logger.info(f"Processing Race {race_id}...")
        
        # 1. Fetch Odds
        odds = self.odds_fetcher.fetch_odds(race_id)
        odds_status = "OK" if odds and (odds.get('tansho') or odds.get('umaren')) else "MISSING"
        
        if odds_status == "MISSING":
            logger.warning(f"Odds missing for {race_id}. proceeding without odds.")
            
        # 2. Predict (Real Model)
        preds = self.model.predict(race_id, odds)
        if not preds:
            logger.error(f"Prediction failed for {race_id}. Skipping.")
            return

        if odds and 'tansho' in odds:
            for h in preds['horses']:
                h_num = h['horse_number']
                if h_num in odds['tansho']:
                    h['odds_win'] = odds['tansho'][h_num]
        
        # 3. Strategy
        budgets = {
            'race_cap': self.config['budgets']['race_cap_total'],
            'day_cap': self.config['budgets']['day_cap_total'],
            'current_day_spent': self.state['daily_spent']
        }
        
        bets = self.strategy_engine.decide_bets(race_id, preds, odds, budgets)
        
        # 4. Notify
        self.notifier.send_prediction(race_meta, preds['horses'], bets, odds_status)
        
        # 5. Record State & Log
        self.state['processed_races'].append(race_id)
        cost = bets.get('total_cost', 0)
        self.state['daily_spent'] += cost
        self._save_state()
        
        # Pre-race Log (JSON)
        log_dir = os.path.join(os.path.dirname(__file__), '../../reports/logs')
        os.makedirs(log_dir, exist_ok=True)
        log_data = {
            'race_meta': {'race_id': race_id, 'start_time': str(race_meta['start_time'])},
            'timestamp': str(datetime.now()),
            'preds_top5': preds['horses'][:5],
            'metrics': preds['metrics'],
            'bets': bets,
            'odds_status': odds_status
        }
        with open(os.path.join(log_dir, f"prerace_{race_id}.json"), 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Race {race_id} Done. Bets: {len(bets.get('final_bets', []))}, Cost: {cost}, DayTotal: {self.state['daily_spent']}")

    def run_loop(self):
        logger.info("Starting RaceScheduler...")
        interval = self.config['system']['cron_interval_sec']
        
        while True:
            try:
                # Refresh day if changed (Simulate cron for day reset)
                now_str = datetime.now().strftime('%Y%m%d')
                if now_str != self.state['last_run_date']:
                    self.state['last_run_date'] = now_str
                    self.state['processed_races'] = []
                    self.state['daily_spent'] = 0
                    self._save_state()

                races = self.get_todays_races()
                now = datetime.now()
                
                for race in races:
                    rid = race['race_id']
                    if rid in self.state['processed_races']: continue
                    
                    diff = race['start_time'] - now
                    diff_min = diff.total_seconds() / 60.0
                    
                    if 5 <= diff_min <= 15:
                         self.process_race(rid, race)
                    
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Scheduler Loop Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    scheduler = RaceScheduler("config/runtime/phase_j_v1.yaml")
    scheduler.run_loop()
