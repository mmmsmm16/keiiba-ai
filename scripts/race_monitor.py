"""
レース直前通知システム
各レースの発走15分前に予測→通知を実行

Usage:
    # 当日モニタリング開始
    docker compose exec app python scripts/race_monitor.py

    # 特定日付でテスト
    docker compose exec app python scripts/race_monitor.py --date 2025-12-07 --dry-run

    # 通知タイミング調整 (デフォルト15分前)
    docker compose exec app python scripts/race_monitor.py --minutes-before 20
"""

import sys
import os
import argparse
import logging
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.loader import InferenceDataLoader
from inference.preprocessor import InferencePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


def send_discord_message(webhook_url: str, message: str) -> bool:
    """Send message to Discord via webhook"""
    if not webhook_url:
        return False
    try:
        response = requests.post(webhook_url, json={"content": message}, timeout=10)
        return response.status_code in [200, 204]
    except:
        return False


def load_v13_models(model_dir: str) -> List[lgb.Booster]:
    """Load v13 models"""
    models = []
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
    for f in model_files:
        models.append(lgb.Booster(model_file=os.path.join(model_dir, f)))
    return models


def run_single_race_prediction(race_id: str, loader: InferenceDataLoader, 
                                preprocessor: InferencePreprocessor,
                                models: List[lgb.Booster]) -> Optional[Dict]:
    """Run prediction for a single race"""
    from scipy.special import expit
    
    try:
        # Load data for this specific race
        raw_df = loader.load(race_ids=[race_id])
        if raw_df.empty:
            return None
        
        # Preprocess
        X, ids, full_df = preprocessor.preprocess(raw_df, return_full_df=True)
        
        # Get feature columns and run inference
        feature_cols = models[0].feature_name()
        for c in feature_cols:
            if c not in full_df.columns:
                full_df[c] = 0
        
        X_model = full_df[feature_cols].fillna(0)
        
        # Ensemble prediction
        preds = np.mean([model.predict(X_model) for model in models], axis=0)
        full_df['prob'] = expit(preds)
        
        # Softmax
        exp_vals = np.exp(full_df['prob'] - full_df['prob'].max())
        full_df['prob'] = exp_vals / exp_vals.sum()
        
        # Merge back horse info
        if 'horse_number' not in full_df.columns:
            full_df['horse_number'] = ids['horse_number']
        if 'race_id' not in full_df.columns:
            full_df['race_id'] = ids['race_id']
        for col in ['horse_name']:
            if col in raw_df.columns and col not in full_df.columns:
                full_df = full_df.merge(
                    raw_df[['race_id', 'horse_number', col]].drop_duplicates(),
                    on=['race_id', 'horse_number'], how='left'
                )
        
        # Sort by probability
        full_df = full_df.sort_values('prob', ascending=False)
        
        # All horses list (sorted by prob)
        all_horses = []
        for _, row in full_df.iterrows():
            h_num = int(row['horse_number'])
            h_name = str(row.get('horse_name', f'馬{h_num}')).strip() if 'horse_name' in row.index else f'馬{h_num}'
            all_horses.append({
                'number': h_num,
                'name': h_name,
                'prob': float(row['prob'])
            })
        
        # Top 4 for tickets
        top4 = full_df.head(4)
        horse_numbers = top4['horse_number'].astype(int).tolist()
        horse_names = [str(n).strip()[:8] for n in top4['horse_name'].fillna('').tolist()] if 'horse_name' in top4.columns else [f'馬{h}' for h in horse_numbers]
        
        tickets = []
        for comb in combinations(range(4), 3):
            nums = [horse_numbers[i] for i in comb]
            tickets.append('-'.join(map(str, nums)))
        
        return {
            'race_id': race_id,
            'top_horses': horse_numbers,
            'top_names': horse_names,
            'tickets': tickets,
            'top4_probs': top4['prob'].tolist(),
            'all_horses': all_horses
        }
        
    except Exception as e:
        logger.error(f"Prediction error for {race_id}: {e}")
        return None


def format_race_notification(race_info: Dict, prediction: Dict) -> str:
    """Format notification for a single race with all horse scores"""
    venue = race_info.get('venue_name', '?')
    race_num = race_info.get('race_number', '?')
    title = race_info.get('title', '')[:20]
    start_time = race_info.get('start_time', '')
    
    msg = f"🏇 **{venue} {race_num}R** {start_time}\n"
    if title:
        msg += f"_{title}_\n"
    msg += "\n"
    
    # All horses with scores
    msg += "📊 **予測スコア** (勝率)\n"
    msg += "```\n"
    
    all_horses = prediction.get('all_horses', [])
    for i, h in enumerate(all_horses):
        rank_mark = "⭐" if i < 4 else "  "
        prob_pct = h['prob'] * 100
        msg += f"{rank_mark} {h['number']:2d}番 {h['name'][:8]:<8} {prob_pct:5.1f}%\n"
    
    msg += "```\n"
    
    # Top 4 and tickets
    msg += f"\n🎯 **推奨 TOP4**: "
    horses = ', '.join([f"{n}番" for n in prediction['top_horses']])
    msg += horses + "\n"
    
    # Tickets
    msg += f"📝 **買目**: {', '.join(prediction['tickets'])} (各¥100)\n"
    msg += f"💰 合計: 4点 ¥400\n"
    msg += "⚠️ 締切に注意！"
    
    return msg


def get_race_schedule(loader: InferenceDataLoader, target_date: str) -> List[Dict]:
    """Get race schedule with start times"""
    
    race_list = loader.load_race_list(target_date)
    
    if race_list.empty:
        return []
    
    course_map = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
        '05': '東京', '06': '中山', '07': '中京', '08': '京都',
        '09': '阪神', '10': '小倉'
    }
    
    races = []
    for _, row in race_list.iterrows():
        race_id = str(row.get('race_id', ''))
        venue_code = str(row.get('venue', '')).zfill(2)
        
        races.append({
            'race_id': race_id,
            'venue': venue_code,
            'venue_name': course_map.get(venue_code, venue_code),
            'race_number': row.get('race_number', 0),
            'start_time': row.get('start_time', ''),
            'title': row.get('title', ''),
            'notified': False
        })
    
    return sorted(races, key=lambda x: (x['start_time'], x['venue']))


def parse_start_time(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse start time string to datetime"""
    try:
        if not time_str or time_str == 'None':
            return None
        # Try HH:MM format
        time_str = str(time_str).strip()
        if ':' in time_str:
            hour, minute = map(int, time_str.split(':')[:2])
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.replace(hour=hour, minute=minute)
    except:
        pass
    return None


def run_monitor(date_str: str, minutes_before: int = 15, dry_run: bool = False):
    """Main monitoring loop"""
    
    load_env()
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '')
    
    logger.info(f"=== Race Monitor: {date_str} ===")
    logger.info(f"Notification timing: {minutes_before} minutes before race")
    
    # Initialize
    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    models = load_v13_models('models/v13_market_residual')
    
    target_date = date_str.replace('-', '')
    races = get_race_schedule(loader, target_date)
    
    if not races:
        logger.warning("No races found for today")
        return
    
    logger.info(f"Found {len(races)} races")
    
    # State tracking
    notified_races = set()
    state_file = Path(f'data/paper_trade/{date_str}/monitor_state.json')
    state_file.parent.mkdir(parents=True, exist_ok=True)
    
    if state_file.exists():
        with open(state_file, 'r') as f:
            notified_races = set(json.load(f).get('notified', []))
    
    if dry_run:
        # Dry run: process all races immediately
        logger.info("DRY RUN: Processing all races immediately")
        for race in races:
            if race['race_id'] in notified_races:
                continue
            
            logger.info(f"Processing {race['venue_name']} {race['race_number']}R...")
            prediction = run_single_race_prediction(
                race['race_id'], loader, preprocessor, models
            )
            
            if prediction:
                msg = format_race_notification(race, prediction)
                print("\n" + "=" * 40)
                print(msg)
                print("=" * 40)
                notified_races.add(race['race_id'])
        
        logger.info(f"Processed {len(notified_races)} races")
        return
    
    # Real monitoring loop
    logger.info("Starting monitoring loop... (Ctrl+C to stop)")
    
    if webhook_url:
        send_discord_message(webhook_url, f"🏇 レースモニター開始: {date_str}\n{len(races)}レースを監視中")
    
    try:
        while True:
            now = datetime.now()
            
            for race in races:
                if race['race_id'] in notified_races:
                    continue
                
                start_time = parse_start_time(date_str, race['start_time'])
                if not start_time:
                    continue
                
                # Calculate time until race
                time_until = (start_time - now).total_seconds() / 60
                
                # Trigger notification X minutes before
                if 0 < time_until <= minutes_before:
                    logger.info(f"⏰ Triggering: {race['venue_name']} {race['race_number']}R (発走まで {time_until:.0f}分)")
                    
                    prediction = run_single_race_prediction(
                        race['race_id'], loader, preprocessor, models
                    )
                    
                    if prediction:
                        msg = format_race_notification(race, prediction)
                        
                        if webhook_url:
                            send_discord_message(webhook_url, msg)
                        else:
                            print(msg)
                        
                        notified_races.add(race['race_id'])
                        
                        # Save state
                        with open(state_file, 'w') as f:
                            json.dump({'notified': list(notified_races)}, f)
            
            # Check if all races are done
            all_done = len(notified_races) >= len(races)
            if all_done:
                logger.info("All races processed. Exiting.")
                break
            
            # Wait before next check
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    
    # Final summary
    if webhook_url:
        send_discord_message(webhook_url, f"📊 本日の監視終了\n処理レース: {len(notified_races)}/{len(races)}")


def main():
    parser = argparse.ArgumentParser(description="Race Monitor - Per-race notifications")
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD), default=today')
    parser.add_argument('--minutes-before', type=int, default=15,
                        help='Minutes before race to send notification')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process all races immediately without waiting')
    
    args = parser.parse_args()
    
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    run_monitor(date_str, args.minutes_before, args.dry_run)


if __name__ == "__main__":
    main()
