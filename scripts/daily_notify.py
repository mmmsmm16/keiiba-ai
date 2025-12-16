"""
Daily Notification: V13予測→Discord通知
Phase 11: 自動予測＆通知システム

Usage:
    docker compose exec app python scripts/daily_notify.py --date 2025-12-07
    docker compose exec app python scripts/daily_notify.py  # 今日の日付

通知のみ（自動購入なし）
"""

import sys
import os
import argparse
import logging
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# .env 読み込み
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
        logger.warning("Discord webhook URL not set")
        return False
    
    try:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code in [200, 204]:
            logger.info("Discord message sent successfully")
            return True
        else:
            logger.error(f"Discord send failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Discord send error: {e}")
        return False


def load_models(model_dir: str) -> List[lgb.Booster]:
    """Load v13 fold models"""
    models = []
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
    for f in model_files:
        path = os.path.join(model_dir, f)
        models.append(lgb.Booster(model_file=path))
    return models


def run_inference(df: pd.DataFrame, models: List[lgb.Booster]) -> pd.DataFrame:
    """Run v13 inference"""
    from scipy.special import expit
    
    # Get feature columns from first model
    feature_cols = models[0].feature_name()
    
    # Prepare features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    # Ensemble prediction
    preds = []
    for model in models:
        pred = model.predict(X)
        preds.append(pred)
    
    avg_pred = np.mean(preds, axis=0)
    df['prob_raw'] = expit(avg_pred)
    
    # Softmax per race
    def softmax_race(group):
        exp_vals = np.exp(group - group.max())
        return exp_vals / exp_vals.sum()
    
    df['prob'] = df.groupby('race_id')['prob_raw'].transform(softmax_race)
    
    return df


def generate_tickets(race_df: pd.DataFrame, topn: int = 4) -> List[Dict]:
    """Generate sanrenpuku BOX tickets"""
    from itertools import combinations
    
    top_horses = race_df.nlargest(topn, 'prob')
    horse_numbers = top_horses['horse_number'].astype(int).tolist()
    horse_names = top_horses['horse_name'].tolist() if 'horse_name' in top_horses.columns else [f'馬{h}' for h in horse_numbers]
    
    tickets = []
    for comb in combinations(range(len(horse_numbers)), 3):
        nums = [horse_numbers[i] for i in comb]
        names = [horse_names[i] for i in comb]
        tickets.append({
            'numbers': nums,
            'names': names,
            'display': '-'.join(map(str, nums)),
            'name_display': '-'.join(names)
        })
    
    return tickets, horse_numbers, horse_names


def format_notification(date_str: str, races_data: List[Dict]) -> str:
    """Format betting notification message"""
    
    msg = f"🏇 **{date_str} 買い目速報**\n"
    msg += f"戦略: sanrenpuku BOX4 (v13モデル)\n"
    msg += "=" * 30 + "\n\n"
    
    total_tickets = 0
    total_stake = 0
    
    for race in races_data:
        venue = race['venue']
        race_num = race['race_num']
        race_name = race.get('race_name', '')
        
        msg += f"**{venue} {race_num}R** {race_name}\n"
        msg += f"推奨馬: {', '.join([f'{n}番{h}' for n, h in zip(race['top_horses'], race['top_names'])])}\n"
        msg += "買い目:\n"
        
        for t in race['tickets']:
            msg += f"  ・三連複 {t['display']} ({t['name_display']}) ¥100\n"
            total_tickets += 1
            total_stake += 100
        
        msg += "\n"
    
    msg += "=" * 30 + "\n"
    msg += f"合計: {len(races_data)}レース, {total_tickets}点, ¥{total_stake:,}\n"
    msg += "\n⚠️ 購入は自己責任でお願いします"
    
    return msg


def run_daily_notification(date_str: str, dry_run: bool = False) -> Dict[str, Any]:
    """Run daily prediction and send notification"""
    
    load_env()
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '')
    
    logger.info(f"=== Daily Notification: {date_str} ===")
    
    # Load data
    base_path = 'data/processed/preprocessed_data_v11.parquet'
    logger.info("Loading base data...")
    df = pd.read_parquet(base_path)
    
    # Filter to target date
    if 'date' in df.columns:
        df['_date'] = pd.to_datetime(df['date'])
        df = df[df['_date'].dt.strftime('%Y-%m-%d') == date_str]
    else:
        target_prefix = date_str.replace('-', '')
        df = df[df['race_id'].astype(str).str[:8] == target_prefix]
    
    if len(df) == 0:
        msg = f"⚠️ {date_str} のデータがありません"
        logger.warning(msg)
        if not dry_run and webhook_url:
            send_discord_message(webhook_url, msg)
        return {'status': 'no_data', 'date': date_str}
    
    # JRA filter
    from utils.race_filter import filter_jra_only
    df = filter_jra_only(df)
    
    logger.info(f"Found {df['race_id'].nunique()} races, {len(df)} runners")
    
    # Load models
    model_dir = 'models/v13_market_residual'
    models = load_models(model_dir)
    logger.info(f"Loaded {len(models)} models")
    
    # Run inference
    df = run_inference(df, models)
    
    # Venue mapping
    course_map = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
        '05': '東京', '06': '中山', '07': '中京', '08': '京都',
        '09': '阪神', '10': '小倉'
    }
    
    # Generate tickets for each race
    races_data = []
    
    for race_id in sorted(df['race_id'].unique()):
        race_df = df[df['race_id'] == race_id].copy()
        
        if len(race_df) < 4:
            continue
        
        # Parse race_id
        rid_str = str(race_id)
        venue_code = rid_str[4:6] if len(rid_str) >= 6 else ''
        race_num = rid_str[10:12] if len(rid_str) >= 12 else ''
        venue = course_map.get(venue_code, f'場{venue_code}')
        
        # Generate tickets
        tickets, top_horses, top_names = generate_tickets(race_df, topn=4)
        
        races_data.append({
            'race_id': race_id,
            'venue': venue,
            'race_num': int(race_num) if race_num.isdigit() else race_num,
            'race_name': race_df.iloc[0].get('race_name', '') if 'race_name' in race_df.columns else '',
            'tickets': tickets,
            'top_horses': top_horses,
            'top_names': top_names
        })
    
    logger.info(f"Generated tickets for {len(races_data)} races")
    
    # Format notification
    notification = format_notification(date_str, races_data)
    
    # Send notification
    if dry_run:
        logger.info("Dry run - not sending notification")
        print("\n" + "=" * 50)
        print("NOTIFICATION PREVIEW:")
        print("=" * 50)
        print(notification)
        print("=" * 50)
    else:
        if webhook_url:
            # Discord has 2000 char limit, split if needed
            if len(notification) > 1900:
                # Split into chunks
                chunks = []
                lines = notification.split('\n')
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > 1900:
                        chunks.append(current_chunk)
                        current_chunk = line + "\n"
                    else:
                        current_chunk += line + "\n"
                if current_chunk:
                    chunks.append(current_chunk)
                
                for i, chunk in enumerate(chunks):
                    send_discord_message(webhook_url, chunk)
                    if i < len(chunks) - 1:
                        import time
                        time.sleep(1)
            else:
                send_discord_message(webhook_url, notification)
        else:
            logger.warning("DISCORD_WEBHOOK_URL not set, printing to console")
            print(notification)
    
    # Save to file
    out_dir = Path(f'data/paper_trade/{date_str}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'notification.txt', 'w', encoding='utf-8') as f:
        f.write(notification)
    
    return {
        'status': 'success',
        'date': date_str,
        'races': len(races_data),
        'total_tickets': sum(len(r['tickets']) for r in races_data)
    }


def main():
    parser = argparse.ArgumentParser(description="Daily Notification")
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD), default=today')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview only, do not send')
    
    args = parser.parse_args()
    
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    result = run_daily_notification(date_str, dry_run=args.dry_run)
    
    logger.info(f"Result: {result}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
