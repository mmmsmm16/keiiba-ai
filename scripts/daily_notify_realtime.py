"""
Daily Notification (Real-time): DB → 前処理 → V13推論 → Discord通知
Phase 11: リアルタイム自動通知システム

Usage:
    # 今日の予測＆通知
    docker compose exec app python scripts/daily_notify_realtime.py

    # 特定日付
    docker compose exec app python scripts/daily_notify_realtime.py --date 2025-12-21

    # プレビューのみ（送信しない）
    docker compose exec app python scripts/daily_notify_realtime.py --dry-run

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
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.loader import InferenceDataLoader
from inference.preprocessor import InferencePreprocessor

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


def load_v13_models(model_dir: str) -> List[lgb.Booster]:
    """Load v13 fold models"""
    models = []
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
    for f in model_files:
        path = os.path.join(model_dir, f)
        models.append(lgb.Booster(model_file=path))
        logger.info(f"Loaded model: {f}")
    return models


def run_v13_inference(df: pd.DataFrame, models: List[lgb.Booster]) -> pd.DataFrame:
    """Run v13 inference with ensemble"""
    from scipy.special import expit
    
    # Get feature columns from first model
    feature_cols = models[0].feature_name()
    logger.info(f"Model expects {len(feature_cols)} features")
    
    # Check missing
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features, filling with 0")
        for c in missing:
            df[c] = 0
    
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


def generate_tickets(race_df: pd.DataFrame, topn: int = 4) -> tuple:
    """Generate sanrenpuku BOX tickets"""
    top_horses = race_df.nlargest(topn, 'prob')
    horse_numbers = top_horses['horse_number'].astype(int).tolist()
    
    # Get horse names
    if 'horse_name' in top_horses.columns:
        horse_names = top_horses['horse_name'].fillna('').str.strip().tolist()
    else:
        horse_names = [f'馬{h}' for h in horse_numbers]
    
    tickets = []
    for comb in combinations(range(len(horse_numbers)), 3):
        nums = [horse_numbers[i] for i in comb]
        names = [horse_names[i] for i in comb]
        tickets.append({
            'numbers': nums,
            'names': names,
            'display': '-'.join(map(str, nums)),
            'name_display': '-'.join([n[:8] for n in names])  # Truncate for readability
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
        title = race.get('title', '')[:20]
        
        msg += f"**{venue} {race_num}R** {title}\n"
        
        # Top horses summary
        horse_info = ', '.join([f"{n}番{h[:6]}" for n, h in zip(race['top_horses'], race['top_names'])])
        msg += f"推奨: {horse_info}\n"
        
        # Tickets
        ticket_strs = [t['display'] for t in race['tickets']]
        msg += f"買目: {', '.join(ticket_strs)} (各¥100)\n\n"
        
        total_tickets += len(race['tickets'])
        total_stake += len(race['tickets']) * 100
    
    msg += "=" * 30 + "\n"
    msg += f"合計: {len(races_data)}R, {total_tickets}点, ¥{total_stake:,}\n"
    msg += "\n⚠️ 購入は自己責任で"
    
    return msg


def run_daily_notification_realtime(date_str: str, dry_run: bool = False) -> Dict[str, Any]:
    """Run daily prediction with real-time data loading"""
    
    load_env()
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '')
    
    logger.info(f"=== Daily Notification (Real-time): {date_str} ===")
    
    # 1. Load data from DB
    logger.info("Loading data from database...")
    loader = InferenceDataLoader()
    target_date = date_str.replace('-', '')  # YYYYMMDD format
    
    try:
        raw_df = loader.load(target_date=target_date)
    except Exception as e:
        logger.error(f"Database load failed: {e}")
        return {'status': 'db_error', 'error': str(e)}
    
    if raw_df.empty:
        msg = f"⚠️ {date_str} のレースデータがありません\nDBにデータが登録されているか確認してください"
        logger.warning(msg)
        if not dry_run and webhook_url:
            send_discord_message(webhook_url, msg)
        return {'status': 'no_data', 'date': date_str}
    
    logger.info(f"Loaded {len(raw_df)} entries from DB")
    
    # 2. Preprocess (with history)
    logger.info("Running preprocessing...")
    preprocessor = InferencePreprocessor()
    
    try:
        X, ids, full_df = preprocessor.preprocess(raw_df, return_full_df=True)
        logger.info(f"Preprocessed: {len(full_df)} rows, {X.shape[1]} features")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {'status': 'preprocess_error', 'error': str(e)}
    
    # Merge back key columns
    if 'race_id' not in full_df.columns:
        full_df['race_id'] = ids['race_id']
    if 'horse_number' not in full_df.columns:
        full_df['horse_number'] = ids['horse_number']
    
    # Copy extra columns from raw_df
    for col in ['horse_name', 'venue', 'race_number', 'title']:
        if col in raw_df.columns and col not in full_df.columns:
            merge_key = 'race_id' if col in ['venue', 'race_number', 'title'] else ['race_id', 'horse_number']
            if isinstance(merge_key, str):
                full_df = full_df.merge(
                    raw_df[['race_id', col]].drop_duplicates('race_id'),
                    on='race_id', how='left'
                )
            else:
                full_df = full_df.merge(
                    raw_df[['race_id', 'horse_number', col]].drop_duplicates(['race_id', 'horse_number']),
                    on=['race_id', 'horse_number'], how='left'
                )
    
    # 3. Load models and run inference
    logger.info("Loading v13 models...")
    model_dir = 'models/v13_market_residual'
    models = load_v13_models(model_dir)
    
    logger.info("Running inference...")
    full_df = run_v13_inference(full_df, models)
    
    # 4. Venue mapping
    course_map = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
        '05': '東京', '06': '中山', '07': '中京', '08': '京都',
        '09': '阪神', '10': '小倉'
    }
    
    # 5. Generate tickets for each race
    races_data = []
    
    for race_id in sorted(full_df['race_id'].unique()):
        race_df = full_df[full_df['race_id'] == race_id].copy()
        
        if len(race_df) < 4:
            logger.warning(f"Race {race_id} has less than 4 horses, skipping")
            continue
        
        # Parse venue from race_id or df
        if 'venue' in race_df.columns and race_df['venue'].notna().any():
            venue_code = str(race_df['venue'].iloc[0]).zfill(2)
        else:
            rid_str = str(race_id)
            venue_code = rid_str[4:6] if len(rid_str) >= 6 else '??'
        
        venue = course_map.get(venue_code, f'場{venue_code}')
        
        # Race number
        if 'race_number' in race_df.columns and race_df['race_number'].notna().any():
            race_num = int(race_df['race_number'].iloc[0])
        else:
            rid_str = str(race_id)
            race_num = int(rid_str[10:12]) if len(rid_str) >= 12 and rid_str[10:12].isdigit() else '?'
        
        # Title
        title = ''
        if 'title' in race_df.columns and race_df['title'].notna().any():
            title = str(race_df['title'].iloc[0])[:20]
        
        # Generate tickets
        tickets, top_horses, top_names = generate_tickets(race_df, topn=4)
        
        races_data.append({
            'race_id': race_id,
            'venue': venue,
            'race_num': race_num,
            'title': title,
            'tickets': tickets,
            'top_horses': top_horses,
            'top_names': top_names
        })
    
    logger.info(f"Generated tickets for {len(races_data)} races")
    
    if not races_data:
        msg = f"⚠️ {date_str} 有効なレースがありません"
        if not dry_run and webhook_url:
            send_discord_message(webhook_url, msg)
        return {'status': 'no_valid_races', 'date': date_str}
    
    # 6. Format notification
    notification = format_notification(date_str, races_data)
    
    # 7. Send notification
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
                
                import time
                for i, chunk in enumerate(chunks):
                    send_discord_message(webhook_url, chunk)
                    if i < len(chunks) - 1:
                        time.sleep(1)
            else:
                send_discord_message(webhook_url, notification)
        else:
            logger.warning("DISCORD_WEBHOOK_URL not set, printing to console")
            print(notification)
    
    # 8. Save to file
    out_dir = Path(f'data/paper_trade/{date_str}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'notification.txt', 'w', encoding='utf-8') as f:
        f.write(notification)
    
    # Save predictions
    pred_cols = ['race_id', 'horse_number', 'prob', 'prob_raw']
    if 'horse_name' in full_df.columns:
        pred_cols.append('horse_name')
    full_df[pred_cols].to_parquet(out_dir / 'predictions.parquet', index=False)
    
    return {
        'status': 'success',
        'date': date_str,
        'races': len(races_data),
        'total_tickets': sum(len(r['tickets']) for r in races_data)
    }


def main():
    parser = argparse.ArgumentParser(description="Daily Notification (Real-time)")
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD), default=today')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview only, do not send')
    
    args = parser.parse_args()
    
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    result = run_daily_notification_realtime(date_str, dry_run=args.dry_run)
    
    logger.info(f"Result: {result}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
