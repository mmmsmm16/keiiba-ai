import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.scripts.auto_predict import AutoPredictor, NotificationManager, load_env_manual

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dummy_test():
    # 1. Load Env for Webhook
    load_env_manual()
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        logger.error("❌ DISCORD_WEBHOOK_URL not found.")
        return

    # 2. Initialize Manager (Use empty rules or load real ones if needed for betting display)
    # Trying to load real rules for realistic output
    predictor = AutoPredictor(dry_run=False) # Only to load rules/models, we won't call run()
    notifier = predictor.notifier
    
    # 3. Create Dummy Race Meta
    today_str = datetime.now().strftime('%Y-%m-%d')
    race_meta = {
        'race_id': '202599999901', # Dummy ID
        'date': today_str,
        'venue_name': 'テスト競馬場',
        'race_number': 11,
        'title': 'テスト記念 (Dummy Race)',
        'start_time': '18:30', # Target time
        'distance': 2400,
        'surface': 1, # Turf
        'weather': '晴',
        'state': '良'
    }

    # 4. Create Dummy Prediction DataFrame
    # Need columns: horse_number, horse_name, score, odds, prob
    data = [
        {'horse_number': 1, 'horse_name': 'ダミーホースワン', 'score': 0.95, 'prob': 0.45, 'odds': 2.5},
        {'horse_number': 2, 'horse_name': 'テストランナー',   'score': 0.88, 'prob': 0.25, 'odds': 4.8},
        {'horse_number': 3, 'horse_name': 'サンプルスター',   'score': 0.85, 'prob': 0.15, 'odds': 7.2},
        {'horse_number': 4, 'horse_name': 'モックアップ号',   'score': 0.80, 'prob': 0.10, 'odds': 12.5},
        {'horse_number': 5, 'horse_name': 'デバッガー',       'score': 0.70, 'prob': 0.05, 'odds': 25.0},
    ]
    df = pd.DataFrame(data)

    # 5. Create Dummy Features for Betting Rules
    # Setting values that trigger some betting rules
    # e.g. score_gap high, reasonable top1_odds
    race_features = {
        'score_gap': 0.07,   # High gap -> High Confidence
        'top1_odds': 2.5,    # Fav
        'avg_top3_odds': 4.8,
        'score_conc': 0.45,
        'n_horses': 5,
        'distance': 2400,
        'surface': 0, # Turf code assumption (preprocessor: 10-22=Turf -> 0?) 
                      # auto_predict.py line 382: surf = int(df['surface']) - 1. 
                      # loader creates string '芝', 'ダート'. 
                      # Wait, auto_predict.py line 382 expects numeric code from raw df?
                      # In auto_predict.py:
                      # if 'surface' in df.columns: try: surf = int(df['surface'].iloc[0]) - 1
                      # BUT loader returns string '芝'. int('芝') fails.
                      # Let's check auto_predict.py line 381 again.
                      # It seems auto_predict connects to `result_df` which comes from `ids` from preprocessor.
                      # `ids` in preprocessor has 'surface' from `inference_df`.
                      # `inference_df` 'surface' comes from `loader`.
                      # `loader` maps it to String ('芝').
                      # So `int(df['surface'])` in auto_predict.py line 382 MIGHT ERROR if it's string.
                      # Re-reading auto_predict.py:
                      # 381: if 'surface' in df.columns:
                      # 382:    try: surf = int(df['surface'].iloc[0]) - 1
                      # 383:    except: pass
                      # It catches exception, so surf stays 0.
                      # If betting rules depend on surf, this might be an issue. But for test it's fine.
        'venue': 5, # Tokyo
        'month': 12
    }

    logger.info("Sending Dummy Notification...")
    success = notifier.send_prediction(race_meta, df, race_features)
    
    if success:
        logger.info("✅ Notification Sent Successfully!")
    else:
        logger.error("❌ Notification Failed.")

if __name__ == "__main__":
    run_dummy_test()
