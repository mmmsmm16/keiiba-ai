import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import NotificationManager from auto_predict
# auto_predict.py is in src/scripts, so from scripts.auto_predict ...
try:
    from src.scripts.auto_predict import NotificationManager
except ImportError:
    # Try relative import if run from different location
    sys.path.append(os.path.dirname(__file__))
    from auto_predict import NotificationManager

# Logging
logging.basicConfig(level=logging.INFO)
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

def main():
    load_env_manual()
    # Reload from env
    # DISCORD_WEBHOOK_URL is loaded into os.environ by load_env_manual
    # So we can just get it directly.
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')

    print(f"Checking Webhook URL in env: {webhook_url}")
    
    if not webhook_url:

        logger.error("❌ DISCORD_WEBHOOK_URL が設定されていません。 .env ファイルを確認してください。")
        return

    logger.info(f"Testing Discord Webhook...")
    
    # Pass the loaded webhook_url, not the imported one (which is likely None)
    manager = NotificationManager(webhook_url)
    
    # ダミーデータ (Dummy Data)
    race_meta = {
        'race_id': 'TEST_NOTIFICATION',
        'title': '通知テスト記念',
        'race_number': '11',
        'start_time': datetime.now().strftime("%H:%M"),
        'venue_name': '東京'
    }
    
    # ダミー予測データ
    df = pd.DataFrame({
        'horse_number': [1, 7, 10, 3, 5],
        'horse_name': ['テストワン', 'サンプルトップ', 'モックヒーロー', 'デバッグスター', 'トライアルキング'],
        'expected_value': [1.8, 1.1, 0.7, 0.6, 0.5],
        'calibrated_prob': [0.45, 0.20, 0.15, 0.10, 0.05],
        'score': [2.5, 1.2, 0.8, 0.5, 0.2],
        'popularity': [1, 2, 3, 4, 5],
        'odds': [4.0, 5.5, 10.0, 20.0, 50.0]
    })
    
    try:
        manager.send_prediction(race_meta, df)
        logger.info("✅ テスト通知を送信しました！Discordを確認してください。")
    except Exception as e:
        logger.error(f"❌ 送信に失敗しました: {e}")

if __name__ == "__main__":
    main()
