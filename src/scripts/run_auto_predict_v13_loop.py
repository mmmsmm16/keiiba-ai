"""
Auto Predict v13 ループ実行スクリプト

1分ごとにauto_predict_v13.pyを実行し、
発走5-15分前のレースを検出してDiscord通知を送信する。

Usage:
    docker compose exec app python src/scripts/run_auto_predict_v13_loop.py
"""
import os
import sys
import time
import subprocess
import logging
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../../logs/auto_predict_v13_loop.log'))
    ]
)
logger = logging.getLogger(__name__)

# 設定
CHECK_INTERVAL_SEC = 60  # チェック間隔（秒）
RACE_HOURS_START = 9     # レース開始時間帯（開始）
RACE_HOURS_END = 17      # レース開始時間帯（終了）

def is_race_day():
    """土日かどうかを判定"""
    today = datetime.now()
    return today.weekday() in [5, 6]  # 5=土, 6=日

def is_race_hours():
    """レース開催時間帯かどうかを判定"""
    now = datetime.now()
    return RACE_HOURS_START <= now.hour <= RACE_HOURS_END

def run_predict():
    """auto_predict_v13.pyを実行"""
    script_path = os.path.join(os.path.dirname(__file__), 'auto_predict_v13.py')
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            logger.info("予測実行成功")
            if result.stdout:
                for line in result.stdout.split('\n')[-5:]:
                    if line.strip():
                        logger.info(f"  {line}")
        else:
            logger.error(f"予測実行失敗: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("予測タイムアウト")
    except Exception as e:
        logger.error(f"予測エラー: {e}")

def main():
    logger.info("=== Auto Predict v13 ループ開始 ===")
    logger.info(f"チェック間隔: {CHECK_INTERVAL_SEC}秒")
    logger.info(f"対象時間帯: {RACE_HOURS_START}:00 - {RACE_HOURS_END}:00")
    
    while True:
        try:
            now = datetime.now()
            
            # 土日かつレース時間帯のみ実行
            if is_race_day() and is_race_hours():
                logger.info(f"チェック実行: {now.strftime('%H:%M:%S')}")
                run_predict()
            else:
                # 時間外は10分おきにログ
                if now.minute % 10 == 0 and now.second < 60:
                    if not is_race_day():
                        logger.info(f"レース日ではありません（{now.strftime('%A')}）")
                    else:
                        logger.info(f"レース時間外です（{now.hour}時）")
            
            time.sleep(CHECK_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            logger.info("ループ終了（Ctrl+C）")
            break
        except Exception as e:
            logger.error(f"ループエラー: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
