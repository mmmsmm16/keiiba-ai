import sys
import os
import logging

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scraping.netkeiba import NetkeibaScraper
from scraping.parser import NetkeibaParser
from scraping.loader import DataLoader

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 例: 2023年 ホープフルステークス
    race_id = "202306050911"

    logger.info("スクレイパーを初期化中...")
    scraper = NetkeibaScraper(sleep_time=1.0)

    logger.info(f"レース {race_id} を取得中...")
    html = scraper.get_race_page(race_id)

    if not html:
        logger.error("HTMLの取得に失敗しました")
        return

    logger.info("HTMLをパース中...")
    data = NetkeibaParser.parse_race_page(html, race_id)

    logger.info("パースデータ概要:")
    for key, df in data.items():
        logger.info(f"--- {key} ---")
        logger.info(f"形状: {df.shape}")
        if not df.empty:
            logger.info(f"\n{df.head()}")
        else:
            logger.warning(f"データフレーム {key} は空です。")

    # DBへの保存を試行
    try:
        logger.info("Loaderを初期化中...")
        loader = DataLoader()
        loader.save_race_data(data)
        logger.info("データがDBに正常に保存されました（DBが稼働している場合）。")
    except Exception as e:
        logger.info(f"DBへの保存をスキップしました（この環境ではDBが稼働していない可能性があります）: {e}")

if __name__ == "__main__":
    main()
