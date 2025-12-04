import time
import logging
import argparse
from tqdm import tqdm
import sys
import os

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scraping.netkeiba import NetkeibaScraper
from scraping.parser import NetkeibaParser
from scraping.loader import DataLoader

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/scraping.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def scrape_year(year: int, loader: DataLoader, scraper: NetkeibaScraper):
    """
    指定された年の全レースをスクレイピングします。
    最適化: 存在しない回(Kai)/日(Day)の組み合わせはスキップします。
    """
    # 開催場所: 01 〜 10
    # 回: 01 〜 06 (推定最大値)
    # 日: 01 〜 12 (推定最大値)
    # レース: 01 〜 12

    total_races = 0

    for venue in range(1, 11): # 01 〜 10
        venue_id = f"{venue:02d}"
        logger.info(f"{year}年 開催場所 {venue_id} のスクレイピングを開始...")

        for kai in range(1, 13): # 念のため12まで許可
            kai_id = f"{kai:02d}"

            # 1日目第1レースを試して、この「回」が存在するか確認
            check_id = f"{year}{venue_id}{kai_id}0101"
            if not _check_race_exists(scraper, check_id):
                if kai == 1:
                    # 第1回が存在しない場合、この場所での開催なしか、何かおかしい可能性がある。
                    # しかし、安全のためスキップする。
                    pass
                else:
                    # 第X回が存在しなければ、それ以降の回も存在しないと仮定してループを抜ける
                    break

                # 第1回が失敗した場合もループを抜ける
                if kai == 1:
                     break

            # 「回」が存在する場合、日ごとのループ
            for day in range(1, 13):
                day_id = f"{day:02d}"

                # 第1レースを試して、この「日」が存在するか確認
                check_day_id = f"{year}{venue_id}{kai_id}{day_id}01"
                if not _check_race_exists(scraper, check_day_id):
                    break # この回にはこれ以上の日程はないと仮定

                # 「日」が存在する場合、レースごとのループ
                for race in range(1, 13):
                    race_num = f"{race:02d}"
                    race_id = f"{year}{venue_id}{kai_id}{day_id}{race_num}"

                    try:
                        html = scraper.get_race_page(race_id)
                        if html:
                            data = NetkeibaParser.parse_race_page(html, race_id)
                            loader.save_race_data(data)
                            total_races += 1
                            if total_races % 10 == 0:
                                logger.info(f"{total_races} レース保存完了 (最新: {race_id})")
                        else:
                            # レースが存在しない (例: 11レースまでしかない場合)
                            pass
                    except Exception as e:
                        logger.error(f"{race_id} の処理中にエラーが発生しました: {e}")

    logger.info(f"{year}年のスクレイピング完了。合計レース数: {total_races}")

def _check_race_exists(scraper: NetkeibaScraper, race_id: str) -> bool:
    """
    レースが存在するかどうかをフェッチして確認します。
    リクエストと待機時間を消費します。
    """
    html = scraper.get_race_page(race_id)
    return html is not None and len(html) > 0

def main():
    parser = argparse.ArgumentParser(description="netkeibaデータのスクレイピングを一括実行します。")
    parser.add_argument("--year_start", type=int, required=True, help="開始年 (例: 2014)")
    parser.add_argument("--year_end", type=int, required=True, help="終了年 (例: 2023)")
    parser.add_argument("--dry_run", action="store_true", help="DBへの保存を行わずに実行")

    args = parser.parse_args()

    scraper = NetkeibaScraper(sleep_time=1.0)

    loader = None
    if not args.dry_run:
        # Loaderの初期化 (DBがない場合は失敗するためハンドリング)
        try:
            loader = DataLoader()
        except Exception as e:
            logger.error(f"DBへの接続に失敗しました: {e}")
            logger.error("docker-composeが起動しているか確認するか、--dry_runを使用してください。")
            return
    else:
        logger.info("DRY RUNモードで実行中。データは保存されません。")
        # ダミーのLoaderを作成
        class DummyLoader:
            def save_race_data(self, data):
                pass
        loader = DummyLoader()

    for year in range(args.year_start, args.year_end + 1):
        scrape_year(year, loader, scraper)

if __name__ == "__main__":
    main()
