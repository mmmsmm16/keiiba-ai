import time
import logging
import argparse
from tqdm import tqdm
import sys
import os

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scraping.netkeiba import NetkeibaScraper, FatalScrapingError
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
    total_races = 0
    consecutive_fatal_errors = 0

    for venue in range(1, 11): # 01 〜 10
        venue_id = f"{venue:02d}"
        logger.info(f"{year}年 開催場所 {venue_id} のスクレイピングを開始...")

        for kai in range(1, 13):
            kai_id = f"{kai:02d}"

            # 1日目第1レースを試して、この「回」が存在するか確認
            check_id = f"{year}{venue_id}{kai_id}0101"

            try:
                if not _check_race_exists(scraper, check_id, loader):
                    if kai == 1:
                        pass
                    else:
                        break

                    if kai == 1:
                         break
                consecutive_fatal_errors = 0 # 成功
            except FatalScrapingError:
                consecutive_fatal_errors += 1
                logger.error(f"開催存在確認で致命的なエラー (連続: {consecutive_fatal_errors}): {check_id}")
                if consecutive_fatal_errors >= 5:
                    logger.critical("連続エラー過多のため停止します。")
                    sys.exit(1)
                continue

            for day in range(1, 13):
                day_id = f"{day:02d}"

                # 第1レースを試して、この「日」が存在するか確認
                check_day_id = f"{year}{venue_id}{kai_id}{day_id}01"
                try:
                    if not _check_race_exists(scraper, check_day_id, loader):
                        break
                    consecutive_fatal_errors = 0
                except FatalScrapingError:
                    consecutive_fatal_errors += 1
                    logger.error(f"日次存在確認で致命的なエラー (連続: {consecutive_fatal_errors}): {check_day_id}")
                    if consecutive_fatal_errors >= 5:
                        sys.exit(1)
                    continue

                for race in range(1, 13):
                    race_num = f"{race:02d}"
                    race_id = f"{year}{venue_id}{kai_id}{day_id}{race_num}"

                    # 既にDBにあるかチェック
                    if loader and hasattr(loader, 'check_race_exists') and loader.check_race_exists(race_id):
                        logger.info(f"スキップ: {race_id} (DBに存在)")
                        consecutive_fatal_errors = 0
                        continue

                    try:
                        html = scraper.get_race_page(race_id)
                        consecutive_fatal_errors = 0 # 成功

                        if html:
                            data = NetkeibaParser.parse_race_page(html, race_id)
                            loader.save_race_data(data)
                            total_races += 1
                            if total_races % 10 == 0:
                                logger.info(f"{total_races} レース保存完了 (最新: {race_id})")
                        else:
                            pass
                    except FatalScrapingError:
                        consecutive_fatal_errors += 1
                        logger.error(f"レース取得で致命的なエラー (連続: {consecutive_fatal_errors}): {race_id}")
                        if consecutive_fatal_errors >= 5:
                            logger.critical("連続エラー過多のため停止します。")
                            sys.exit(1)
                    except Exception as e:
                        logger.error(f"{race_id} の処理中に予期せぬエラー: {e}")

    logger.info(f"{year}年のスクレイピング完了。合計レース数: {total_races}")

def _check_race_exists(scraper: NetkeibaScraper, race_id: str, loader: DataLoader = None) -> bool:
    """
    レースが存在するかどうかを確認します。DB優先、なければスクレイピング。
    """
    if loader and hasattr(loader, 'check_race_exists') and loader.check_race_exists(race_id):
        return True

    # get_race_page は 404 なら None を返す。FatalScrapingError なら送出する。
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
        try:
            loader = DataLoader()
        except Exception as e:
            logger.error(f"DBへの接続に失敗しました: {e}")
            return
    else:
        logger.info("DRY RUNモードで実行中。データは保存されません。")
        class DummyLoader:
            def save_race_data(self, data):
                pass
            def check_race_exists(self, race_id):
                return False
        loader = DummyLoader()

    for year in range(args.year_start, args.year_end + 1):
        scrape_year(year, loader, scraper)

if __name__ == "__main__":
    main()
