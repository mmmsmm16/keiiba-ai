import time
import logging
import argparse
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scraping.netkeiba import NetkeibaScraper
from scraping.parser import NetkeibaParser
from scraping.loader import DataLoader

# Configure logging
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
    Scrapes all races for a given year.
    Optimization: Skips invalid Kai/Day combinations.
    """
    # Venues: 01 to 10
    # Kai: 01 to 06 (Estimated max)
    # Day: 01 to 12 (Estimated max)
    # Race: 01 to 12

    total_races = 0

    for venue in range(1, 11): # 01 to 10
        venue_id = f"{venue:02d}"
        logger.info(f"Scraping Venue {venue_id} for Year {year}...")

        for kai in range(1, 13): # Allow up to 12 just in case
            kai_id = f"{kai:02d}"

            # Check if this Kai exists by trying Day 1 Race 1
            check_id = f"{year}{venue_id}{kai_id}0101"
            if not _check_race_exists(scraper, check_id):
                if kai == 1:
                    # If Kai 01 doesn't exist, this venue might not have races this year?
                    # Or maybe we just continue? Usually Kai starts at 01.
                    # But to be safe, we break if Kai=1 fails?
                    # Wait, some venues might not hold races every year?
                    # But if Kai 1 fails, Kai 2 unlikely exists.
                    pass
                else:
                    # If Kai X fails, assume no more Kai for this venue
                    break
                # If Kai 1 failed, we break too (optimization)
                if kai == 1:
                     # However, Sapporo might start later? No, Kai is just a counter.
                     # If check fails, we assume Kai doesn't exist.
                     break

            # If Kai exists, loop Days
            for day in range(1, 13):
                day_id = f"{day:02d}"

                # Check if Day exists by trying Race 1
                check_day_id = f"{year}{venue_id}{kai_id}{day_id}01"
                if not _check_race_exists(scraper, check_day_id):
                    break # Assume no more days in this Kai

                # If Day exists, loop Races
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
                                logger.info(f"Saved {total_races} races (Latest: {race_id})")
                        else:
                            # Race doesn't exist (e.g. only 11 races)
                            pass
                    except Exception as e:
                        logger.error(f"Error processing {race_id}: {e}")

    logger.info(f"Finished scraping Year {year}. Total races: {total_races}")

def _check_race_exists(scraper: NetkeibaScraper, race_id: str) -> bool:
    """
    Checks if a race exists by fetching it.
    This consumes a request and time.
    """
    html = scraper.get_race_page(race_id)
    return html is not None and len(html) > 0

def main():
    parser = argparse.ArgumentParser(description="Bulk scrape netkeiba data.")
    parser.add_argument("--year_start", type=int, required=True, help="Start year (e.g., 2014)")
    parser.add_argument("--year_end", type=int, required=True, help="End year (e.g., 2023)")
    parser.add_argument("--dry_run", action="store_true", help="Run without saving to DB")

    args = parser.parse_args()

    scraper = NetkeibaScraper(sleep_time=1.0)

    loader = None
    if not args.dry_run:
        # Initialize Loader (will fail if no DB, so we handle that)
        try:
            loader = DataLoader()
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            logger.error("Please ensure docker-compose is running or use --dry_run.")
            return
    else:
        logger.info("Running in DRY RUN mode. Data will not be saved.")
        # Create a dummy loader
        class DummyLoader:
            def save_race_data(self, data):
                pass
        loader = DummyLoader()

    for year in range(args.year_start, args.year_end + 1):
        scrape_year(year, loader, scraper)

if __name__ == "__main__":
    main()
