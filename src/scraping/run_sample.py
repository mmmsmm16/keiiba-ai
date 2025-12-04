import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scraping.netkeiba import NetkeibaScraper
from scraping.parser import NetkeibaParser
from scraping.loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Example: Hopeful Stakes 2023
    race_id = "202306050911"

    logger.info("Initializing Scraper...")
    scraper = NetkeibaScraper(sleep_time=1.0)

    logger.info(f"Fetching race {race_id}...")
    html = scraper.get_race_page(race_id)

    if not html:
        logger.error("Failed to fetch HTML")
        return

    logger.info("Parsing HTML...")
    data = NetkeibaParser.parse_race_page(html, race_id)

    logger.info("Parsed Data Summary:")
    for key, df in data.items():
        logger.info(f"--- {key} ---")
        logger.info(f"Shape: {df.shape}")
        if not df.empty:
            logger.info(f"\n{df.head()}")
        else:
            logger.warning(f"DataFrame {key} is empty.")

    # Try to load to DB
    try:
        logger.info("Initializing Loader...")
        loader = DataLoader()
        loader.save_race_data(data)
        logger.info("Data saved to DB successfully (if DB is running).")
    except Exception as e:
        logger.info(f"Skipping DB save (DB likely not running in this environment): {e}")

if __name__ == "__main__":
    main()
