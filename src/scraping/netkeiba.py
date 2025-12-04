import time
import requests
from typing import Optional
import logging

# Logger settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetkeibaScraper:
    """
    Class to scrape data from netkeiba.com.
    Includes rate limiting to be polite to the server.
    """
    BASE_URL = "https://db.netkeiba.com"

    def __init__(self, sleep_time: float = 1.0):
        """
        Args:
            sleep_time (float): Wait time in seconds between requests. Default is 1.0.
        """
        self.sleep_time = sleep_time
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_html(self, url: str) -> Optional[bytes]:
        """
        Fetches HTML from the given URL with rate limiting and error handling.

        Args:
            url (str): Target URL.

        Returns:
            Optional[bytes]: HTML content in bytes, or None if failed.
        """
        time.sleep(self.sleep_time)
        try:
            logger.info(f"Fetching URL: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            # netkeiba uses EUC-JP encoding often, but requests might auto-detect.
            # We return bytes to let parser handle encoding or decoding.
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_race_page(self, race_id: str) -> Optional[bytes]:
        """
        Fetches the race result page for a given race_id.

        Args:
            race_id (str): Race ID (e.g., "202306050811").

        Returns:
            Optional[bytes]: HTML content.
        """
        url = f"{self.BASE_URL}/race/{race_id}/"
        return self._get_html(url)
