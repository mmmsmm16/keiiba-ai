import time
import requests
from typing import Optional
import logging

# ロガー設定
logger = logging.getLogger(__name__)

class NetkeibaScraper:
    """
    netkeiba.com からデータをスクレイピングするクラス。
    サーバー負荷を考慮し、レート制限（待機時間）を含みます。
    """
    BASE_URL = "https://db.netkeiba.com"

    def __init__(self, sleep_time: float = 1.0):
        """
        初期化メソッド。

        Args:
            sleep_time (float): リクエスト間の待機時間（秒）。デフォルトは1.0秒。
        """
        self.sleep_time = sleep_time
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_html(self, url: str) -> Optional[bytes]:
        """
        指定されたURLからHTMLを取得します。レート制限とエラーハンドリングを含みます。

        Args:
            url (str): 対象のURL。

        Returns:
            Optional[bytes]: HTMLコンテンツ（バイト列）。失敗した場合はNone。
        """
        time.sleep(self.sleep_time)
        try:
            logger.info(f"URLを取得中: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            # netkeibaはEUC-JPエンコーディングを使用することが多いですが、requestsが自動検出する場合もあります。
            # パース側でエンコーディングやデコードを処理させるため、バイト列を返します。
            return response.content
        except requests.RequestException as e:
            logger.error(f"{url} の取得に失敗しました: {e}")
            return None

    def get_race_page(self, race_id: str) -> Optional[bytes]:
        """
        指定されたレースIDのレース結果ページを取得します。

        Args:
            race_id (str): レースID（例: "202306050811"）。

        Returns:
            Optional[bytes]: HTMLコンテンツ。
        """
        url = f"{self.BASE_URL}/race/{race_id}/"
        return self._get_html(url)
