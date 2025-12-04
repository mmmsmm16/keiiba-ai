import time
import requests
import random
from typing import Optional
import logging

# ロガー設定
logger = logging.getLogger(__name__)

class FatalScrapingError(Exception):
    """リトライ後も回復不能なスクレイピングエラーを表す例外。"""
    pass

class NetkeibaScraper:
    """
    netkeiba.com からデータをスクレイピングするクラス。
    サーバー負荷を考慮し、レート制限（待機時間）とリトライ機能を含みます。
    """
    BASE_URL = "https://db.netkeiba.com"

    # ユーザーエージェントのリスト (ローテーション用)
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    ]

    def __init__(self, sleep_time: float = 1.0, max_retries: int = 3, cooldown_time: float = 60.0):
        """
        初期化メソッド。

        Args:
            sleep_time (float): リクエスト間の待機時間（秒）。デフォルト1.0。
            max_retries (int): エラー時の最大リトライ回数。デフォルト3。
            cooldown_time (float): 連続エラー時のクールダウン待機時間（秒）。デフォルト60.0。
        """
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.cooldown_time = cooldown_time
        self.session = requests.Session()
        self._rotate_user_agent()

    def _rotate_user_agent(self):
        """ユーザーエージェントをランダムに変更します。"""
        ua = random.choice(self.USER_AGENTS)
        self.session.headers.update({'User-Agent': ua})
        # logger.debug(f"User-Agentを変更しました: {ua}")

    def _get_html(self, url: str) -> Optional[bytes]:
        """
        指定されたURLからHTMLを取得します。

        Args:
            url (str): 対象のURL。

        Returns:
            Optional[bytes]: HTMLコンテンツ（バイト列）。404の場合はNone。

        Raises:
            FatalScrapingError: リトライ回数を超えても取得できない場合（404以外）。
        """
        retries = 0
        while retries <= self.max_retries:
            time.sleep(self.sleep_time)
            try:
                logger.info(f"URLを取得中 (試行 {retries+1}/{self.max_retries+1}): {url}")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                logger.error(f"{url} の取得に失敗しました: {e}")

                # 404の場合はリトライしても無駄なのでNoneを返して終了
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    logger.info("404 Not Found. リトライしません。")
                    return None

                retries += 1
                if retries <= self.max_retries:
                    # 2回目以降の失敗は長めに待機
                    wait = self.cooldown_time if retries > 1 else self.sleep_time * 2
                    logger.warning(f"待機時間 {wait}秒 後にリトライします...")
                    time.sleep(wait)
                    self._rotate_user_agent() # 失敗したらUAを変えてみる
                else:
                    logger.error("最大リトライ回数を超えました。")
                    raise FatalScrapingError(f"Failed to fetch {url} after {self.max_retries} retries.")
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
