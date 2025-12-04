import unittest
from unittest.mock import MagicMock
from scraping.bulk_loader import scrape_year

class TestBulkLoader(unittest.TestCase):
    def test_scrape_year(self):
        # Loaderのモック化
        loader = MagicMock()

        # Scraperのモック化
        scraper = MagicMock()

        # シナリオ:
        # 場所 01, 回 01, 日 01 は存在する (HTMLを返す)
        # 場所 01, 回 01, 日 02 は存在しない (Noneを返す)
        # 場所 01, 回 02 は存在しない (Noneを返す)
        # 場所 02... は存在しない

        def side_effect(race_id):
            # 202301010101 -> 存在する
            if race_id == "202301010101":
                return b"<html>...</html>"
            # 202301010102...0112 -> 存在する (レースループ)
            if race_id.startswith("2023010101") and int(race_id[-2:]) <= 12:
                return b"<html>...</html>"

            # 2日目の存在チェック (check_day_id は第1レースを使用)
            # 202301010101 (1日目 第1レース) -> 上記で処理済み

            # 2日目の存在チェック: 202301010201
            if race_id == "202301010201":
                return None

            # 2回目の存在チェック: 202301020101
            if race_id == "202301020101":
                return None

            # 場所2 回1 日1 第1レースのチェック: 202302010101
            if race_id == "202302010101":
                return None

            return None

        scraper.get_race_page.side_effect = side_effect

        # 2023年のスクレイピングを実行
        # 場所1, 回1, 日1 (12レース) が処理されることを期待
        # その後、日2 (失敗), 回2 (失敗) を確認
        # その後、場所2...10 (失敗) を確認

        scrape_year(2023, loader, scraper)

        # loader.save_race_data が12回呼び出されたか検証
        self.assertEqual(loader.save_race_data.call_count, 12)
        print("Bulk loader ロジック検証完了: 期待通り12レースが処理されました。")

if __name__ == '__main__':
    unittest.main()
