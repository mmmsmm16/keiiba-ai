import unittest
from unittest.mock import MagicMock
from scraping.bulk_loader import scrape_year

class TestBulkLoader(unittest.TestCase):
    def test_scrape_year(self):
        # Mock loader
        loader = MagicMock()

        # Mock scraper
        scraper = MagicMock()

        # Scenario:
        # Venue 01, Kai 01, Day 01 exists (returns HTML)
        # Venue 01, Kai 01, Day 02 does not exist (returns None)
        # Venue 01, Kai 02 does not exist (returns None)
        # Venue 02... does not exist

        def side_effect(race_id):
            # 202301010101 -> exists
            if race_id == "202301010101":
                return b"<html>...</html>"
            # 202301010102...0112 -> exists (loop race)
            if race_id.startswith("2023010101") and int(race_id[-2:]) <= 12:
                return b"<html>...</html>"

            # Check for Day 1 existence (check_day_id uses race 01)
            # 202301010101 (Day 1 Race 1) -> handled above

            # Check for Day 2 existence: 202301010201
            if race_id == "202301010201":
                return None

            # Check for Kai 2 existence: 202301020101
            if race_id == "202301020101":
                return None

            # Check for Venue 2 Kai 1 Day 1 Race 1: 202302010101
            if race_id == "202302010101":
                return None

            return None

        scraper.get_race_page.side_effect = side_effect

        # Run scrape_year for 2023
        # We expect it to process Venue 1, Kai 1, Day 1 (12 races).
        # Then check Day 2 (fail), check Kai 2 (fail).
        # Then check Venue 2...10 (fail).

        scrape_year(2023, loader, scraper)

        # Verify loader.save_race_data was called 12 times
        self.assertEqual(loader.save_race_data.call_count, 12)
        print("Bulk loader logic verified: Processed 12 races as expected.")

if __name__ == '__main__':
    unittest.main()
