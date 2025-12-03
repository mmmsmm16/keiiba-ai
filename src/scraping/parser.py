import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple

class NetkeibaParser:
    """
    Parses HTML content from netkeiba.com race pages.
    """

    @staticmethod
    def parse_race_page(html: bytes, race_id: str) -> Dict[str, pd.DataFrame]:
        """
        Parses the race page HTML and returns DataFrames for races, results, and payouts.

        Args:
            html (bytes): HTML content.
            race_id (str): Race ID.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing 'races', 'results', 'payouts', 'horses' DataFrames.
        """
        soup = BeautifulSoup(html, 'html.parser')

        # 1. Parse Race Info
        race_info = NetkeibaParser._parse_race_info(soup, race_id)
        df_races = pd.DataFrame([race_info])

        # 2. Parse Results
        results, horses = NetkeibaParser._parse_results(soup, race_id)
        df_results = pd.DataFrame(results)
        df_horses = pd.DataFrame(horses).drop_duplicates(subset=['horse_id'])

        # 3. Parse Payouts
        payouts = NetkeibaParser._parse_payouts(soup, race_id)
        df_payouts = pd.DataFrame(payouts)

        return {
            'races': df_races,
            'results': df_results,
            'horses': df_horses,
            'payouts': df_payouts
        }

    @staticmethod
    def _parse_race_info(soup: BeautifulSoup, race_id: str) -> Dict:
        """Parses race metadata."""
        # This selector is based on standard netkeiba layout
        # The title is usually in .racedata h1
        title = soup.select_one('.racedata h1')
        title_text = title.text.strip() if title else ""

        # Race details (Distance, Surface, Weather, etc.)
        # Usually in .racedata p span
        racedata_text = soup.select_one('.racedata p').text if soup.select_one('.racedata p') else ""

        # Example: "芝右2500m / 天候 : 晴 / 芝 : 良 / 発走 : 15:25"
        # Parse using regex
        surface = "Unknown"
        distance = 0
        weather = "Unknown"
        state = "Unknown"
        date_val = None

        # Surface & Distance
        if "芝" in racedata_text:
            surface = "芝"
        elif "ダ" in racedata_text:
            surface = "ダート"
        elif "障" in racedata_text:
            surface = "障害"

        dist_match = re.search(r'(\d+)m', racedata_text)
        if dist_match:
            distance = int(dist_match.group(1))

        # Weather
        if "天候 : 晴" in racedata_text:
            weather = "晴"
        elif "天候 : 曇" in racedata_text:
            weather = "曇"
        elif "天候 : 雨" in racedata_text:
            weather = "雨"
        elif "天候 : 小雨" in racedata_text:
            weather = "小雨"
        elif "天候 : 雪" in racedata_text:
            weather = "雪"

        # State (Going)
        if "良" in racedata_text:
            state = "良"
        elif "稍重" in racedata_text:
            state = "稍重"
        elif "重" in racedata_text:
            state = "重"
        elif "不良" in racedata_text:
            state = "不良"

        # Date is usually in .smalltxt (e.g., 2022年12月25日)
        smalltxt = soup.select_one('.smalltxt')
        if smalltxt:
            date_text = smalltxt.text.split()[0] # Take first part
            # Convert "2022年12月25日" to "2022-12-25"
            date_text = date_text.replace('年', '-').replace('月', '-').replace('日', '')
            date_val = date_text

        # Race number
        # Often in .racedata .data_intro dl dt (e.g., 11 R)
        race_number = 0
        r_num_elem = soup.select_one('.racedata dt')
        if r_num_elem:
            r_num_text = r_num_elem.text.strip()
            r_num_match = re.search(r'(\d+)', r_num_text)
            if r_num_match:
                race_number = int(r_num_match.group(1))

        venue = ""
        # Venue extraction is tricky from just text without mapping codes.
        # But usually "5回中山8日目" part exists.
        if smalltxt:
             parts = smalltxt.text.split()
             if len(parts) > 1:
                 venue_part = parts[1]
                 # Extract "中山" from "5回中山8日目"
                 venue_match = re.search(r'回(.*)\d+日目', venue_part)
                 if venue_match:
                     venue = venue_match.group(1)

        return {
            'race_id': race_id,
            'date': date_val,
            'venue': venue,
            'race_number': race_number,
            'distance': distance,
            'surface': surface,
            'weather': weather,
            'state': state,
            'title': title_text
        }

    @staticmethod
    def _parse_results(soup: BeautifulSoup, race_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Parses the results table."""
        results = []
        horses = []

        table = soup.select_one('table.race_table_01')
        if not table:
            return [], []

        rows = table.find_all('tr')
        # Skip header
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) < 10:
                continue

            try:
                rank_text = cols[0].text.strip()
                try:
                    rank = int(rank_text)
                except ValueError:
                    # Handle "取消", "中止", "失格" etc.
                    rank = None

                frame_number = int(cols[1].text.strip())
                horse_number = int(cols[2].text.strip())

                # Horse info
                horse_a = cols[3].find('a')
                horse_name = horse_a.text.strip()
                horse_id = ""
                if horse_a and 'href' in horse_a.attrs:
                    # /horse/2019105219/
                    horse_id = horse_a['href'].split('/')[-2]

                # Sex & Age (e.g., 牡3)
                sex_age = cols[4].text.strip()
                sex = sex_age[0]
                # age = int(sex_age[1:]) # Not storing age directly in results, implicitly in date - birthday

                # Weight (Jockey)
                weight_val = float(cols[5].text.strip())

                # Jockey
                jockey_a = cols[6].find('a')
                jockey_id = ""
                if jockey_a and 'href' in jockey_a.attrs:
                    jockey_id = jockey_a['href'].split('/')[-2]

                # Time
                time_str = cols[7].text.strip()
                time_sec = 0.0
                if time_str:
                     # 2:32.4 -> 152.4
                     parts = time_str.split(':')
                     if len(parts) == 2:
                         time_sec = float(parts[0]) * 60 + float(parts[1])
                     else:
                         try:
                            time_sec = float(time_str)
                         except:
                            time_sec = None

                # Passing rank
                # Sometimes in col 10 or somewhere?
                # Headers: 着順, 枠, 馬, 馬名, 性齢, 斤量, 騎手, タイム, 着差, 人気, 単勝オッズ, 後3F, コナー通過, 厩舎, ...
                # Wait, the column indices depend on the table layout.
                # Standard layout:
                # 0: rank, 1: frame, 2: horse_num, 3: name, 4: sex_age, 5: weight, 6: jockey, 7: time, 8: margin,
                # 9: popularity (sometimes), 10: odds (sometimes), ...
                # Actually, let's look at the text output from before to guess.
                # "タイム 着差 ﾀｲﾑ指数 通過 上り 単勝 人気 馬体重"
                # So:
                # 7: Time
                # 8: Margin
                # 9: Time Index (often empty or **)
                # 10: Passing (通過)
                # 11: Last 3F (上り)
                # 12: Odds (単勝)
                # 13: Popularity (人気)
                # 14: Horse Weight (馬体重)

                passing_rank = cols[10].text.strip()
                last_3f_text = cols[11].text.strip()
                last_3f = float(last_3f_text) if last_3f_text and last_3f_text.replace('.', '', 1).isdigit() else None

                odds_text = cols[12].text.strip()
                odds = float(odds_text) if odds_text and odds_text.replace('.', '', 1).isdigit() else None

                pop_text = cols[13].text.strip()
                popularity = int(pop_text) if pop_text and pop_text.isdigit() else None

                weight_text = cols[14].text.strip()
                # 492(+4) -> 492, +4
                horse_weight = None
                weight_diff = None
                match = re.match(r'(\d+)\((.*)\)', weight_text)
                if match:
                    horse_weight = int(match.group(1))
                    diff_str = match.group(2)
                    try:
                        weight_diff = int(diff_str)
                    except ValueError:
                        weight_diff = 0

                # Trainer
                trainer_a = cols[18].find('a') # Approx index
                # Let's search for trainer link
                trainer_id = ""
                for col in cols[15:]:
                    t_link = col.find('a')
                    if t_link and '/trainer/' in t_link.get('href', ''):
                        trainer_id = t_link['href'].split('/')[-2]
                        break

                results.append({
                    'race_id': race_id,
                    'horse_id': horse_id,
                    'jockey_id': jockey_id,
                    'trainer_id': trainer_id,
                    'frame_number': frame_number,
                    'horse_number': horse_number,
                    'rank': rank,
                    'time': time_sec,
                    'passing_rank': passing_rank,
                    'last_3f': last_3f,
                    'odds': odds,
                    'popularity': popularity,
                    'weight': horse_weight,
                    'weight_diff': weight_diff
                })

                horses.append({
                    'horse_id': horse_id,
                    'name': horse_name,
                    'sex': sex,
                    'birthday': None, # Need horse page for this
                    'sire_id': None,
                    'mare_id': None
                })

            except Exception as e:
                # logger.warning(f"Error parsing row in {race_id}: {e}")
                continue

        return results, horses

    @staticmethod
    def _parse_payouts(soup: BeautifulSoup, race_id: str) -> List[Dict]:
        """Parses payout information."""
        payouts = []

        # Payouts are usually in tables with class 'pay_block' or 'pay_table_01'
        # There might be multiple tables (tan/fuku, waku, uma, wide, etc.)
        tables = soup.select('table.pay_block, table.pay_table_01')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                th = row.find('th')
                if not th: continue
                ticket_type = th.text.strip() # 単勝, 複勝, etc.

                # There can be multiple winning numbers/payouts in one row (e.g. Fukusho)
                # Structure is tricky. Usually:
                # <th>TicketType</th> <td>Numbers</td> <td>Payout</td> <td>Popularity</td>
                # But if multiple, they are separated by <br> or in separate tds?
                # netkeiba uses <br> usually inside the td.

                cols = row.find_all('td')
                if len(cols) < 2: continue

                numbers_html = str(cols[0])
                payouts_html = str(cols[1])

                # Split by <br>
                nums = [n.strip() for n in numbers_html.replace('<td>', '').replace('</td>', '').split('<br/>') if n.strip()]
                pays = [p.strip().replace(',', '') for p in payouts_html.replace('<td>', '').replace('</td>', '').split('<br/>') if p.strip()]

                # Sometimes <br> is rendered as \n in text?
                # Let's use BeautifulSoup on the cell

                nums_cell = cols[0]
                pays_cell = cols[1]

                # Function to extract text lines
                def get_lines(cell):
                    return [text for text in cell.stripped_strings]

                num_lines = get_lines(nums_cell)
                pay_lines = get_lines(pays_cell)

                for n, p in zip(num_lines, pay_lines):
                    try:
                        payout_val = int(p.replace(',', ''))
                        payouts.append({
                            'race_id': race_id,
                            'ticket_type': ticket_type,
                            'winning_numbers': n,
                            'payout': payout_val
                        })
                    except ValueError:
                        continue

        return payouts
