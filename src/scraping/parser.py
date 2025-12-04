import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple

class NetkeibaParser:
    """
    netkeiba.com のレースページHTMLをパースするクラス。
    """

    @staticmethod
    def parse_race_page(html: bytes, race_id: str) -> Dict[str, pd.DataFrame]:
        """
        レースページのHTMLをパースし、レース情報、結果、払い戻し情報のDataFrameを返します。

        Args:
            html (bytes): HTMLコンテンツ。
            race_id (str): レースID。

        Returns:
            Dict[str, pd.DataFrame]: 'races', 'results', 'payouts', 'horses' のDataFrameを含む辞書。
        """
        soup = BeautifulSoup(html, 'html.parser')

        # 1. レース情報のパース
        race_info = NetkeibaParser._parse_race_info(soup, race_id)
        df_races = pd.DataFrame([race_info])

        # 2. 結果情報のパース
        results, horses = NetkeibaParser._parse_results(soup, race_id)
        df_results = pd.DataFrame(results)
        df_horses = pd.DataFrame(horses).drop_duplicates(subset=['horse_id'])

        # 3. 払い戻し情報のパース
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
        """レースのメタデータをパースします。"""
        # 標準的なnetkeibaのレイアウトに基づくセレクタ
        # タイトルは通常 .racedata h1 にある
        title = soup.select_one('.racedata h1')
        title_text = title.text.strip() if title else ""

        # レース詳細 (距離, 馬場, 天候など)
        # 通常 .racedata p span にある
        racedata_text = soup.select_one('.racedata p').text if soup.select_one('.racedata p') else ""

        # 例: "芝右2500m / 天候 : 晴 / 芝 : 良 / 発走 : 15:25"
        # 正規表現を使用してパース
        surface = "Unknown"
        distance = 0
        weather = "Unknown"
        state = "Unknown"
        date_val = None

        # 馬場と距離
        if "芝" in racedata_text:
            surface = "芝"
        elif "ダ" in racedata_text:
            surface = "ダート"
        elif "障" in racedata_text:
            surface = "障害"

        dist_match = re.search(r'(\d+)m', racedata_text)
        if dist_match:
            distance = int(dist_match.group(1))

        # 天候
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

        # 馬場状態
        if "良" in racedata_text:
            state = "良"
        elif "稍重" in racedata_text:
            state = "稍重"
        elif "重" in racedata_text:
            state = "重"
        elif "不良" in racedata_text:
            state = "不良"

        # 日付は通常 .smalltxt にある (例: 2022年12月25日)
        smalltxt = soup.select_one('.smalltxt')
        if smalltxt:
            date_text = smalltxt.text.split()[0] # 最初の部分を取得
            # "2022年12月25日" を "2022-12-25" に変換
            date_text = date_text.replace('年', '-').replace('月', '-').replace('日', '')
            date_val = date_text

        # レース番号
        # 多くの場合 .racedata .data_intro dl dt (例: 11 R)
        race_number = 0
        r_num_elem = soup.select_one('.racedata dt')
        if r_num_elem:
            r_num_text = r_num_elem.text.strip()
            r_num_match = re.search(r'(\d+)', r_num_text)
            if r_num_match:
                race_number = int(r_num_match.group(1))

        venue = ""
        # 開催場所の抽出はテキストだけからは難しいが、通常 "5回中山8日目" という部分がある
        if smalltxt:
             parts = smalltxt.text.split()
             if len(parts) > 1:
                 venue_part = parts[1]
                 # "5回中山8日目" から "中山" を抽出
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
        """結果テーブルをパースします。"""
        results = []
        horses = []

        table = soup.select_one('table.race_table_01')
        if not table:
            return [], []

        rows = table.find_all('tr')
        # ヘッダーをスキップ
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) < 10:
                continue

            try:
                rank_text = cols[0].text.strip()
                try:
                    rank = int(rank_text)
                except ValueError:
                    # "取消", "中止", "失格" などの処理
                    rank = None

                frame_number = int(cols[1].text.strip())
                horse_number = int(cols[2].text.strip())

                # 馬情報
                horse_a = cols[3].find('a')
                horse_name = horse_a.text.strip()
                horse_id = ""
                if horse_a and 'href' in horse_a.attrs:
                    # /horse/2019105219/
                    horse_id = horse_a['href'].split('/')[-2]

                # 性別と年齢 (例: 牡3)
                sex_age = cols[4].text.strip()
                sex = sex_age[0]
                # age = int(sex_age[1:]) # 年齢は結果に直接保存せず、日付 - 誕生日から計算するためここでは省略

                # 斤量
                weight_val = float(cols[5].text.strip())

                # 騎手
                jockey_a = cols[6].find('a')
                jockey_id = ""
                if jockey_a and 'href' in jockey_a.attrs:
                    jockey_id = jockey_a['href'].split('/')[-2]

                # タイム
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

                # 通過順位など
                # カラムインデックスはテーブルレイアウトに依存します。
                # 標準的なレイアウト:
                # 7: タイム, 8: 着差, 9: 指数, 10: 通過, 11: 上り, 12: 単勝, 13: 人気, 14: 体重

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

                # 調教師
                trainer_a = cols[18].find('a') # およそのインデックス
                # 調教師リンクを検索
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
                    'birthday': None, # 馬ページが必要
                    'sire_id': None,
                    'mare_id': None
                })

            except Exception as e:
                # logger.warning(f"Error parsing row in {race_id}: {e}")
                continue

        return results, horses

    @staticmethod
    def _parse_payouts(soup: BeautifulSoup, race_id: str) -> List[Dict]:
        """払い戻し情報をパースします。"""
        payouts = []

        # 払い戻しは通常 'pay_block' または 'pay_table_01' クラスのテーブルにあります
        tables = soup.select('table.pay_block, table.pay_table_01')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                th = row.find('th')
                if not th: continue
                ticket_type = th.text.strip() # 単勝, 複勝, etc.

                # 1つの行に複数の当選番号/払い戻しがある場合があります（例: 複勝）
                # netkeibaでは通常 <br> で区切られています。

                cols = row.find_all('td')
                if len(cols) < 2: continue

                nums_cell = cols[0]
                pays_cell = cols[1]

                # テキスト行を抽出する関数
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
