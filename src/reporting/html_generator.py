
import os
import pandas as pd
import datetime

class HTMLReportGenerator:
    def __init__(self, output_dir='reports/weekly'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate(self, bets_df, bankroll, race_data=None, date_str=None):
        """
        bets_df: Evaluated bets DataFrame.
        race_data: Full race DataFrame (df_pred) containing all horses/odds/ranks.
        date_str: Target date (YYYY-MM-DD).
        """
        if date_str:
            today = date_str
        else:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # HTML Template
        html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Keiba AI Report ({today})</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f4f6f9; color: #333; }}
                .container {{ max-width: 950px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                .summary {{ background: #e8f0fe; padding: 20px; border-radius: 8px; margin-bottom: 30px; display: flex; justify-content: space-around; }}
                .summary div {{ text-align: center; }}
                .summary strong {{ display: block; font-size: 0.9em; color: #5f6368; }}
                .summary span {{ font-size: 1.4em; font-weight: bold; color: #1a73e8; }}
                
                /* Accordion Style */
                details {{ margin-bottom: 15px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white; }}
                summary {{ cursor: pointer; padding: 15px; background-color: #f8f9fa; font-weight: bold; display: flex; align-items: center; justify-content: space-between; }}
                summary:hover {{ background-color: #f1f3f4; }}
                details[open] summary {{ border-bottom: 1px solid #e0e0e0; }}
                
                .race-content {{ padding: 20px; }}
                
                .tag {{ padding: 3px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 10px; }}
                .tag-venue {{ background: #e6f4ea; color: #137333; }}
                .tag-miss {{ background: #fce8e6; color: #c5221f; }}
                .tag-hit {{ background: #fce8e6; color: #d93025; background: #e6f4ea; color: #1e8e3e; }} /* Fixed typo logic in python code below */
                
                .bet-section {{ background: #fffbe6; padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 5px solid #fbc02d; }}
                .result-msg {{ font-weight: bold; margin-top: 10px; }}
                
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                th {{ background-color: #f8f9fa; text-align: left; padding: 8px; border-bottom: 2px solid #ddd; }}
                td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .rank-1 {{ background-color: #fff9c4; font-weight: bold; }}
                .rank-2 {{ background-color: #f0f0f0; }}
                .rank-3 {{ background-color: #f5f5f5; }}
                .prob-bar {{ height: 4px; background-color: #1a73e8; border-radius: 2px; }}
                .footer {{ text-align: center; margin-top: 40px; color: #888; font-size: 0.8em; border-top: 1px solid #eee; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>♘ Keiba AI Report ({today})</h1>
                
                <div class="summary">
                    <div><strong>勝負レース</strong><span>{len(bets_df)}</span></div>
                    <div><strong>総投資</strong><span>¥{bets_df['cost'].sum():,}</span></div>
                    <div><strong>総回収</strong><span>¥{bets_df['return'].sum():,}</span></div>
                    <div><strong>純利益</strong><span style="color: {'red' if bets_df['return'].sum() - bets_df['cost'].sum() > 0 else 'blue'}">¥{bets_df['return'].sum() - bets_df['cost'].sum():,}</span></div>
                </div>
        """
        
        if bets_df.empty:
            html += "<p style='text-align:center;'>本日の推奨レースはありません。</p>"
        else:
            # Sort by Time or Race ID
            bets_df = bets_df.sort_values('race_id')
            
            for _, row in bets_df.iterrows():
                rid = str(row['race_id'])
                # Metadata
                venue = row.get('venue', 'Unknown')
                race_num = row.get('race_number', '??')
                rname = row.get('title', 'Race')
                
                # Bet Results
                is_hit = row['hit']
                profit = row['profit']
                ret_amt = row['return']
                
                status_class = "tag-hit" if is_hit else "tag-miss"
                status_text = f"WIN (+{profit:,})" if is_hit else f"LOSE ({profit:,})"
                
                summary_line = f"""
                    <div>
                        <span class="tag tag-venue">{venue} {race_num}R</span>
                        {rname}
                    </div>
                    <div class="{status_class} tag">{status_text}</div>
                """
                
                # Entry Table Construction
                entry_table_html = ""
                if race_data is not None:
                    # Filter for this race
                    # race_data might have 'race_id' as int or str, ensure str matching
                    race_entries = race_data[race_data['race_id'].astype(str) == rid].sort_values('horse_number')
                    
                    if not race_entries.empty:
                        rows_html = ""
                        for _, horse in race_entries.iterrows():
                            # Format check
                            rank = horse.get('rank')
                            # Rank formatting
                            tr_class = ""
                            rank_str = "-"
                            if pd.notnull(rank):
                                r = int(rank)
                                rank_str = str(r)
                                if r == 1: tr_class = "rank-1"
                                elif r == 2: tr_class = "rank-2"
                                elif r == 3: tr_class = "rank-3"
                            
                            # Prob bar
                            prob = horse.get('calibrated_prob', 0)
                            prob_pct = f"{prob*100:.1f}%"
                            
                            rows_html += f"""
                            <tr class="{tr_class}">
                                <td>{int(horse['horse_number'])}</td>
                                <td>{horse['horse_name']}</td>
                                <td>{horse.get('odds', '-')}</td>
                                <td>{horse.get('popularity', '-')}</td>
                                <td>
                                    {prob_pct}
                                    <div class="prob-bar" style="width: {min(100, prob*500)}px;"></div>
                                </td>
                                <td>{rank_str}</td>
                            </tr>
                            """
                        
                        entry_table_html = f"""
                        <h4 style="margin-top:20px; margin-bottom:10px;">出馬表 & 結果</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th width="50">番</th>
                                    <th>馬名</th>
                                    <th width="60">Odds</th>
                                    <th width="40">人</th>
                                    <th width="100">AI確率</th>
                                    <th width="40">着</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows_html}
                            </tbody>
                        </table>
                        """

                # Bet Details
                eyes_display = row['eyes'].replace('\n', '<br>')
                
                html += f"""
                <details>
                    <summary>{summary_line}</summary>
                    <div class="race-content">
                        <div class="bet-section">
                            <div style="display:flex; justify-content:space-between;">
                                <div>
                                    <strong>推奨買い目 ({row['bet_type']})</strong>
                                    <p style="margin:5px 0 0 0; font-family:monospace;">{eyes_display}</p>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:0.9em; color:#666;">投資額</div>
                                    <div style="font-weight:bold; font-size:1.1em;">¥{row['cost']:,}</div>
                                    <br>
                                    <div style="font-size:0.9em; color:#666;">払戻額</div>
                                    <div style="font-weight:bold; font-size:1.2em; color:{'#1e8e3e' if ret_amt > 0 else '#333'}">¥{ret_amt:,}</div>
                                </div>
                            </div>
                        </div>
                        {entry_table_html}
                    </div>
                </details>
                """
        
        html += """
                <div class="footer">
                    Generated by Keiba AI Operation System
                </div>
            </div>
        </body>
        </html>
        """
        
        filename = f"report_{today.replace('-', '')}.html"
        save_path = os.path.join(self.output_dir, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Report generated: {save_path}")
        return save_path
