
import os
import requests
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import entropy
from typing import List, Dict, Optional, Any

from src.betting.ticket import Ticket

logger = logging.getLogger(__name__)

class NotificationManager:
    """Discord Notification Manager."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def _calculate_confidence(self, df: pd.DataFrame) -> tuple[str, str]:
        """Estimate confidence based on score/prob distribution."""
        if 'prob' not in df.columns:
            return "B", "不明"
            
        probs = df['prob'].values 
        ent = entropy(probs)
        
        # Sort by prob or score
        if 'score' in df.columns:
            top_horse = df.sort_values('score', ascending=False).iloc[0]
        else:
            top_horse = df.sort_values('prob', ascending=False).iloc[0]
            
        top_prob = top_horse['prob']
        
        if top_prob >= 0.40:
            return "S", "鉄板"
        elif top_prob >= 0.25:
            return "A", "安定"
        elif ent > 2.0 or top_prob < 0.20:
             return "C", "波乱"
        else:
             return "B", "混戦"

    def send_tickets(self, race_meta: Dict, df: pd.DataFrame, tickets: List[Ticket]) -> bool:
        """
        Send prediction summary and generated tickets to Discord.
        
        Args:
            race_meta: Dict with {'race_id', 'venue_name', 'race_number', 'title', 'start_time', 'date'}
            df: DataFrame of model predictions (horse_number, horse_name, prob, score).
            tickets: List of generated Ticket objects.
        """
        if not self.webhook_url:
            return False

        chart_rank, chart_desc = self._calculate_confidence(df)
        date_str = race_meta.get('date', '')
        
        title_str = f"🎯 [{date_str}] {race_meta.get('venue_name','')}{race_meta.get('race_number','')}R {race_meta.get('title','')} ({race_meta.get('start_time','')}) - [{chart_rank}] {chart_desc}"
        
        # 1. Prediction Table
        df_sorted = df.sort_values('prob', ascending=False) # or score
        description = "**🏆 AI予測モデル (Top 5)**\n"
        
        for i in range(min(5, len(df_sorted))):
            row = df_sorted.iloc[i]
            h_num = str(int(row['horse_number'])).zfill(2)
            h_name = row.get('horse_name', f'Horse{h_num}')
            prob = row.get('prob', 0)
            score = row.get('score', 0)
            odds = row.get('odds', 0)
            odds_str = f"{odds:.1f}" if odds > 0 else "-"
            
            description += f"`{h_num}` **{h_name}** (勝率:{prob*100:.0f}%, オッズ:{odds_str})\n"

        # 2. Tickets
        bet_msg = "\n**📈 自動生成買い目**\n"
        
        if not tickets:
             bet_msg += "⚠️ 買い目なし (期待値不足)\n"
        else:
            # Group by Bet Type
            # Sort: Win, Place, Umaren, Wide...
            order = ['win', 'place', 'umaren', 'wide', 'umatan', 'sanrenpuku', 'sanrentan']
            
            # Map type to Japanese
            type_jp = {
                'win': '単勝', 'place': '複勝', 'umaren': '馬連', 
                'wide': 'ワイド', 'umatan': '馬単', 
                'sanrenpuku': '三連複', 'sanrentan': '三連単'
            }
            
            grouped_tickets = {}
            for t in tickets:
                grouped_tickets.setdefault(t.bet_type, []).append(t)
                
            for btype in order:
                if btype not in grouped_tickets: continue
                ts = grouped_tickets[btype]
                
                # Check odds type
                # Usually consistent per type, but let's check first ticket
                odds_mark = "⚡" if ts[0].odds_type == 'estimated' else "🏢"
                odds_text = "推定オッズ" if ts[0].odds_type == 'estimated' else "実オッズ"
                
                bet_msg += f"**{type_jp.get(btype, btype)}** ({odds_mark} {odds_text})\n"
                
                # Compact format
                lines = []
                for t in ts:
                    # combination, odds, stake
                    c_str = t.combination_str
                    o_str = f"{t.odds:.1f}" if t.odds else "N/A"
                    ev_str = f"EV:{t.expected_value:.2f}" if t.odds_type=='real' else f"Sc:{t.selection_score:.2f}"
                    lines.append(f"`{c_str}` (@{o_str} {t.stake}円) {ev_str}")
                
                bet_msg += "\n".join(lines) + "\n"

        # Link
        rid = race_meta.get('race_id', '')
        netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        description += f"\n🔗 [NetKeiba]({netkeiba_url})\n"
        
        embed = {
            "title": title_str,
            "description": description + bet_msg,
            "color": 0x00FF00 if chart_rank in ['S', 'A'] else 0xFFA500,
            "footer": {
                "text": "Keiiba-AI Unified (Real+Synthetic)"
            }
        }
        
        payload = {"username": "Antigravity AI", "embeds": [embed]}
        
        try:
            resp = requests.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            logger.info(f"Sent notification for {rid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    def send_jit_report(self, race_meta: Dict, df: pd.DataFrame) -> bool:
        """
        Send JIT Prediction Report (Mobile Optimized).
        Highlights Recommended Horses based on ROI logic.
        """
        if not self.webhook_url:
            return False

        # Sort by score/prob
        sort_col = 'pred_prob' if 'pred_prob' in df.columns else 'prob'
        df_sorted = df.sort_values(sort_col, ascending=False).head(5)
        
        date_str = race_meta.get('date', '')
        place = race_meta.get('venue_name', '')
        rnum = race_meta.get('race_number', '')
        title = f"🏇 {place} {rnum}R Prediction ({date_str})"
        
        # Build description
        desc_lines = []
        rec_lines = []
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            rank = i + 1
            h_name = row.get('horse_name', 'Unknown')
            h_num = int(row.get('horse_number', 0))
            prob = row.get(sort_col, 0)
            odds = row.get('odds_10min', row.get('odds', 0))
            odds_str = f"{odds:.1f}" if pd.notna(odds) and odds > 0 else "-"
            
            # Recommendation Logic
            # 1. Conf >= 0.45 & Odds >= 2.0
            # 2. Conf >= 0.50 & Odds >= 1.5
            is_rec = False
            rec_type = ""
            
            if pd.notna(odds) and odds > 0:
                if prob >= 0.50 and odds >= 1.5:
                    is_rec = True
                    rec_type = "🔥鉄板"
                elif prob >= 0.45 and odds >= 2.0:
                    is_rec = True
                    rec_type = "🎯妙味"
            
            # Formatting
            # Mobile friendly: Name (Num) \n Prob | Odds
            # Or one line: [1] Name(05) 45% 3.2倍
            
            mark = "👑" if rank == 1 else f"{rank}."
            
            line = f"**{mark} {h_name}** (`{h_num:02}`)\n"
            line += f"📊 Conf:**{prob:.0%}**  🏢 Odds:**{odds_str}**"
            
            if is_rec:
                line += f"  {rec_type} **Buy!**"
                rec_lines.append(f"{h_name}({h_num:02}) - {rec_type} (Conf:{prob:.0%}, Odds:{odds_str})")
            
            desc_lines.append(line)

        # Construct Embed
        embed_desc = "\n\n".join(desc_lines)
        
        # Color
        color = 0x00FF00 if rec_lines else 0xCCCCCC # Green if recommendation exists
        
        embed = {
            "title": title,
            "description": embed_desc,
            "color": color,
            "fields": []
        }
        
        # Add Recommendation Field if any
        if rec_lines:
            embed["fields"].append({
                "name": "🔥 推奨買い目 (単勝)",
                "value": "\n".join([f"• {r}" for r in rec_lines]),
                "inline": False
            })
            
        # Add Link
        rid = race_meta.get('race_id', '')
        if rid:
             netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
             embed["url"] = netkeiba_url

        payload = {"username": "Keiiba-AI JIT", "embeds": [embed]}
        
        try:
            resp = requests.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            logger.info(f"Sent JIT notification for {rid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
