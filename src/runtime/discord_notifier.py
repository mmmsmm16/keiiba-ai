import requests
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def send_prediction(self, 
                        race_meta: Dict, 
                        top_horses: List[Dict], 
                        bets: Dict, # from StrategyEngine.decide_bets
                        odds_status: str = "OK"
                       ):
        """
        Send formatted prediction message to Discord.
        """
        if not self.webhook_url:
            logger.warning("No Webhook URL provided. Skipping notification.")
            return

        # Header
        rid = race_meta.get('race_id')
        name = race_meta.get('race_name', 'Unknown Race')
        time_str = race_meta.get('start_time', '??:??')
        place = race_meta.get('place', '')
        
        # Color: Green if any BUY, Blue if SKIP, Red if Error
        final_bets = bets.get('final_bets', [])
        color = 0x00ff00 if final_bets else 0x3498db
        if odds_status != "OK":
            color = 0xe74c3c # Red
            
        embed = {
            "title": f"🏁 {place} {name} ({time_str})",
            "description": f"Race ID: {rid} | Odds: {odds_status}",
            "color": color,
            "fields": []
        }
        
        # Field 1: Top 5 Prediction
        lines = []
        for i, h in enumerate(top_horses[:5]):
            rank = i + 1
            name = h.get('horse_name', f"H{h['horse_number']}")
            num = h.get('horse_number')
            p_cal = h.get('p_cal', 0.0)
            # Add odds if available (passed in h?)
            # Assuming h has 'odds_win' added by fetcher logic outside
            o_str = f"({h['odds_win']:.1f})" if 'odds_win' in h else ""
            
            mark = "◎" if rank==1 else "○" if rank==2 else "▲" if rank==3 else "△"
            lines.append(f"`{rank}.` {mark} **{num:02} {name}** {o_str} (P:{p_cal:.1%})")
            
        embed["fields"].append({
            "name": "🐴 予想TOP5 (推奨順)",
            "value": "\n".join(lines),
            "inline": False
        })
        
        # Field 2: Strategy Decisions
        decisions = bets.get('decisions', [])
        buy_lines = []
        skip_lines = []
        
        for d in decisions:
            strat_name = d['strategy'].replace('_', ' ').title()
            if d['buy']:
                # Format: "Win Core: #05 (200円)"
                targets = ",".join([str(t) for t in d['target']])
                buy_lines.append(f"✅ **{strat_name}**: {targets} ({d['amount']}円)\n   └ {d.get('details','')}")
            else:
                skip_lines.append(f"❌ **{strat_name}**: {d.get('reason','')}")
                
        if buy_lines:
            embed["fields"].append({
                "name": "💰 推奨買い目 (BUY)",
                "value": "\n".join(buy_lines),
                "inline": False
            })
            
        if skip_lines:
             embed["fields"].append({
                "name": "👀 見送り理由 (SKIP)",
                "value": "\n".join(skip_lines),
                "inline": False
            })

        # Send
        payload = {
            "username": "AI Prediction Agent (v13)",
            "embeds": [embed]
        }
        
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            if resp.status_code not in [200, 204]:
                logger.error(f"Discord send failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Discord send exception: {e}")

