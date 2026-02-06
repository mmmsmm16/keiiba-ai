@echo off
cd /d %~dp0..
docker compose exec -T app python scripts/auto_predict_ev.py --paper-trade --notify --threshold 1.5 --min_prob 0.15 --min_odds 2.0 --fraction 0.05
