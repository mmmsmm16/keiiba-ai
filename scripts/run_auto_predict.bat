@echo off
REM Auto Predict v13 - 自動予測スクリプト
REM Windows Task Scheduler から呼び出すバッチファイル
REM 設定: 土日 9:00-17:00 に1分ごとに実行

cd /d C:\Users\masat\MyLab\Dev\keiiba-ai

REM Docker経由で実行
docker compose exec -T app python src/scripts/auto_predict_v13.py

REM ログ出力
echo [%date% %time%] Auto Predict v13 executed >> logs\auto_predict_batch.log
