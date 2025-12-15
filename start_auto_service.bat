@echo off
cd /d %~dp0

echo ===================================================
echo   Keiiba-AI Auto Predict Service Startup
echo   (Supports Auto Notification on PC Boot)
echo ===================================================

echo [1/2] Ensure Docker services are running...
docker-compose up -d

echo [2/2] Starting Auto Predict Loop in background...
REM -d for detached mode (runs in background inside the container)
docker-compose exec -d app python src/scripts/run_scheduler.py

echo.
echo Success! The auto-prediction service is running in the background.
echo You can close this window.
echo.
timeout /t 5
exit
