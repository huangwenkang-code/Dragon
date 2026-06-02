@echo off
chcp 65001 >nul
cd /d D:\K\dragon-engine

echo ========================================
echo   Dragon Engine - Graph Service
echo   venv: D:\K\dragon-engine\venv
echo ========================================

:: Activate virtual environment
call D:\K\dragon-engine\venv\Scripts\activate.bat

echo.
echo Starting graph-service on http://localhost:8000 ...
python -m uvicorn services.graph_service.main:app --host 0.0.0.0 --port 8000

pause
