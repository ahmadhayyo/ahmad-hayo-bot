@echo off
title Ultimate Agent — Starting...
color 0A

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   Ultimate Secure Local OS Executive Agent       ║
echo  ║   Starting server — please wait...               ║
echo  ╚══════════════════════════════════════════════════╝
echo.

:: Move to the project folder
cd /d "C:\HAYO AI AGENT"

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Check if activation succeeded
if errorlevel 1 (
    echo  [ERROR] Could not activate virtual environment.
    echo  Make sure 'venv' folder exists inside C:\HAYO AI AGENT
    pause
    exit /b 1
)

echo  [OK] Virtual environment activated.
echo.

:: Install / update dependencies silently (only if needed)
echo  [..] Checking dependencies...
pip install langgraph-checkpoint-sqlite aiosqlite duckduckgo-search --quiet --exists-action i >nul 2>&1
echo  [OK] Dependencies up to date.
echo.

:: Launch Chainlit — it will open the browser automatically
echo  [OK] Launching agent UI...
echo  [>>] Opening http://localhost:8000 in your browser.
echo.
echo  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo  Press CTRL+C to stop the server when you are done.
echo  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

chainlit run app.py --port 8000

:: If chainlit exits, pause so user can read any error messages
echo.
echo  [!] Server stopped.
pause
