@echo off
chcp 65001 >nul 2>&1
title HAYO AI Agent — Starting...
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║     🤖  HAYO AI Agent — وكيل ذكي خارق القدرات            ║
echo  ║                                                          ║
echo  ║     Starting server — please wait...                     ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Move to the project folder (works from any location)
cd /d "%~dp0"

:: Check if venv exists, create if not
if not exist "venv\Scripts\activate.bat" (
    echo  [..] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        echo  Make sure Python 3.10+ is installed and in PATH.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

:: Activate the virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo  [OK] Virtual environment activated.
echo.

:: Install / update dependencies
echo  [..] Checking dependencies...
pip install -r requirements.txt --quiet --exists-action i >nul 2>&1
echo  [OK] Dependencies up to date.
echo.

:: Install Playwright browsers if not already installed
echo  [..] Checking Playwright browsers...
python -m playwright install chromium --with-deps >nul 2>&1
echo  [OK] Browser ready.
echo.

:: Check .env file exists
if not exist ".env" (
    echo  [WARNING] .env file not found!
    echo  Creating a template .env file...
    echo MODEL_PROVIDER=google> .env
    echo GOOGLE_API_KEY=YOUR_KEY_HERE>> .env
    echo ANTHROPIC_API_KEY=>> .env
    echo OPENAI_API_KEY=>> .env
    echo DEEPSEEK_API_KEY=>> .env
    echo.
    echo  [!] Please edit .env and add your API keys.
    echo.
)

:: Launch Chainlit
echo  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo  🌐 Opening http://localhost:8000
echo  📋 Model: Check .env for MODEL_PROVIDER setting
echo.
echo  Press CTRL+C to stop the server.
echo  Or double-click STOP.bat to stop from another window.
echo.
echo  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

:: --headless prevents chainlit from auto-opening browser; we open one window ourselves
start "" http://localhost:8000
chainlit run app.py --port 8000 --headless

echo.
echo  [!] Server stopped.
pause
