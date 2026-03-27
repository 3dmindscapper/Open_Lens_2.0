@echo off
title OpenLens 2.0
color 0A

echo.
echo ============================================================
echo   OpenLens 2.0 — Starting...
echo ============================================================
echo.

if not exist venv (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo   Opening browser at http://localhost:7860
echo   Press Ctrl+C in this window to stop the app.
echo.

venv\Scripts\python.exe app.py

pause
