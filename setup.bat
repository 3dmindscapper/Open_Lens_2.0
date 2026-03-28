@echo off
setlocal enabledelayedexpansion
title OpenLens 2.0 — First-time Setup
color 0A

echo.
echo ============================================================
echo   OpenLens 2.0 — One-time Setup
echo ============================================================
echo.

REM ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python 3.11 from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Tick "Add Python to PATH" during install.
    echo Then re-run this setup.bat
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

REM ── Create virtual environment ────────────────────────────────
echo.
echo [1/5] Creating virtual environment...
if exist venv (
    echo       venv already exists, skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Done.
)

REM ── Activate venv ────────────────────────────────────────────
call venv\Scripts\activate.bat

REM ── Upgrade pip ──────────────────────────────────────────────
echo.
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ── Detect CUDA ──────────────────────────────────────────────
echo.
echo [3/5] Detecting hardware...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo       NVIDIA GPU detected — installing PyTorch with CUDA 12.1 support.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo       No NVIDIA GPU found — installing CPU-only PyTorch.
    echo       Note: OCR will be slower ^(~20-60s per page^).
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
)

REM ── Install dependencies ──────────────────────────────────────
echo.
echo [4/5] Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo       Done.
echo.
echo   Note: Flash Attention 2 is not available on Windows.
echo   Using PyTorch SDPA backend instead ^(still fast on CUDA^).
echo   For maximum speed, deploy on Linux — see setup_server.sh

REM ── Download dots.ocr model ───────────────────────────────────
echo.
echo [5/5] Downloading dots.ocr model weights (~2-4 GB)...
echo       This only happens once. Please wait...
python download_model.py
if errorlevel 1 (
    echo [ERROR] Model download failed. Check your internet connection and try again.
    pause
    exit /b 1
)

REM ── Create pipeline __init__ if missing ───────────────────────
if not exist pipeline mkdir pipeline
if not exist pipeline\__init__.py echo. > pipeline\__init__.py

REM ── Done ──────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   To start the app, double-click run.bat
echo   Then open http://localhost:7860 in your browser.
echo.
pause
