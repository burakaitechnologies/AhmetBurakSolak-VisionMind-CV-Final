@echo off
REM ============================================================
REM   VisionMind — Live Demo Launcher
REM   ITAI 1378 Final Project — Ahmet Burak Solak
REM ============================================================
REM   Just double-click this file. It will:
REM     1) check Python is installed
REM     2) install the demo's Python dependencies (first run only)
REM     3) run demo.py — pop up a window per image, save stills
REM        into demo_outputs/ for your video editor.
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo    VisionMind - ITAI 1378 Final Project Demo
echo    Author: Ahmet Burak Solak
echo ============================================================
echo.

REM -- Resolve a Python launcher (py preferred, then python) --
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "PY=python"
    ) else (
        echo [ERROR] Python is not installed or not on PATH.
        echo         Install Python 3.10+ from https://www.python.org/downloads/
        echo         then re-run this script.
        pause
        exit /b 1
    )
)

echo Using Python: %PY%
%PY% --version
echo.

REM -- Install lightweight demo dependencies (idempotent) --
echo Step 1/2 - Checking / installing dependencies (first run only) ...
%PY% -m pip install --quiet --upgrade pip
%PY% -m pip install --quiet torch torchvision pillow matplotlib numpy
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Dependency install failed. Try running:
    echo     %PY% -m pip install torch torchvision pillow matplotlib numpy
    pause
    exit /b 1
)
echo   dependencies ready.
echo.

REM -- Run the demo --
echo Step 2/2 - Launching demo ...
echo.
%PY% demo.py
set EXIT=%ERRORLEVEL%

echo.
if %EXIT% NEQ 0 (
    echo [ERROR] demo.py exited with code %EXIT%.
) else (
    echo Demo finished. Look in demo_outputs\ for the saved stills.
)
echo.
pause
endlocal
exit /b %EXIT%
