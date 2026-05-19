@echo off
REM ============================================================
REM  deploy_modal.bat — Deploy webtoon-cleaner to Modal GPU cloud
REM ============================================================
REM  Run this from the project root:
REM      deploy_modal.bat
REM ============================================================

echo === Webtoon Cleaner — Modal GPU Deploy ===
echo.

REM Set the Modal token (already saved to ~/.modal.toml, so this is optional
REM but included here as a fallback for CI/CD environments).
REM py -3.12 -m modal token set --token-id ak-2dZDHul9PILgd8STYus2ys --token-secret as-8yRxcUJ6I686lWVmMk3iXv

REM Deploy to Modal
echo Deploying backend\modal_backend.py to Modal...
cd /d "%~dp0backend"
set PYTHONUTF8=1
py -3.12 -m modal deploy modal_backend.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Deploy successful! ===
    echo.
    echo Your GPU-powered backend is now live on Modal.
    echo Copy the URL shown above and paste it into the frontend as NEXT_PUBLIC_BACKEND_URL.
) else (
    echo.
    echo === Deploy FAILED. Check the output above for errors. ===
)

pause
