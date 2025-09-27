@echo off
cls

:menu
echo =============================================
echo     ... HPO System - Interactive Control ...
echo =============================================
echo Welcome! Please choose an option to start.
echo.
echo   1. ... System Check (check)
echo   2. ... Run Quick Example (quick)
echo   3. ... Start Full Optimization (full run)
echo   4. ... Install Requirements (install)
echo   5. ... Exit
echo.

set /p choice="Please enter your choice (1-5): "

if not defined choice (
    echo ... Invalid choice. Please try again.
    goto menu
)

if "%choice%"=="1" (
    echo ... Running system check...
    python run_hpo.py --check
) else if "%choice%"=="2" (
    echo ... Running quick example...
    python run_hpo.py --quick
) else if "%choice%"=="3" (
    echo ... Starting full optimization...
    python run_hpo.py
) else if "%choice%"=="4" (
    echo ... Installing requirements...
    if exist install.py (
        python install.py
    ) else (
        echo ... Error: 'install.py' not found.
    )
) else if "%choice%"=="5" (
    echo ... Goodbye!
    exit /b
) else (
    echo ... Invalid choice. Please enter a number between 1 and 5.
)

echo.
pause
cls
goto menu