@echo off
echo Starting Osmix Platform...
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Warning: Virtual environment not found. Please create it first.
    pause
    exit /b 1
)

echo.
echo Starting Backend API...
cd backend
start "Osmix Backend" cmd /k "python all.py"
cd ..

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend...
cd frontend
start "Osmix Frontend" cmd /k "npm start"

echo.
echo Both services are starting...
echo Backend API: http://localhost:8000
echo Frontend: Electron app will open shortly
echo.
pause
