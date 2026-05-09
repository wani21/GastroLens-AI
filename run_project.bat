@echo off
echo Starting GastroLens-AI Project...
echo.

echo Starting Backend API in new terminal...
start "Backend API" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && uvicorn backend.main:app --reload --port 8000"

timeout /t 2 /nobreak > nul

echo Starting Frontend Development Server in new terminal...
start "Frontend Dev Server" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both services are starting up...
echo Backend API will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:5173 (default Vite port)
echo API Documentation: http://localhost:8000/docs
echo.
echo Press any key to close this window (services will continue running)...
pause > nul