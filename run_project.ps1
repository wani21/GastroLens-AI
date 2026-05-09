# GastroLens-AI Project Runner
# This script starts both the backend API and frontend development server

Write-Host "Starting GastroLens-AI Project..." -ForegroundColor Green

# Set execution policy for the session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

# Function to start backend
function Start-Backend {
    Write-Host "Starting Backend API..." -ForegroundColor Yellow
    # Activate virtual environment (assuming .venv contains backend dependencies)
    & ".\.venv\Scripts\Activate.ps1"
    # Start uvicorn server
    uvicorn backend.main:app --reload --port 8000
}

# Function to start frontend
function Start-Frontend {
    Write-Host "Starting Frontend Development Server..." -ForegroundColor Yellow
    Set-Location ".\frontend"
    npm run dev
}

# Start both services in parallel using jobs
$backendJob = Start-Job -ScriptBlock ${function:Start-Backend}
$frontendJob = Start-Job -ScriptBlock ${function:Start-Frontend}

Write-Host "Both services are starting up..." -ForegroundColor Cyan
Write-Host "Backend API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend will be available at: http://localhost:5173 (default Vite port)" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan

# Wait for jobs to complete (they run indefinitely)
Wait-Job -Job $backendJob, $frontendJob

# Clean up
Remove-Job -Job $backendJob, $frontendJob