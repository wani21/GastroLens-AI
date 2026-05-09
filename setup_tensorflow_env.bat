@echo off
echo Setting up TensorFlow environment for Python 3.11...
echo.

REM Check if Python 3.11 is available
python --version 2>nul | findstr "3.11" >nul
if errorlevel 1 (
    echo ERROR: Python 3.11 not found. Please ensure Python 3.11 is installed and in PATH.
    echo You can download it from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python 3.11 found. Creating virtual environment...

REM Remove existing tf-env if it exists
if exist tf-env (
    echo Removing existing tf-env...
    rmdir /s /q tf-env
)

REM Create new virtual environment
python -m venv tf-env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo Virtual environment created successfully.

REM Activate the environment and install TensorFlow
echo Activating environment and installing TensorFlow...
call tf-env\Scripts\activate.bat

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install TensorFlow (CPU version - change to tensorflow-gpu if you have CUDA-compatible GPU)
pip install tensorflow==2.18.0

REM Install other common ML dependencies
pip install numpy scipy matplotlib scikit-learn pillow opencv-python

REM Verify installation
echo.
echo Verifying TensorFlow installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"

echo.
echo TensorFlow environment setup complete!
echo.
echo To activate this environment in the future, run:
echo   call tf-env\Scripts\activate.bat
echo.
echo To deactivate, run: deactivate
echo.
pause