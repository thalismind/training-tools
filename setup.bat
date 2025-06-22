@echo off
setlocal enabledelayedexpansion

REM Setup script for training tools (Windows)
REM This script creates a virtual environment, installs dependencies, and runs the training script

echo ==========================================
echo Training Tools Setup Script (Windows)
echo ==========================================
echo.

REM Check if Python is available
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python found: %PYTHON_VERSION%

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment 'venv' already exists
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q venv
    ) else (
        echo [INFO] Using existing virtual environment
        goto :install_requirements
    )
)

python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created successfully

:install_requirements
REM Activate virtual environment and install requirements
echo [INFO] Activating virtual environment and installing requirements...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist requirements.txt (
    echo [INFO] Installing core requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install core requirements
        pause
        exit /b 1
    )
    echo [SUCCESS] Core requirements installed successfully
) else (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

REM Ask if user wants to install development requirements
set /p INSTALL_DEV="Do you want to install development requirements? (y/N): "
if /i "!INSTALL_DEV!"=="y" (
    if exist requirements-dev.txt (
        echo [INFO] Installing development requirements...
        pip install -r requirements-dev.txt
        echo [SUCCESS] Development requirements installed successfully
    ) else (
        echo [WARNING] requirements-dev.txt not found, skipping development requirements
    )
)

REM Check if config directory exists and has YAML files
echo [INFO] Checking configuration files...
if not exist config (
    echo [WARNING] Config directory not found, creating it...
    mkdir config
    echo [INFO] Please add your YAML configuration files to the config\ directory
    goto :create_sample_config
)

REM Check for YAML files
set YAML_COUNT=0
for %%f in (config\*.yaml config\*.yml) do set /a YAML_COUNT+=1
if %YAML_COUNT% equ 0 (
    echo [WARNING] No YAML configuration files found in config\ directory
    echo [INFO] Please add your YAML configuration files to the config\ directory
    goto :create_sample_config
) else (
    echo [SUCCESS] Found %YAML_COUNT% YAML configuration file(s)
    goto :run_training_script
)

:create_sample_config
echo [INFO] Creating sample configuration...
mkdir config 2>nul

(
echo presets:
echo   - name: "default"
echo     metadata:
echo       category: "general"
echo       description: "Default training configuration"
echo       stable: true
echo       version: "1.0"
echo     parameters:
echo       model_family: "flux"
echo       learning_rate: 0.0001
echo       min_snr_gamma: 5.0
echo       noise_offset: 0.0
echo       save_every_n_steps: 100
echo       sample_every_n_steps: 500
echo       timestep_sampling: "sigmoid"
echo       network:
echo         alpha: 32
echo         dim: 64
echo         train_t5xxl: false
echo         split_qkv: false
echo       optimizer:
echo         name: "adamw8bit"
echo         args:
echo           weight_decay: 0.01
echo           betas: [0.9, 0.999]
echo       scheduler:
echo         name: "cosine"
echo         cycles: 1
echo       total_images: 1000
echo       batch_size: 1
echo       network_train_unet_only: false
echo.
echo   - name: "simple"
echo     metadata:
echo       category: "general"
echo       description: "Simple training configuration for basic concepts"
echo       stable: true
echo       version: "1.0"
echo     inherits: ["default"]
echo     parameters:
echo       total_images: 500
echo       save_every_n_steps: 50
echo       sample_every_n_steps: 250
echo.
echo   - name: "complex"
echo     metadata:
echo       category: "general"
echo       description: "Complex training configuration for detailed concepts"
echo       stable: true
echo       version: "1.0"
echo     inherits: ["default"]
echo     parameters:
echo       total_images: 2000
echo       save_every_n_steps: 200
echo       sample_every_n_steps: 1000
echo       network:
echo         alpha: 64
echo         dim: 128
) > config\sample.yaml

echo [SUCCESS] Sample configuration created at config\sample.yaml

:run_training_script
REM Run the training script
echo [INFO] Running training script...

REM Get training name from user or use default
set /p TRAINING_NAME="Enter training name (or press Enter for 'test-training'): "
if "!TRAINING_NAME!"=="" set TRAINING_NAME=test-training

echo [INFO] Running training script with name: !TRAINING_NAME!

REM Run the script with wizard mode
call venv\Scripts\activate.bat
python train.py --wizard "!TRAINING_NAME!"

echo.
echo ==========================================
echo [SUCCESS] Setup completed successfully!
echo ==========================================
echo.
echo To run the training script again:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the script: python train.py --wizard ^<training-name^>
echo.
echo Or use a specific preset:
echo python train.py --preset ^<preset-name^> ^<training-name^>
echo.
pause