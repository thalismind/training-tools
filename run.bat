@echo off
setlocal enabledelayedexpansion

REM Run script for training tools (Windows)
REM This script activates the virtual environment and runs the training command
REM Usage: run.bat [training-name] [--preset preset-name] [other-args...]

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment 'venv' not found. Please run setup.bat first. >&2
    exit /b 1
)

REM Check if train.py exists
if not exist train.py (
    echo train.py not found in current directory. >&2
    exit /b 1
)

REM Activate virtual environment (silently)
call venv\Scripts\activate.bat >nul 2>&1

REM Run the training script with all arguments passed through
REM The script will output only the training command as requested
python train.py %*