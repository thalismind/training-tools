@echo off
setlocal enabledelayedexpansion

REM Convenience script to run training in wizard mode (Windows)
REM Usage: scripts\run_wizard.bat <training-name>

if "%~1"=="" (
    echo Usage: %0 ^<training-name^>
    echo Example: %0 my-training
    exit /b 1
)

set TRAINING_NAME=%~1

REM Run the training script in wizard mode
cd /d "%~dp0.." && run.bat --wizard "%TRAINING_NAME%"