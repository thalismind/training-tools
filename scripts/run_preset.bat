@echo off
setlocal enabledelayedexpansion

REM Convenience script to run training with a specific preset (Windows)
REM Usage: scripts\run_preset.bat <preset-name> <training-name> [additional-args...]

if "%~2"=="" (
    echo Usage: %0 ^<preset-name^> ^<training-name^> [additional-args...]
    echo Example: %0 simple my-training
    echo Example: %0 complex my-training --batch-size 2 --total-images 2000
    exit /b 1
)

set PRESET_NAME=%~1
set TRAINING_NAME=%~2
shift
shift

REM Run the training script with preset mode
cd /d "%~dp0.." && run.bat --preset "%PRESET_NAME%" "%TRAINING_NAME%" %*