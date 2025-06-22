#!/bin/bash

# Run script for training tools
# This script activates the virtual environment and runs the training command
# Usage: ./run.sh [training-name] [--preset preset-name] [other-args...]

set -e  # Exit on any error

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment 'venv' not found. Please run setup.sh first." >&2
    exit 1
fi

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo "train.py not found in current directory." >&2
    exit 1
fi

# Activate virtual environment (silently)
source venv/bin/activate >/dev/null 2>&1

# Run the training script with all arguments passed through
# The script will output only the training command as requested
python train.py "$@"