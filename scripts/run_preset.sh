#!/bin/bash

# Convenience script to run training with a specific preset
# Usage: ./scripts/run_preset.sh <preset-name> <training-name> [additional-args...]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <preset-name> <training-name> [additional-args...]"
    echo "Example: $0 simple my-training"
    echo "Example: $0 complex my-training --batch-size 2 --total-images 2000"
    exit 1
fi

PRESET_NAME="$1"
TRAINING_NAME="$2"
shift 2  # Remove the first two arguments, leaving any additional args

# Run the training script with preset mode
cd "$(dirname "$0")/.." && ./run.sh --preset "$PRESET_NAME" "$TRAINING_NAME" "$@"