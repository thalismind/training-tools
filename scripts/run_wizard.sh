#!/bin/bash

# Convenience script to run training in wizard mode
# Usage: ./scripts/run_wizard.sh <training-name>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <training-name>"
    echo "Example: $0 my-training"
    exit 1
fi

TRAINING_NAME="$1"

# Run the training script in wizard mode
cd "$(dirname "$0")/.." && ./run.sh --wizard "$TRAINING_NAME"