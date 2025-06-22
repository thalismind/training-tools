#!/bin/bash

# Setup script for training tools
# This script creates a virtual environment, installs dependencies, and runs the training script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is available
check_python() {
    print_status "Checking Python installation..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python 3 found: $PYTHON_VERSION"
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        if [[ $PYTHON_VERSION == 3* ]]; then
            print_success "Python 3 found: $PYTHON_VERSION"
            PYTHON_CMD="python"
        else
            print_error "Python 3 is required, but found Python $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."

    if [ -d "venv" ]; then
        print_warning "Virtual environment 'venv' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf venv
        else
            print_status "Using existing virtual environment"
            return
        fi
    fi

    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created successfully"
}

# Activate virtual environment and install requirements
install_requirements() {
    print_status "Activating virtual environment and installing requirements..."

    # Source the virtual environment
    source venv/bin/activate

    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing core requirements..."
        pip install -r requirements.txt
        print_success "Core requirements installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi

    # Ask if user wants to install development requirements
    read -p "Do you want to install development requirements? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "requirements-dev.txt" ]; then
            print_status "Installing development requirements..."
            pip install -r requirements-dev.txt
            print_success "Development requirements installed successfully"
        else
            print_warning "requirements-dev.txt not found, skipping development requirements"
        fi
    fi
}

# Check if config directory exists and has YAML files
check_config() {
    print_status "Checking configuration files..."

    if [ ! -d "config" ]; then
        print_warning "Config directory not found, creating it..."
        mkdir -p config
        print_status "Please add your YAML configuration files to the config/ directory"
        return 1
    fi

    # Check for YAML files
    YAML_FILES=$(find config -name "*.yaml" -o -name "*.yml" 2>/dev/null | wc -l)
    if [ "$YAML_FILES" -eq 0 ]; then
        print_warning "No YAML configuration files found in config/ directory"
        print_status "Please add your YAML configuration files to the config/ directory"
        return 1
    else
        print_success "Found $YAML_FILES YAML configuration file(s)"
        return 0
    fi
}

# Create sample configuration if none exists
create_sample_config() {
    print_status "Creating sample configuration..."

    mkdir -p config

    cat > config/sample.yaml << 'EOF'
presets:
  - name: "default"
    metadata:
      category: "general"
      description: "Default training configuration"
      stable: true
      version: "1.0"
    parameters:
      model_family: "flux"
      learning_rate: 0.0001
      min_snr_gamma: 5.0
      noise_offset: 0.0
      save_every_n_steps: 100
      sample_every_n_steps: 500
      timestep_sampling: "sigmoid"
      network:
        alpha: 32
        dim: 64
        train_t5xxl: false
        split_qkv: false
      optimizer:
        name: "adamw8bit"
        args:
          weight_decay: 0.01
          betas: [0.9, 0.999]
      scheduler:
        name: "cosine"
        cycles: 1
      total_images: 1000
      batch_size: 1
      network_train_unet_only: false

  - name: "simple"
    metadata:
      category: "general"
      description: "Simple training configuration for basic concepts"
      stable: true
      version: "1.0"
    inherits: ["default"]
    parameters:
      total_images: 500
      save_every_n_steps: 50
      sample_every_n_steps: 250

  - name: "complex"
    metadata:
      category: "general"
      description: "Complex training configuration for detailed concepts"
      stable: true
      version: "1.0"
    inherits: ["default"]
    parameters:
      total_images: 2000
      save_every_n_steps: 200
      sample_every_n_steps: 1000
      network:
        alpha: 64
        dim: 128
EOF

    print_success "Sample configuration created at config/sample.yaml"
}

# Run the training script
run_training_script() {
    print_status "Running training script..."

    # Check if we have configuration files
    if ! check_config; then
        print_warning "No configuration files found. Creating sample configuration..."
        create_sample_config
    fi

    # Get training name from user or use default
    read -p "Enter training name (or press Enter for 'test-training'): " TRAINING_NAME
    TRAINING_NAME=${TRAINING_NAME:-test-training}

    print_status "Running training script with name: $TRAINING_NAME"

    # Run the script with wizard mode
    source venv/bin/activate
    python train.py --wizard "$TRAINING_NAME"
}

# Main execution
main() {
    echo "=========================================="
    echo "Training Tools Setup Script"
    echo "=========================================="
    echo

    # Check Python installation
    check_python

    # Create virtual environment
    create_venv

    # Install requirements
    install_requirements

    # Run the training script
    run_training_script

    echo
    echo "=========================================="
    print_success "Setup completed successfully!"
    echo "=========================================="
    echo
    echo "To run the training script again:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the script: python train.py --wizard <training-name>"
    echo
    echo "Or use a specific preset:"
    echo "python train.py --preset <preset-name> <training-name>"
}

# Run main function
main "$@"