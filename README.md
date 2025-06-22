# Training Tools

A comprehensive training script for AI model training with support for multiple model families (Flux, SDXL, Pony) and various training configurations.

## Features

- **Multiple Model Families**: Support for Flux, SDXL, and Pony models
- **Interactive Wizard**: Guided setup for training configuration
- **Preset System**: Reusable training configurations with inheritance
- **Type Safety**: Full type annotations and Pydantic validation
- **Flexible Training**: Support for different training duration methods
- **LoRA/LyCORIS Support**: Multiple network types for fine-tuning

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Automated Setup

#### Linux/macOS
```bash
# Clone the repository
git clone <repository-url>
cd training-tools

# Run the setup script
./setup.sh
```

#### Windows
```cmd
# Clone the repository
git clone <repository-url>
cd training-tools

# Run the setup script
setup.bat
```

The setup script will:
1. Check Python installation
2. Create a virtual environment
3. Install dependencies
4. Create sample configuration files
5. Launch the interactive wizard

### Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

## Usage

### Interactive Wizard

The easiest way to get started is using the interactive wizard:

```bash
python train.py --wizard <training-name>
```

The wizard will guide you through:
- Selecting a training category
- Choosing model family (Flux, SDXL, Pony)
- Setting complexity level
- Configuring training duration
- Setting batch size and VRAM mode
- Choosing LoRA or LyCORIS
- Configuring SNR gamma and other parameters

### Using Presets

You can also use predefined presets:

```bash
python train.py --preset <preset-name> <training-name>
```

### Command Line Options

```bash
python train.py [OPTIONS] <training-name>

Options:
  --sources TEXT...           YAML files to load presets from
  --preset TEXT              Name of the preset to use
  --wizard                   Run the wizard to select a preset
  --resume-from TEXT         Resume training from existing weights file
  --model-family [flux|sdxl|pony]  Model family to use
  --lycoris-subtype [loha|lokr]  LyCORIS subtype
  --batch-size INTEGER       Training batch size
  --total-images INTEGER     Total number of images to process
  --max-steps INTEGER        Maximum training steps
  --max-epochs INTEGER       Maximum training epochs
  --vram-mode [highvram|lowvram|none]  VRAM mode
  --network-train-unet-only  Train UNet only
```

## Configuration

Training configurations are defined in YAML files in the `config/` directory. Each configuration file contains presets with training parameters.

### Sample Configuration Structure

```yaml
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
      network:
        alpha: 32
        dim: 64
      optimizer:
        name: "adamw8bit"
        args:
          weight_decay: 0.01
      scheduler:
        name: "cosine"
        cycles: 1
      total_images: 1000
      batch_size: 1
```

### Preset Inheritance

Presets can inherit from other presets:

```yaml
  - name: "simple"
    metadata:
      category: "general"
      description: "Simple training configuration"
    inherits: ["default"]
    parameters:
      total_images: 500
      save_every_n_steps: 50
```

## Model Families

### Flux
- Uses Flux 1.0 model
- Supports FP8 base training
- Optimized for high-quality image generation

### SDXL
- Uses Stable Diffusion XL model
- Standard LoRA training
- Good balance of quality and speed

### Pony
- Uses Pony v6 model
- Specialized for anime/cartoon style
- Optimized for character training

## Network Types

### LoRA
- Standard LoRA training
- Good for most use cases
- Compatible with all model families

### LyCORIS
- Advanced LoRA variants
- **LoHa**: Higher rank adaptation
- **LoKr**: Kronecker product adaptation

## Environment Variables

The script uses several environment variables for configuration:

- `REMOTE_ROOT`: Base workspace directory (default: `/workspace`)
- `FLUX_PATH`: Path to Flux model files (default: `$REMOTE_ROOT/flux`)
- `SDXL_PATH`: Path to SDXL model files (default: `$REMOTE_ROOT/sdxl`)

## Development

### Installing Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Type Checking

```bash
mypy train.py
```

### Code Formatting

```bash
black train.py
isort train.py
```

### Linting

```bash
flake8 train.py
```

## Project Structure

```
training-tools/
├── train.py              # Main training script
├── setup.sh              # Linux/macOS setup script
├── setup.bat             # Windows setup script
├── requirements.txt      # Core dependencies
├── requirements-dev.txt  # Development dependencies
├── README.md            # This file
└── config/              # Configuration directory
    ├── sample.yaml      # Sample configuration
    └── *.yaml           # Your configuration files
```

## Troubleshooting

### Common Issues

1. **Python not found**: Ensure Python 3.8+ is installed and in PATH
2. **Virtual environment issues**: Delete `venv/` directory and run setup again
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **Configuration errors**: Check YAML syntax and Pydantic validation errors

### Getting Help

- Check the error messages for specific issues
- Verify your YAML configuration syntax
- Ensure all required model files are in the correct paths
- Check that environment variables are set correctly

## License

[Add your license information here]