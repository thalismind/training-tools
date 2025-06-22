# Migration Guide: Old Scripts to Unified Python Script

This guide shows how to migrate from the old bash training scripts to the new unified Python training script.

## Script Mapping

### Flux Model Scripts

#### `train-flux-d2p-snr-adafactor.sh` → `flux_adafactor_simple`
**Old command:**
```bash
./train-flux-d2p-snr-adafactor.sh my_training
```

**New command:**
```bash
python train.py --preset flux_adafactor_simple my_training
```

**Key differences:**
- Old: Used `constant` scheduler by default
- New: Uses `cosine` scheduler with 1 cycle (as requested)
- Old: Hardcoded parameters in script
- New: Configurable via YAML preset
- New: Supports `--max_train_epochs` in addition to `--max_train_steps`
- New: Supports `total_images` parameter for consistent training across datasets

#### `train-flux-d2p-snr-prodigy.sh` → `flux_prodigy_simple`
**Old command:**
```bash
./train-flux-d2p-snr-prodigy.sh my_training
```

**New command:**
```bash
python train.py --preset flux_prodigy_simple my_training
```

#### `train-flux-d2p-snr-ranger.sh` → `flux_ranger_simple`
**Old command:**
```bash
./train-flux-d2p-snr-ranger.sh my_training
```

**New command:**
```bash
python train.py --preset flux_ranger_simple my_training
```

#### `train-flux-d2p-snr-lokr.sh` → `flux_lokr_simple` + `--lycoris-subtype lokr`
**Old command:**
```bash
./train-flux-d2p-snr-lokr.sh my_training
```

**New command:**
```bash
python train.py --preset flux_lokr_simple --lycoris-subtype lokr my_training
```

### SDXL Model Scripts

#### `train-sdxl-prodigy.sh` → `sdxl_prodigy_simple`
**Old command:**
```bash
./train-sdxl-prodigy.sh my_training
```

**New command:**
```bash
python train.py --preset sdxl_prodigy_simple my_training
```

**Key differences:**
- Old: Used `constant` scheduler by default
- New: Uses `cosine` scheduler with 1 cycle
- Old: Used `--max_train_epochs 5`
- New: Configurable via YAML preset
- New: Supports `--max_train_steps` in addition to `--max_train_epochs`
- New: Supports `total_images` parameter for consistent training across datasets

#### `train-sdxl-adafactor.sh` → `sdxl_adafactor_simple`
**Old command:**
```bash
./train-sdxl-adafactor.sh my_training
```

**New command:**
```bash
python train.py --preset sdxl_adafactor_simple my_training
```

### Pony Model Scripts

#### `train-pony-prodigy.sh` → `pony_prodigy_simple`
**Old command:**
```bash
./train-pony-prodigy.sh my_training
```

**New command:**
```bash
python train.py --preset pony_prodigy_simple my_training
```

**Key differences:**
- Old: Used `constant` scheduler by default
- New: Uses `cosine` scheduler with 1 cycle
- Old: Used `safeguard_warmup=True` for Pony
- New: Preserved in YAML configuration
- New: Supports `--max_train_epochs` in addition to `--max_train_steps`
- New: Supports `total_images` parameter for consistent training across datasets

#### `train-pony-adafactor.sh` → `pony_adafactor_simple`
**Old command:**
```bash
./train-pony-adafactor.sh my_training
```

**New command:**
```bash
python train.py --preset pony_adafactor_simple my_training
```

## Parameter Comparison

### Model Family Specific Settings

| Setting | Flux | SDXL | Pony |
|---------|------|------|------|
| Training Script | `flux_train_network.py` | `sdxl_train_network.py` | `sdxl_train_network.py` |
| Model File | `flux1-dev2pro.safetensors` | `sdxl_v1_vae_fix.safetensors` | `pony_v6_base.safetensors` |
| Network Module | `networks.lora_flux` | `networks.lora` | `networks.lora` |
| Precision | `bf16` | `no` (float32) | `no` (float32) |
| Training Duration | `--max_train_steps` OR `--max_train_epochs` | `--max_train_steps` OR `--max_train_epochs` | `--max_train_steps` OR `--max_train_epochs` |
| Special Args | `--fp8_base`, `--discrete_flow_shift` | `--no_half_vae`, `--network_train_unet_only` | `--no_half_vae`, `--network_train_unet_only` |

### Training Duration Options

#### Total Images (New Feature)
```yaml
parameters:
  total_images: 2000  # Process 2000 total images
  batch_size: 2       # With batch size 2 = 1000 steps
```

#### Max Steps
```yaml
parameters:
  max_steps: 1200
```

#### Max Epochs (All Model Families)
```yaml
parameters:
  max_epochs: 5
```

### Optimizer Configurations

#### Adafactor
```yaml
optimizer:
  name: "adafactor"
  args:
    weight_decay: 0.01
```

#### Prodigy
```yaml
optimizer:
  name: "prodigy"
  args:
    decouple: true
    weight_decay: 0.01
    betas: [0.9, 0.999]
    use_bias_correction: false
    safeguard_warmup: false  # true for Pony
    d_coef: 2
```

#### Ranger
```yaml
optimizer:
  name: "ranger"
  args:
    decouple: true
    weight_decay: 0.01
    betas: [0.9, 0.999]
    use_bias_correction: false
    safeguard_warmup: false
    num_iterations: 5
```

## New Features

### 1. Interactive Wizard
Instead of manually editing scripts, use the interactive wizard:
```bash
python train.py --wizard my_training
```

### 2. Resume Training
Resume from existing weights:
```bash
python train.py --preset flux_prodigy_simple --resume-from existing_weights.safetensors my_training
```

### 3. LyCORIS Support
Easy LyCORIS configuration:
```bash
python train.py --preset flux_lokr_simple --lycoris-subtype lokr my_training
```

### 4. YAML Configuration
All parameters are now configurable via YAML files with inheritance support.

### 5. Default Cosine Scheduler
All presets now use cosine scheduler with 1 cycle by default (as requested).

### 6. Total Images Training (New)
Specify total images to process for consistent training across datasets:
```bash
python train.py --preset flux_total_images_example my_training
```

### 7. Batch Size Support (New)
Specify batch size for all model families:
```bash
python train.py --preset flux_adafactor_simple --batch-size 4 my_training
```

### 8. Flexible Training Duration (New)
All model families now support both steps and epochs:
```bash
# Flux with epochs
python train.py --preset flux_adafactor_simple --max-epochs 10 my_training

# SDXL with steps
python train.py --preset sdxl_prodigy_simple --max-steps 2000 my_training
```

## Configuration Files

The new system uses YAML configuration files in the `config/` directory:

- `config/models.yaml` - Model family presets (Flux, SDXL, Pony)
- `config/characters.yaml` - Character training presets
- `config/effects.yaml` - Effect training presets

## Example Migrations

### Simple Character Training
**Old:**
```bash
./train-flux-d2p-snr-prodigy.sh my_character
```

**New:**
```bash
python train.py --preset flux_prodigy_simple my_character
```

### Complex Style Training
**Old:**
```bash
# Edit script parameters manually
./train-sdxl-prodigy.sh my_style
```

**New:**
```bash
python train.py --preset sdxl_prodigy_complex my_style
```

### LyCORIS Training
**Old:**
```bash
./train-flux-d2p-snr-lokr.sh my_character
```

**New:**
```bash
python train.py --preset flux_lokr_simple --lycoris-subtype lokr my_character
```

### Resume Training
**Old:**
```bash
# Edit config.env file
./train-flux-d2p-snr-prodigy.sh my_character
```

**New:**
```bash
python train.py --preset flux_prodigy_simple --resume-from existing_weights.safetensors my_character
```

### Training with Total Images (New)
**Old:**
```bash
# Calculate steps manually based on dataset size
./train-flux-d2p-snr-prodigy.sh my_character
```

**New:**
```bash
python train.py --preset flux_total_images_example my_character
```

### Override Training Parameters (New)
**Old:**
```bash
# Edit script or config.env file
./train-flux-d2p-snr-prodigy.sh my_character
```

**New:**
```bash
python train.py --preset flux_prodigy_simple --total-images 1500 --batch-size 3 my_character
```

## Benefits of the New System

1. **Unified Interface**: Single script for all model families
2. **Configuration Management**: YAML-based configuration with inheritance
3. **Interactive Setup**: Wizard for easy configuration
4. **Flexibility**: Easy to add new optimizers and configurations
5. **Maintainability**: No more duplicate bash scripts
6. **Documentation**: Self-documenting YAML configurations
7. **Version Control**: Easy to track configuration changes
8. **Automation**: Better suited for CI/CD pipelines
9. **Consistent Training**: Total images parameter ensures consistent training across datasets
10. **Flexible Duration**: All model families support both steps and epochs
11. **Batch Size Control**: Explicit batch size control for all model families