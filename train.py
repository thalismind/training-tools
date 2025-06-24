import yaml
import argparse
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator
import os


# Type aliases for better readability
ModelFamily = Literal["flux", "sdxl", "pony"]
VRAMMode = Literal["highvram", "lowvram", "none"]
LycorisSubtype = Literal["loha", "lokr"]
LoraType = Literal["lora", "lycoris"]
Complexity = Literal["simple", "complex"]
DurationType = Literal["total_images", "max_steps", "max_epochs"]


# Pydantic models for configuration validation
class NetworkConfig(BaseModel):
    alpha: int = Field(..., ge=1, le=128, description="Network alpha value")
    dim: int = Field(..., ge=1, le=256, description="Network dimension")
    module: Optional[str] = Field(None, description="Network module name")
    train_t5xxl: Optional[bool] = Field(default=False, description="Train T5XXL")
    split_qkv: Optional[bool] = Field(default=False, description="Split QKV")

class OptimizerArgs(BaseModel):
    weight_decay: Optional[float] = Field(None, ge=0.0, description="Weight decay")
    decouple: Optional[bool] = Field(None, description="Decouple option")
    betas: Optional[List[float]] = Field(None, description="Beta values")
    use_bias_correction: Optional[bool] = Field(None, description="Use bias correction")
    safeguard_warmup: Optional[bool] = Field(None, description="Safeguard warmup")
    d_coef: Optional[float] = Field(None, description="D coefficient")
    num_iterations: Optional[int] = Field(None, ge=1, description="Number of iterations")
    split_qkv: Optional[bool] = Field(None, description="Split QKV option")

class OptimizerConfig(BaseModel):
    name: str = Field(..., description="Optimizer name")
    args: OptimizerArgs = Field(default_factory=OptimizerArgs, description="Optimizer arguments")

class SchedulerConfig(BaseModel):
    name: str = Field(default="cosine", description="Scheduler name")
    cycles: int = Field(default=1, ge=1, description="Number of cycles")

class TrainingParameters(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_family: str = Field(..., pattern="^(flux|sdxl|pony)$", description="Model family")
    learning_rate: float = Field(..., gt=0.0, description="Learning rate")
    min_snr_gamma: Optional[float] = Field(None, ge=0.0, description="Minimum SNR gamma")
    noise_offset: Optional[float] = Field(default=0.0, description="Noise offset")
    save_every_n_steps: Optional[int] = Field(None, ge=1, description="Save every N steps")
    sample_every_n_steps: Optional[int] = Field(None, ge=1, description="Sample every N steps")
    timestep_sampling: Optional[str] = Field(default="sigmoid", description="Timestep sampling method")
    network: NetworkConfig = Field(..., description="Network configuration")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig, description="Scheduler configuration")

    # Training duration options (mutually exclusive)
    total_images: Optional[int] = Field(None, ge=1, description="Total images to process")
    max_steps: Optional[int] = Field(None, ge=1, description="Maximum training steps")
    max_epochs: Optional[int] = Field(None, ge=1, description="Maximum training epochs")

    # Additional options
    batch_size: int = Field(default=1, ge=1, description="Training batch size")
    network_train_unet_only: bool = Field(default=False, description="Train UNet only")

    @field_validator('total_images', 'max_steps', 'max_epochs')
    @classmethod
    def validate_training_duration(cls, v, info):
        """Ensure only one training duration method is specified"""
        values = info.data
        duration_methods = [values.get('total_images'), values.get('max_steps'), values.get('max_epochs')]
        if sum(1 for x in duration_methods if x is not None) > 1:
            raise ValueError("Only one of total_images, max_steps, or max_epochs should be specified")
        return v

class Metadata(BaseModel):
    category: str = Field(..., description="Preset category")
    description: str = Field(..., description="Preset description")
    stable: bool = Field(default=False, description="Whether preset is stable")
    version: str = Field(default="1.0", description="Preset version")

class Preset(BaseModel):
    name: str = Field(..., description="Preset name")
    metadata: Metadata = Field(..., description="Preset metadata")
    parameters: TrainingParameters = Field(..., description="Training parameters")
    inherits: Optional[List[str]] = Field(None, description="Parent presets to inherit from")

class ConfigFile(BaseModel):
    presets: List[Preset] = Field(..., description="List of presets")


def load_presets(yaml_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load presets from YAML files and return a dictionary of preset configurations."""
    all_presets: Dict[str, Dict[str, Any]] = {}

    # First pass: load all raw data without validation
    for path in yaml_paths:
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                for preset_data in data.get("presets", []):
                    preset_name = preset_data.get("name")
                    if preset_name:
                        all_presets[preset_name] = preset_data
        except Exception as e:
            print(f"Error loading config file {path}: {e}", file=sys.stderr)
            raise

    # Second pass: resolve inheritance and validate
    resolved_presets: Dict[str, Dict[str, Any]] = {}
    for preset_name in all_presets:
        try:
            resolved = resolve_preset(all_presets, preset_name)
            # Validate the resolved preset
            validated_preset = Preset(**resolved)
            resolved_presets[preset_name] = validated_preset.model_dump()
        except Exception as e:
            print(f"Error resolving preset '{preset_name}': {e}", file=sys.stderr)
            raise

    return resolved_presets


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override dictionary into base dictionary."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            merge_dicts(base[k], v)
        else:
            base[k] = v

    # Special handling for training duration conflicts
    # If the override specifies any training duration, clear the others from base
    if 'parameters' in override and 'parameters' in base:
        override_params = override['parameters']
        base_params = base['parameters']

        # Check if override specifies any training duration
        duration_fields = ['total_images', 'max_steps', 'max_epochs']
        override_has_duration = any(field in override_params for field in duration_fields)

        if override_has_duration:
            # Clear conflicting duration fields from base
            for field in duration_fields:
                if field in base_params and field not in override_params:
                    del base_params[field]


def resolve_preset(presets: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Resolve a preset by merging with its parent presets."""
    if name not in presets:
        raise ValueError(f"Preset '{name}' not found.")

    base: Dict[str, Any] = {}
    preset = presets[name]

    # Always start with default preset if it exists and we're not resolving default itself
    if name != "default" and "default" in presets:
        default_preset = resolve_preset(presets, "default")
        merge_dicts(base, default_preset)

    # Then merge with any explicitly inherited presets
    if 'inherits' in preset and preset['inherits'] is not None:
        for parent_name in preset['inherits']:
            if parent_name != "default":  # Skip default as it's already handled above
                parent = resolve_preset(presets, parent_name)
                merge_dicts(base, parent)

    # Finally merge the current preset on top
    merge_dicts(base, preset)
    return base


# --- Helper functions for runtime logic ---
def calculate_steps_from_total_images(total_images: int, batch_size: int) -> int:
    """Calculate training steps from total images and batch size."""
    if batch_size < 1:
        batch_size = 1
    steps = total_images // batch_size
    return max(1, steps)


def get_effective_step_count(parameters: Dict[str, Any]) -> int:
    batch_size = parameters.get('batch_size', 1)
    total_images = parameters.get('total_images')
    max_steps = parameters.get('max_steps')
    max_epochs = parameters.get('max_epochs')
    if total_images is not None:
        return calculate_steps_from_total_images(total_images, batch_size)
    elif max_steps is not None:
        return max_steps
    elif max_epochs is not None:
        # Fallback: use 1200 as a default for epoch-based runs if not otherwise specified
        return 1200
    else:
        return 1200


def get_save_every_n_steps(step_count: int, explicit: Optional[int] = None) -> int:
    """Calculate save frequency based on step count or explicit value."""
    if explicit and explicit > 0:
        return explicit
    return max(1, min(step_count // 10, 100))


def get_sample_every_n_steps(explicit: Optional[int] = None) -> int:
    """Get sample frequency, defaulting to 5000 if not specified."""
    return explicit if explicit and explicit > 0 else 5000


def build_command(
    preset: Dict[str, Any],
    training_name: str,
    fp8_base: bool = True,
    vram_mode: VRAMMode = "highvram",
    precision: str = "bf16",
    threads: int = 2,
    lycoris_subtype: Optional[LycorisSubtype] = None,
    resume_from: Optional[str] = None
) -> str:
    """Build the training command string from preset configuration."""
    # Load paths from environment with defaults
    REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/workspace")
    FLUX_PATH = os.environ.get("FLUX_PATH", f"{REMOTE_ROOT}/flux")
    SDXL_PATH = os.environ.get("SDXL_PATH", f"{REMOTE_ROOT}/sdxl")

    p = preset['parameters']
    net = p['network']
    opt = p['optimizer']
    opt_args = opt.get('args', {})
    sch = p['scheduler']
    model_family: ModelFamily = p.get('model_family', 'flux')  # Default to flux if not specified

    opt_type = opt['name']
    lr_sched = sch.get('name', 'cosine')  # Default to cosine as requested
    num_cycles = sch.get('cycles', 1)  # Default to 1 cycle as requested

    if 'timestep_sampling' not in p:
        p['timestep_sampling'] = 'sigmoid'

    # Runtime defaults for train_t5xxl and split_qkv
    if 'train_t5xxl' not in net:
        net['train_t5xxl'] = False
    if 'split_qkv' not in net:
        net['split_qkv'] = False

    # Convert optimizer args to strings
    opt_args_strs: List[str] = []
    for k, v in opt_args.items():
        if isinstance(v, bool):
            v = str(v)
        elif isinstance(v, list):
            v = ','.join(str(x) for x in v)
        opt_args_strs.append(f'"{k}={v}"')

    # --- Cleaned up step logic ---
    batch_size = p.get('batch_size', 1)
    step_count = get_effective_step_count(p)
    # Training duration string
    if 'total_images' in p:
        print(f"Calculated {step_count} steps from {p['total_images']} total images with batch size {batch_size}", file=sys.stderr)
        training_duration = f"    --max_train_steps {step_count}"
    elif 'max_steps' in p:
        training_duration = f"    --max_train_steps {p['max_steps']}"
    elif 'max_epochs' in p:
        training_duration = f"    --max_train_epochs {p['max_epochs']}"
    else:
        training_duration = f"    --max_train_steps {step_count}"

    # --- Cleaned up runtime defaults ---
    save_every_n_steps = get_save_every_n_steps(step_count, p.get('save_every_n_steps'))
    sample_every_n_steps = get_sample_every_n_steps(p.get('sample_every_n_steps'))

    # Model family specific configurations
    if model_family == 'flux':
        training_script = f"{REMOTE_ROOT}/sd-scripts/flux_train_network.py"
        model_args = [
            f"    --pretrained_model_name_or_path {FLUX_PATH}/flux1-dev2pro.safetensors",
            f"    --clip_l {FLUX_PATH}/clip_l.safetensors",
            f"    --t5xxl {FLUX_PATH}/t5xxl_fp16.safetensors",
            f"    --ae {FLUX_PATH}/ae.safetensors",
        ]
        if fp8_base:
            model_args.append("    --fp8_base")
        network_module = "networks.lora_flux" if lycoris_subtype is None else "lycoris.kohya"
        precision = "bf16"
        save_precision = "bf16"
        mixed_precision = "bf16"

    elif model_family == 'sdxl':
        training_script = f"{REMOTE_ROOT}/sd-scripts/sdxl_train_network.py"
        model_args = [
            f"    --pretrained_model_name_or_path {SDXL_PATH}/sdxl_v1_vae_fix.safetensors",
            "    --no_half_vae",
        ]
        network_module = "networks.lora" if lycoris_subtype is None else "lycoris.kohya"
        precision = "no"
        save_precision = "float"
        mixed_precision = "no"

    elif model_family == 'pony':
        training_script = f"{REMOTE_ROOT}/sd-scripts/sdxl_train_network.py"
        model_args = [
            f"    --pretrained_model_name_or_path {SDXL_PATH}/pony_v6_base.safetensors",
            "    --no_half_vae",
        ]
        network_module = "networks.lora" if lycoris_subtype is None else "lycoris.kohya"
        precision = "no"
        save_precision = "float"
        mixed_precision = "no"

    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    # Network arguments based on model family and type
    network_args_strs: List[str] = []
    network_args_strs.append(f'"train_t5xxl={net.get("train_t5xxl", False)}"')
    network_args_strs.append(f'"split_qkv={net.get("split_qkv", False)}"')

    # Additional network arguments for LyCORIS
    if lycoris_subtype:
        network_args_strs.extend([
            f"conv_dim={net['dim']}",
            f"conv_alpha={net['alpha']}",
            f"algo={lycoris_subtype}",
            "bypass_mode=false",
        ])

    timestamp = int(datetime.now().timestamp())

    # If opt_type has a dot, keep the final part
    if '.' in opt_type:
        opt_type_name = opt_type.split('.')[-1]
    else:
        opt_type_name = opt_type

    DATASET_CONFIG = f"{REMOTE_ROOT}/config/{training_name}/dataset.json"
    OUTPUT_DIR = f"{REMOTE_ROOT}/output/{training_name}"
    OUTPUT_NAME = f"{training_name}-{opt_type_name}-{lr_sched}-{timestamp}"

    # Resume weights handling
    resume_weights = ""
    if resume_from:
        resume_weights = f"    --network_weights {REMOTE_ROOT}/dataset/{resume_from}"

    launch_args = [
        "time accelerate launch",
        f"    --mixed_precision {mixed_precision}",
        f"    --num_cpu_threads_per_process {threads}",
        f"    {training_script}",
    ]

    # Add VRAM mode (highvram or lowvram, but not both)
    if vram_mode == "highvram":
        model_args.append("    --highvram")
    elif vram_mode == "lowvram":
        model_args.append("    --lowvram")

    cache_args = [
        "    --cache_latents_to_disk",
        "    --cache_text_encoder_outputs",
        "    --cache_text_encoder_outputs_to_disk",
        "    --persistent_data_loader_workers",
        f"    --max_data_loader_n_workers {threads}",
    ]

    save_args = [
        f"    --mixed_precision {mixed_precision}",
        f"    --save_precision {save_precision}",
        "    --save_model_as safetensors",
    ]

    network_args = [
        f"    --network_module {network_module}",
        f"    --network_alpha {net['alpha']}",
        f"    --network_dim {net['dim']}",
        "    --network_args",
        "        " + " \\\n        ".join(network_args_strs),
    ]

    # Add network_train_unet_only if configured
    if p.get('network_train_unet_only', False):
        network_args.append("    --network_train_unet_only")

    optimizer_args = [
         f"    --optimizer_type {opt_type}",
        "    --optimizer_args",
        "        " + " \\\n        ".join(opt_args_strs),
    ]

    scheduler_args = [
        f"    --lr_scheduler {lr_sched}",
        f"    --lr_scheduler_num_cycles {num_cycles}",
    ]

    # Training arguments - model family specific
    train_args = [
        "    --gradient_checkpointing",
        f"    --learning_rate {p['learning_rate']}",
        "    --loss_type l2",
        f"    --noise_offset {p.get('noise_offset', 0.0)}",
        "    --sdpa",
        "    --seed 42",
    ]

    # Add batch size if specified
    if batch_size > 1:
        train_args.append(f"    --train_batch_size {batch_size}")

    # Add model-specific arguments
    if model_family == 'flux':
        train_args.extend([
            "    --discrete_flow_shift 3.1582",
            "    --guidance_scale 1.0",
            "    --model_prediction_type raw",
            f"    --timestep_sampling {p['timestep_sampling']}",
        ])

    # Add SNR gamma if specified
    min_snr_gamma = p.get('min_snr_gamma', 5)
    if min_snr_gamma > 0:
        train_args.extend([
          "    --huber_schedule=snr",
          f"    --min_snr_gamma={min_snr_gamma}",
        ])

    lines = [
        *launch_args,
        *model_args,
        *cache_args,
        *save_args,
        *network_args,
        *optimizer_args,
        *scheduler_args,
        *train_args,
        training_duration,
        f"    --save_every_n_steps {save_every_n_steps}",
        f"    --dataset_config {DATASET_CONFIG}",
        f"    --output_dir {OUTPUT_DIR}",
        f"    --output_name {OUTPUT_NAME}",
        f"    --sample_every_n_steps {sample_every_n_steps}",
        f"    --sample_prompts {REMOTE_ROOT}/config/{training_name}/sample-prompts.txt",
        "    --log_with=all",
        f"    --logging_dir {OUTPUT_DIR}/logging/{OUTPUT_NAME}",
        f"    --wandb_run_name={training_name}-{opt_type_name}-{lr_sched}-{timestamp}",
    ]

    # Add resume weights if specified
    if resume_weights:
        lines.append(resume_weights)

    # Add backslash to the end of each line except the last one
    slashes = [" \\"] * (len(lines) - 1) + [""]
    lines = [line + slash for line, slash in zip(lines, slashes)]

    return "\n".join(lines)


def wizard(yaml_files: List[str]) -> Dict[str, Any]:
    """Interactive wizard to select training configuration."""
    import inquirer

    # Load all presets from given YAML files and group them by metadata.category
    all_presets: List[Dict[str, Any]] = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
            for preset in data.get("presets", []):
                preset["_source_file"] = yaml_file
                all_presets.append(preset)

    # Group presets by category
    category_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for preset in all_presets:
        category = preset.get("metadata", {}).get("category", "uncategorized")
        category_map[category].append(preset)

    # Ask user which category they want to train
    category_q = inquirer.prompt([
        inquirer.List("category", message="What category are you training?", choices=list(category_map.keys()))
    ])
    selected_category = category_q["category"]

    # Ask which model family to use
    model_family_q = inquirer.prompt([
        inquirer.List("model_family", message="Which model family are you using?", choices=["flux", "sdxl", "pony"])
    ])
    model_family: ModelFamily = model_family_q["model_family"]

    # Ask how complex the thing is
    complexity_q = inquirer.prompt([
        inquirer.List("complexity", message="How complex is the thing you are training?", choices=["simple", "complex"])
    ])
    complexity: Complexity = complexity_q["complexity"]

    # Ask about training duration preference
    duration_q = inquirer.prompt([
        inquirer.List("duration_type", message="How do you want to specify training duration?", choices=[
            "total_images", "max_steps", "max_epochs"
        ])
    ])
    duration_type: DurationType = duration_q["duration_type"]

    # Ask for batch size
    batch_size_q = inquirer.prompt([
        inquirer.Text("batch_size", message="What batch size do you want to use?", default="1")
    ])
    batch_size = int(batch_size_q["batch_size"])

    # Ask for duration value based on type
    if duration_type == "total_images":
        duration_q = inquirer.prompt([
            inquirer.Text("total_images", message="How many total images do you want to process?", default="1000")
        ])
        total_images = int(duration_q["total_images"])
        max_steps = None
        max_epochs = None
    elif duration_type == "max_steps":
        duration_q = inquirer.prompt([
            inquirer.Text("max_steps", message="How many training steps?", default="1200")
        ])
        max_steps = int(duration_q["max_steps"])
        total_images = None
        max_epochs = None
    else:  # max_epochs
        duration_q = inquirer.prompt([
            inquirer.Text("max_epochs", message="How many training epochs?", default="5")
        ])
        max_epochs = int(duration_q["max_epochs"])
        total_images = None
        max_steps = None

    # Ask about VRAM mode
    vram_q = inquirer.prompt([
        inquirer.List("vram_mode", message="What VRAM mode do you want to use?", choices=[
            "highvram", "lowvram", "none"
        ])
    ])
    vram_mode: VRAMMode = vram_q["vram_mode"]

    # Ask about network training mode
    network_q = inquirer.prompt([
        inquirer.Confirm("network_train_unet_only", message="Train UNet only?", default=False)
    ])
    network_train_unet_only = network_q["network_train_unet_only"]

    # Ask if using LoRA or LyCORIS
    lora_type_q = inquirer.prompt([
        inquirer.List("lora_type", message="Are you using LoRA or LyCORIS?", choices=["lora", "lycoris"])
    ])
    lora_type: LoraType = lora_type_q["lora_type"]

    lycoris_subtype: Optional[LycorisSubtype] = None
    if lora_type == "lycoris":
        subtype_q = inquirer.prompt([
            inquirer.List("lycoris_subtype", message="What kind of LyCORIS? (loha or lokr)", choices=["loha", "lokr"])
        ])
        lycoris_subtype = subtype_q["lycoris_subtype"]

    # Ask for a min_snr_gamma value
    snr_gamma_q = inquirer.prompt([
        inquirer.Text("snr_gamma", message="What minimum SNR gamma value do you want to use?", default="5")
    ])
    snr_gamma = float(snr_gamma_q["snr_gamma"])
    snr_gamma_enabled = snr_gamma > 0
    if snr_gamma_enabled:
        print(f"Using min_snr_gamma={snr_gamma}", file=sys.stderr)
    else:
        print("Not using min_snr_gamma", file=sys.stderr)

    # Ask about resuming training
    resume_q = inquirer.prompt([
        inquirer.Text("resume_from", message="Resume from existing weights? (leave empty for no)", default="")
    ])
    resume_from = resume_q["resume_from"] if resume_q["resume_from"].strip() else None

    # Choose a matching preset name based on complexity
    suffix = "_simple" if complexity == "simple" else "_complex"
    candidates = [p for p in category_map[selected_category] if p["name"].endswith(suffix)]
    print(f"Found {len(candidates)} candidates for {suffix} in {selected_category}: {', '.join(p['name'] for p in candidates)}", file=sys.stderr)

    if not candidates:
        selected_preset = category_map[selected_category][0]
    else:
        selected_preset = candidates[0]

    # Ask user to confirm the selected preset
    confirm_q = inquirer.prompt([
        inquirer.Confirm("confirm", message=f"Use preset '{selected_preset['name']}'?", default=True)
    ])
    if not confirm_q["confirm"]:
        print("Aborting.", file=sys.stderr)
        raise SystemExit

    # Ask the user to confirm if the preset is not stable
    if not selected_preset['metadata'].get("stable", False):
        confirm_stable_q = inquirer.prompt([
            inquirer.Confirm("confirm_stable", message="The selected preset is not stable. Do you want to continue?", default=False)
        ])
        if not confirm_stable_q["confirm_stable"]:
            print("Aborting.", file=sys.stderr)
            raise SystemExit

    print(f"Using preset: {selected_preset['name']}", file=sys.stderr)
    print(f"Preset source file: {selected_preset['_source_file']}", file=sys.stderr)
    print(f"Preset description: {selected_preset['metadata'].get('description', 'No description available')}", file=sys.stderr)

    return {
        "yaml": selected_preset["_source_file"],
        "preset": selected_preset["name"],
        "model_family": model_family,
        "batch_size": batch_size,
        "total_images": total_images,
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "vram_mode": vram_mode,
        "network_train_unet_only": network_train_unet_only,
        "snr_gamma": snr_gamma,
        "lora_type": lora_type,
        "lycoris_subtype": lycoris_subtype,
        "resume_from": resume_from,
    }


def save_final_configuration(
    resolved_config: Dict[str, Any],
    training_name: str,
    preset_name: str,
    vram_mode: VRAMMode,
    lycoris_subtype: Optional[LycorisSubtype],
    resume_from: Optional[str]
) -> str:
    """Save the final configuration to a YAML file for later analysis."""
    timestamp = int(datetime.now().timestamp())

    # Create a new preset name for the final configuration
    final_preset_name = f"{training_name}_{timestamp}"

    # Create the final configuration
    final_config = {
        "name": final_preset_name,
        "metadata": {
            "category": "generated",
            "description": f"Auto-generated configuration for training run '{training_name}' based on preset '{preset_name}'",
            "stable": False,
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "original_preset": preset_name,
            "training_name": training_name,
            "vram_mode": vram_mode,
            "lycoris_subtype": lycoris_subtype,
            "resume_from": resume_from
        },
        "parameters": resolved_config["parameters"]
    }

    # Create the full config file structure
    config_file = {
        "presets": [final_config]
    }

    # Ensure the output directory exists
    output_dir = "config/generated"
    os.makedirs(output_dir, exist_ok=True)

    # Save to YAML file
    filename = f"{output_dir}/{training_name}_{timestamp}.yaml"
    with open(filename, 'w') as f:
        yaml.dump(config_file, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"Saved final configuration to: {filename}", file=sys.stderr)
    return filename


def main() -> None:
    """Main entry point for the training script."""
    # Get yaml files in the config directory
    config_yaml = glob("config/*.yaml")
    print(f"Found {len(config_yaml)} YAML files in config directory: {', '.join(config_yaml)}", file=sys.stderr)
    if not config_yaml:
        print("No YAML files found in config directory.", file=sys.stderr)
        return

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs='+', default=config_yaml,
                        help="YAML files to load presets from")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", help="Name of the preset to use")
    group.add_argument("--wizard", action="store_true", help="Run the wizard to select a preset")

    parser.add_argument("--resume-from", help="Resume training from existing weights file")
    parser.add_argument("--model-family", choices=["flux", "sdxl", "pony"],
                        help="Model family to use (flux, sdxl, pony)")
    parser.add_argument("--lycoris-subtype", choices=["loha", "lokr"],
                        help="LyCORIS subtype (loha or lokr)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--total-images", type=int,
                        help="Total number of images to process (overrides max_steps/max_epochs)")
    parser.add_argument("--max-steps", type=int,
                        help="Maximum training steps (overrides max_epochs)")
    parser.add_argument("--max-epochs", type=int,
                        help="Maximum training epochs")
    parser.add_argument("--vram-mode", choices=["highvram", "lowvram", "none"], default="highvram",
                        help="VRAM mode (highvram, lowvram, or none)")
    parser.add_argument("--network-train-unet-only", action="store_true",
                        help="Train UNet only")
    parser.add_argument("--min-snr-gamma", type=float,
                        help="Minimum SNR gamma value")
    parser.add_argument("--noise-offset", type=float,
                        help="Noise offset value")
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate (overrides preset)")
    parser.add_argument("--save-every-n-steps", type=int,
                        help="Save every N steps")
    parser.add_argument("--sample-every-n-steps", type=int,
                        help="Sample every N steps")
    parser.add_argument("training_name", help="Name of the training run")
    args = parser.parse_args()

    if args.wizard:
        result = wizard(args.sources)
        preset_name = result["preset"]
        training_name = args.training_name or input("Enter a training name: ")
        lycoris_subtype = result["lycoris_subtype"]
        resume_from = result["resume_from"]
        model_family = result["model_family"]
        batch_size = result["batch_size"]
        total_images = result["total_images"]
        max_steps = result["max_steps"]
        max_epochs = result["max_epochs"]
        vram_mode = result["vram_mode"]
        network_train_unet_only = result["network_train_unet_only"]
        min_snr_gamma = result["snr_gamma"]
    else:
        preset_name = args.preset
        training_name = args.training_name
        lycoris_subtype = args.lycoris_subtype
        resume_from = args.resume_from
        model_family = args.model_family
        batch_size = args.batch_size
        total_images = args.total_images
        max_steps = args.max_steps
        max_epochs = args.max_epochs
        vram_mode = args.vram_mode
        network_train_unet_only = args.network_train_unet_only
        min_snr_gamma = args.min_snr_gamma

    presets = load_presets(args.sources)
    resolved = resolve_preset(presets, preset_name)

    # Update the preset with command line overrides
    if model_family:
        resolved['parameters']['model_family'] = model_family
    if batch_size > 1:
        resolved['parameters']['batch_size'] = batch_size
    if total_images:
        resolved['parameters']['total_images'] = total_images
    if max_steps:
        resolved['parameters']['max_steps'] = max_steps
    if max_epochs:
        resolved['parameters']['max_epochs'] = max_epochs
    if network_train_unet_only:
        resolved['parameters']['network_train_unet_only'] = network_train_unet_only
    if min_snr_gamma is not None:
        resolved['parameters']['min_snr_gamma'] = min_snr_gamma
    if args.noise_offset is not None:
        resolved['parameters']['noise_offset'] = args.noise_offset
    if args.learning_rate is not None:
        resolved['parameters']['learning_rate'] = args.learning_rate
    if args.save_every_n_steps is not None:
        resolved['parameters']['save_every_n_steps'] = args.save_every_n_steps
    if args.sample_every_n_steps is not None:
        resolved['parameters']['sample_every_n_steps'] = args.sample_every_n_steps

    # Save final configuration before printing command
    save_final_configuration(resolved, training_name, preset_name, vram_mode, lycoris_subtype, resume_from)

    cmd = build_command(resolved, training_name, vram_mode=vram_mode, lycoris_subtype=lycoris_subtype, resume_from=resume_from)
    print(cmd)


if __name__ == "__main__":
    main()
