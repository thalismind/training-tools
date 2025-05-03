import yaml
import argparse
from collections import defaultdict
from datetime import datetime
from glob import glob


def load_presets(yaml_paths):
    all_presets = {}
    for path in yaml_paths:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            for preset in data['presets']:
                all_presets[preset['name']] = preset
    return all_presets


def merge_dicts(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            merge_dicts(base[k], v)
        else:
            base[k] = v


def resolve_preset(presets, name):
    if name not in presets:
        raise ValueError(f"Preset '{name}' not found.")
    base = {}
    preset = presets[name]
    if 'inherits' in preset:
        for parent_name in preset['inherits']:
            parent = resolve_preset(presets, parent_name)
            merge_dicts(base, parent)
    merge_dicts(base, preset)
    return base


def build_command(preset, training_name, fp8_base=True, highvram=True, precision="bf16", threads=2, lycoris_subtype=None):
    p = preset['parameters']
    net = p['network']
    opt = p['optimizer']
    opt_args = opt.get('args', {})
    sch = p['scheduler']

    opt_type = opt['name']
    lr_sched = sch['name']
    num_cycles = sch.get('cycles', 1)

    if 'timestep_sampling' not in p:
        p['timestep_sampling'] = 'sigmoid'

    opt_args_strs = []
    for k, v in opt_args.items():
        if isinstance(v, bool):
            v = str(v)
        elif isinstance(v, list):
            v = ','.join(str(x) for x in v)
        opt_args_strs.append(f'"{k}={v}"')

    network_args_strs = ['"train_t5xxl=False"', f'"split_qkv={opt_args.get("split_qkv", False)}"']

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

    # TODO: paths
    REMOTE_ROOT = "/workspace"
    FLUX_PATH = "/workspace/flux"
    DATASET_CONFIG = f"{REMOTE_ROOT}/config/{training_name}/dataset.json"
    OUTPUT_DIR = f"{REMOTE_ROOT}/output/{training_name}"
    OUTPUT_NAME = f"{training_name}-{opt_type_name}-{lr_sched}-{timestamp}"
    RESUME_WEIGHTS = f"{OUTPUT_DIR}/weights/{OUTPUT_NAME}.safetensors"

    launch_args = [
        "time accelerate launch",
        f"    --mixed_precision {precision}",
        f"    --num_cpu_threads_per_process {threads}",
        f"    {REMOTE_ROOT}/sd-scripts/flux_train_network.py",
    ]

    model_args = [
        f"    --pretrained_model_name_or_path {FLUX_PATH}/flux1-dev2pro.safetensors",
        f"    --clip_l {FLUX_PATH}/clip_l.safetensors",
        f"    --t5xxl {FLUX_PATH}/t5xxl_fp16.safetensors",
        f"    --ae {FLUX_PATH}/ae.safetensors",
    ]

    if fp8_base:
        model_args.append("    --fp8_base")

    if highvram:
        model_args.append("    --highvram")

    cache_args = [
        "    --cache_latents_to_disk",
        "    --cache_text_encoder_outputs",
        "    --cache_text_encoder_outputs_to_disk",
        "    --persistent_data_loader_workers",
        f"    --max_data_loader_n_workers {threads}",
    ]

    save_args = [
        f"    --mixed_precision {precision}",
        f"    --save_precision {precision}",
        "    --save_model_as safetensors",
    ]

    network_module = "networks.lora_flux" if lycoris_subtype is None else "lycoris.kohya"

    network_args = [
        f"    --network_module {network_module}",
        f"    --network_alpha {net['alpha']}",
        f"    --network_dim {net['dim']}",
        "    --network_args",
        "        " + " \\\n        ".join(network_args_strs),
    ]

    optimizer_args = [
         f"    --optimizer_type {opt_type}",
        "    --optimizer_args",
        "        " + " \\\n        ".join(opt_args_strs),
    ]

    scheduler_args = [
        f"    --lr_scheduler {lr_sched}",
        f"    --lr_scheduler_num_cycles {num_cycles}",
    ]

    train_args = [
        "    --discrete_flow_shift 3.1582",
        "    --gradient_checkpointing",
        "    --guidance_scale 1.0",
        f"    --learning_rate {p['learning_rate']}",
        "    --loss_type l2",
        "    --model_prediction_type raw",
        f"    --noise_offset {p.get('noise_offset', 0.0)}",
        "    --sdpa",
        "    --seed 42",
        f"    --timestep_sampling {p['timestep_sampling']}",
    ]

    min_snr_gamma = p.get('min_snr_gamma', 5)
    if min_snr_gamma > 0:
        train_args.extend([
          "    --huber_schedule=snr",
          f"    --min_snr_gamma={p.get('min_snr_gamma', 5)}",
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
        f"    --max_train_steps {p['max_steps']}",
        f"    --save_every_n_steps {p['save_every_n_steps']}",
        f"    --dataset_config {DATASET_CONFIG}",
        f"    --output_dir {OUTPUT_DIR}",
        f"    --output_name {OUTPUT_NAME}",
        f"    --sample_every_n_steps {p['sample_every_n_steps']}",
        f"    --sample_prompts {REMOTE_ROOT}/config/{training_name}/sample-prompts.txt",
        "    --log_with=all",
        f"    --logging_dir {OUTPUT_DIR}/logging/{OUTPUT_NAME}",
        f"    --wandb_run_name={training_name}-{opt_type_name}-{lr_sched}-{timestamp}",
        f"    {RESUME_WEIGHTS}"
    ]

    # Add backslash to the end of each line except the last one
    slashes = [" \\"] * (len(lines) - 1) + [""]
    lines = [line + slash for line, slash in zip(lines, slashes)]

    return "\n".join(lines)


def wizard(yaml_files):
    import inquirer

    # Load all presets from given YAML files and group them by metadata.category
    all_presets = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
            for preset in data.get("presets", []):
                preset["_source_file"] = yaml_file
                all_presets.append(preset)

    # Group presets by category
    category_map = defaultdict(list)
    for preset in all_presets:
        category = preset.get("metadata", {}).get("category", "uncategorized")
        category_map[category].append(preset)

    # Ask user which category they want to train
    category_q = inquirer.prompt([
        inquirer.List("category", message="What category are you training?", choices=list(category_map.keys()))
    ])
    selected_category = category_q["category"]

    # Ask how complex the thing is
    complexity_q = inquirer.prompt([
        inquirer.List("complexity", message="How complex is the thing you are training?", choices=["simple", "complex"])
    ])
    complexity = complexity_q["complexity"]

    # TODO: Ask about dataset size once we can calculate the step count
    # dataset_q = inquirer.prompt([
    #     inquirer.List("dataset", message="How large is your dataset?", choices=[
    #         "1-10", "10-25", "25-50", "50-100", "100-500", "500-1000",
    #         "1000-5000", "5k-10k", "10-100k", "100k+"
    #     ])
    # ])
    # dataset_size = dataset_q["dataset"]

    # Ask if using LoRA or LyCORIS
    lora_type_q = inquirer.prompt([
        inquirer.List("lora_type", message="Are you using LoRA or LyCORIS?", choices=["lora", "lycoris"])
    ])
    lora_type = lora_type_q["lora_type"]

    lycoris_subtype = None
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
        print(f"Using min_snr_gamma={snr_gamma}")
    else:
        print("Not using min_snr_gamma")

    # Choose a matching preset name based on complexity
    suffix = "_simple" if complexity == "simple" else "_complex"
    candidates = [p for p in category_map[selected_category] if p["name"].endswith(suffix)]
    print(f"Found {len(candidates)} candidates for {suffix} in {selected_category}: {', '.join(p['name'] for p in candidates)}")

    if not candidates:
        selected_preset = category_map[selected_category][0]
    else:
        selected_preset = candidates[0]

    # Ask user to confirm the selected preset
    confirm_q = inquirer.prompt([
        inquirer.Confirm("confirm", message=f"Use preset '{selected_preset['name']}'?", default=True)
    ])
    if not confirm_q["confirm"]:
        print("Aborting.")
        raise SystemExit

    # Ask the user to confirm if the preset is not stable
    if not selected_preset['metadata'].get("stable", False):
        confirm_stable_q = inquirer.prompt([
            inquirer.Confirm("confirm_stable", message="The selected preset is not stable. Do you want to continue?", default=False)
        ])
        if not confirm_stable_q["confirm_stable"]:
            print("Aborting.")
            raise SystemExit

    print(f"Using preset: {selected_preset['name']}")
    print(f"Preset source file: {selected_preset['_source_file']}")
    print(f"Preset description: {selected_preset['metadata'].get('description', 'No description available')}")

    return {
        "yaml": selected_preset["_source_file"],
        "preset": selected_preset["name"],
        "snr_gamma": snr_gamma,
        "lora_type": lora_type,
        "lycoris_subtype": lycoris_subtype,
        # "dataset_size": dataset_size,
    }


def main():
    # Get yaml files in the config directory
    config_yaml = glob("config/*.yaml")
    print(f"Found {len(config_yaml)} YAML files in config directory: {', '.join(config_yaml)}")
    if not config_yaml:
        print("No YAML files found in config directory.")
        return

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs='+', default=config_yaml,
                        help="YAML files to load presets from")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", help="Name of the preset to use")
    group.add_argument("--wizard", action="store_true", help="Run the wizard to select a preset")

    parser.add_argument("training_name", help="Name of the training run")
    args = parser.parse_args()

    if args.wizard:
        result = wizard(args.sources)
        preset_name = result["preset"]
        training_name = args.training_name or input("Enter a training name: ")
        lycoris_subtype = result["lycoris_subtype"]
    else:
        preset_name = args.preset
        training_name = args.training_name
        lycoris_subtype = None

    presets = load_presets(config_yaml)
    resolved = resolve_preset(presets, preset_name)
    cmd = build_command(resolved, training_name, lycoris_subtype=lycoris_subtype)
    print(cmd)


if __name__ == "__main__":
    main()
