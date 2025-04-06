import yaml
import argparse
from datetime import datetime


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


def build_command(preset, training_name):
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

    timestamp = int(datetime.now().timestamp())

    # TODO: paths
    REMOTE_ROOT = "/workspace"
    FLUX_PATH = "/workspace/flux"
    DATASET_CONFIG = f"{REMOTE_ROOT}/config/{training_name}/dataset_config.yaml"
    OUTPUT_DIR = f"{REMOTE_ROOT}/output/{training_name}"
    OUTPUT_NAME = f"{training_name}-{opt_type}-{lr_sched}-{timestamp}"
    RESUME_WEIGHTS = f"{OUTPUT_DIR}/weights/{OUTPUT_NAME}.safetensors"

    launch_args = [
        "time accelerate launch \\",
        "    --mixed_precision bf16 \\",
        "    --num_cpu_threads_per_process 2 \\",
        f"    {REMOTE_ROOT}/sd-scripts/flux_train_network.py \\",
    ]

    model_args = [
        f"    --pretrained_model_name_or_path {FLUX_PATH}/flux1-dev2pro.safetensors \\",
        f"    --clip_l {FLUX_PATH}/clip_l.safetensors \\",
        f"    --t5xxl {FLUX_PATH}/t5xxl_fp16.safetensors \\",
        f"    --ae {FLUX_PATH}/ae.safetensors \\",
        "    --fp8_base \\",
        "    --highvram \\",
    ]

    cache_args = [
        "    --cache_latents_to_disk \\",
        "    --cache_text_encoder_outputs \\",
        "    --cache_text_encoder_outputs_to_disk \\",
        "    --persistent_data_loader_workers \\",
        "    --max_data_loader_n_workers 2 \\",
    ]

    save_args = [
        "    --mixed_precision bf16 \\",
        "    --save_precision bf16 \\",
        "    --save_model_as safetensors \\",
    ]

    network_args = [
        f"    --network_module {net['module']} \\",
        f"    --network_alpha {net['alpha']} \\",
        f"    --network_dim {net['dim']} \\",
        "    --network_args " + " ".join(network_args_strs) + " \\",
    ]

    optimizer_args = [
         f"    --optimizer_type {opt_type} \\",
        "    --optimizer_args \\",
        "        " + " \\\n        ".join(opt_args_strs) + " \\",
    ]

    scheduler_args = [
        f"    --lr_scheduler {lr_sched} \\",
        f"    --lr_scheduler_num_cycles {num_cycles} \\",
    ]

    train_args = [
        "    --discrete_flow_shift 3.1582 \\",
        "    --gradient_checkpointing \\",
        "    --guidance_scale 1.0 \\",
        f"    --learning_rate {p['learning_rate']} \\",
        "    --loss_type l2 \\",
        "    --model_prediction_type raw \\",
        f"    --noise_offset {p.get('noise_offset', 0.0)} \\",
        "    --sdpa \\",
        "    --seed 42 \\",
        f"    --timestep_sampling {p['timestep_sampling']} \\",
    ]

    min_snr_gamma = p.get('min_snr_gamma', 5)
    if min_snr_gamma > 0:
        train_args.extend([
          "    --huber_schedule=snr \\",
          f"    --min_snr_gamma={p.get('min_snr_gamma', 5)} \\",
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
        f"    --max_train_steps {p['max_steps']} \\",
        f"    --save_every_n_steps {p['save_every_n_steps']} \\",
        f"    --dataset_config {DATASET_CONFIG} \\",
        f"    --output_dir {OUTPUT_DIR} \\",
        f"    --output_name {OUTPUT_NAME} \\",
        f"    --sample_every_n_steps {p['sample_every_n_steps']} \\",
        f"    --sample_prompts {REMOTE_ROOT}/config/{training_name}/sample-prompts.txt \\",
        "    --log_with=all \\",
        f"    --logging_dir {OUTPUT_DIR}/logging/{OUTPUT_NAME} \\",
        f"    --wandb_run_name={training_name}-{opt_type}-{lr_sched}-{timestamp} \\",
        f"    {RESUME_WEIGHTS}"
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset_name", help="Name of the preset to use")
    parser.add_argument("--training-name", required=True, help="Training name for logging and prompt paths")
    parser.add_argument("--yaml", nargs='+', default=["config/effects.yaml", "config/characters.yaml"],
                        help="YAML files to load presets from")
    args = parser.parse_args()

    presets = load_presets(args.yaml)
    resolved = resolve_preset(presets, args.preset_name)
    cmd = build_command(resolved, args.training_name)
    print(cmd)


if __name__ == "__main__":
    main()
