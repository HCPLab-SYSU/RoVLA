# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import os
from pathlib import Path
from typing import List

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run
# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")

def preprocess_lerobot_dir(dataset_path: List[str], embodiment_tag: List[str]):
    """Preprocess the LeRobot dataset directory to ensure compatibility with GR00T fine-tuning."""
    assert len(dataset_path) == len(embodiment_tag), \
        "Length of dataset_path and embodiment_tag must be the same."
    all_dataset_path, all_embodiment_tag = [], []
    for dp, et in zip(dataset_path, embodiment_tag):
        if os.path.exists(os.path.join(dp, "meta")):
            if 'test' in dp.split('/')[-1].lower():
                print(f"Skipping test dataset at {dp}")
                continue
            all_dataset_path.append(dp)
            all_embodiment_tag.append(et)
        else:
            # Check for subdirectories
            subdirs = [
                os.path.join(dp, d) for d in os.listdir(dp)
                if os.path.isdir(os.path.join(dp, d)) and os.path.exists(os.path.join(dp, d, "meta"))
            ]
            if len(subdirs) == 0:
                raise ValueError(f"No valid dataset found in {dp}")
            all_dataset_path.extend(subdirs)
            all_embodiment_tag.extend([et] * len(subdirs))
    return all_dataset_path, all_embodiment_tag

if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    all_dataset_path, all_embodiment_tag = preprocess_lerobot_dir(dataset_path=ft_config.dataset_path, embodiment_tag = ft_config.embodiment_tag)
    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag.value,
                    } for dataset_path, embodiment_tag in zip(all_dataset_path, all_embodiment_tag)
                ],
            }
        }
    )
    config.load_config_path = None
    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.tune_top_llm_layers = ft_config.tune_top_llm_layers
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_pgd = ft_config.use_pgd
    config.model.use_consistency_learning = ft_config.use_consistency_learning
    config.model.add_condition_noise_sampler = ft_config.add_condition_noise_sampler
    
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    # config.model.eagle_collator = True
    # config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = ft_config.trainable_params_fp32
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"
    
    # Set learning rate scheduler type
    config.training.lr_scheduler_type = ft_config.lr_scheduler_type
    
    # Configure learning rate scheduler with key_module_lr_ratios
    config.training.key_module_lr_ratios = ft_config.key_module_lr_ratios
    
    # Enable gradient checkpointing and set DeepSpeed stage
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.deepspeed_stage = ft_config.deepspeed_stage
    # Configure learning rate scheduler with min_lr_rate if applicable
    lr_scheduler_kwargs = {}
    if ft_config.lr_scheduler_type in ['cosine_with_min_lr', 'cosine_warmup_with_min_lr']:
        lr_scheduler_kwargs['min_lr_rate'] = ft_config.min_lr_rate
    config.training.lr_scheduler_kwargs = lr_scheduler_kwargs

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
