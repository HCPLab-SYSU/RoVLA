# Finetune config used for single node post-training.
from dataclasses import dataclass, field

from gr00t.data.embodiment_tags import EmbodimentTag
from typing import Any, List, Optional

@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: List[str]
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: List[EmbodimentTag]
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """
    resume_from_checkpoint: bool = False
    """ If True, resume fine-tuning from the specified base_model_path checkpoint."""
    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    """Number of top LLM layers to tune."""
    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """
    trainable_params_fp32: bool = False
    """If True, cast trainable backbone parameters to float32 precision during training."""
    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    # --- Learning Rate Scheduler Configuration ---
    lr_scheduler_type: str = "cosine"
    """Type of learning rate scheduler to use. Options: 'cosine', 'cosine_with_min_lr', 'cosine_warmup_with_min_lr'."""

    key_module_lr_ratios: Optional[dict[str, float]] = field(default_factory=dict)
    """
    Learning rate ratios for key modules.
    Keys are module names and values are corresponding learning rate ratios.
    """
    
    min_lr_rate: float = 0.0
    """Minimum learning rate ratio for schedulers that support it (e.g., cosine_with_min_lr).
    This is the ratio of the minimum LR to the initial LR. Only used when lr_scheduler_type
    is 'cosine_with_min_lr' or 'cosine_warmup_with_min_lr'."""

    use_pgd:  bool = False
    """If True, use Projected Gradient Descent (PGD) for adversarial training in action head."""
    
    use_consistency_learning: bool = False
    """If True, use consistency loss for training in action head."""
    
    # DeepSpeed (default)
    deepspeed_stage: int = 2  # ZeRO stage (1, 2, or 3)
    gradient_checkpointing: bool = False
    
    # backbone
    add_condition_noise_sampler: bool = False