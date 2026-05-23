# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import torch
import tyro
from dataclasses import dataclass
from typing import Optional
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import signal
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 配置：每个任务使用的 GPU 比例（例如 0.25 表示 1 GPU 可跑 4 个实例）
# PER_GPU_RATIO = 0.125
PER_GPU_RATIO = 1

DEFAULT_MODEL_SERVER_PORT = 5555

@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""


def run_server_process(config: ServerConfig, worker_id: int, gpu_id: int):
    """启动一个服务器进程，绑定到特定GPU"""
    # # 设置CUDA_VISIBLE_DEVICES以限制可见GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 设置随机种子，确保不同进程间独立性
    set_seed(42 + worker_id)
    
    # 动态分配端口以避免冲突
    actual_port = config.port + 10 * worker_id
    print(f"[Worker {worker_id}] Starting server on GPU {gpu_id}, port {actual_port}...")

    # 检查 model_path（仅对绝对路径做检查）
    if config.model_path and config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # 构建策略
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=f"cuda:{gpu_id}",
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # 应用 Sim Wrapper（如果启用）
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
        policy = Gr00tSimPolicyWrapper(policy)

    # 启动服务器
    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=actual_port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print(f"\n[Worker {worker_id}] Shutting down server on GPU {gpu_id}...")
    except Exception as e:
        print(f"[Worker {worker_id}] Server error: {e}")
        raise


def get_gpu_allocation(n_gpus: int, per_gpu_ratio: float):
    """计算GPU分配方案"""
    processes_per_gpu = int(1 / per_gpu_ratio)
    total_processes = n_gpus * processes_per_gpu
    
    gpu_allocations = []
    for gpu_id in range(n_gpus):
        for proc_idx in range(processes_per_gpu):
            gpu_allocations.append(gpu_id)
    
    return gpu_allocations


def main():
    config = tyro.cli(ServerConfig)

    # 检测GPU数量
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No GPU found. Distributed inference requires GPU.")

    # 计算分配方案
    gpu_allocations = get_gpu_allocation(n_gpu, PER_GPU_RATIO)
    total_workers = len(gpu_allocations)

    print(f"Detected {n_gpu} GPUs. Launching {total_workers} workers (each using {PER_GPU_RATIO} GPU).")
    print(f"GPU allocation: {gpu_allocations}")

    # 创建进程列表
    processes = []
    
    # 启动所有进程
    for i, gpu_id in enumerate(gpu_allocations):
        p = Process(target=run_server_process, args=(config, i, gpu_id))
        p.start()
        processes.append(p)
        print(f"Started worker {i} on GPU {gpu_id}")

    def signal_handler(signum, frame):
        print("\nReceived interrupt signal. Terminating all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)  # 等待最多5秒
        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 等待所有进程完成（实际上服务器是长期运行的）
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Main process interrupted. Terminating child processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)


if __name__ == "__main__":
    # 设置multiprocessing启动方式为spawn
    mp.set_start_method('spawn', force=True)
    main()



