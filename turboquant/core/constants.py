"""
Constants and configuration for TurboQuant Universal.
Supports multiple GPU architectures and model configurations.
"""

from dataclasses import dataclass
from typing import Dict, Set
import torch


@dataclass
class GPUArchitecture:
    name: str
    compute_capability: int
    sm_version: int
    supports_cutile: bool
    supports_triton: bool
    tensor_core_dtype: torch.dtype
    max_shared_mem: int


GPU_ARCHITECTURES: Dict[str, GPUArchitecture] = {
    "blackwell": GPUArchitecture(
        name="Blackwell",
        compute_capability=100,
        sm_version=10,
        supports_cutile=True,
        supports_triton=True,
        tensor_core_dtype=torch.float8_e4m3fn,
        max_shared_mem=227328,
    ),
    "hopper": GPUArchitecture(
        name="Hopper",
        compute_capability=90,
        sm_version=9,
        supports_cutile=False,
        supports_triton=True,
        tensor_core_dtype=torch.float8_e4m3fn,
        max_shared_mem=232448,
    ),
    "ampere": GPUArchitecture(
        name="Ampere",
        compute_capability=86,
        sm_version=8,
        supports_cutile=False,
        supports_triton=True,
        tensor_core_dtype=torch.float16,
        max_shared_mem=99200,
    ),
}


@dataclass
class ModelConfig:
    name: str
    head_dim_k: int
    head_dim_v: int
    num_heads: int
    num_kv_heads: int
    latent_dim: int
    uses_mla: bool
    uses_dsa: bool


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "llama3": ModelConfig(
        name="Llama 3",
        head_dim_k=128,
        head_dim_v=128,
        num_heads=32,
        num_kv_heads=8,
        latent_dim=0,
        uses_mla=False,
        uses_dsa=False,
    ),
    "mistral": ModelConfig(
        name="Mistral",
        head_dim_k=128,
        head_dim_v=128,
        num_heads=32,
        num_kv_heads=8,
        latent_dim=0,
        uses_mla=False,
        uses_dsa=False,
    ),
    "qwen2": ModelConfig(
        name="Qwen 2",
        head_dim_k=128,
        head_dim_v=128,
        num_heads=32,
        num_kv_heads=8,
        latent_dim=0,
        uses_mla=False,
        uses_dsa=False,
    ),
    "glm5": ModelConfig(
        name="GLM-5",
        head_dim_k=192,
        head_dim_v=256,
        num_heads=64,
        num_kv_heads=8,
        latent_dim=576,
        uses_mla=True,
        uses_dsa=True,
    ),
    "deepseek": ModelConfig(
        name="DeepSeek-V3",
        head_dim_k=192,
        head_dim_v=128,
        num_heads=128,
        num_kv_heads=128,
        latent_dim=512,
        uses_mla=True,
        uses_dsa=False,
    ),
}


DEFAULT_BLOCK_SIZES = {
    "blackwell": {
        "BLOCK_Q": 16,
        "BLOCK_KV": 64,
        "BLOCK_S": 64,
    },
    "hopper": {
        "BLOCK_Q": 16,
        "BLOCK_KV": 64,
        "BLOCK_S": 64,
    },
    "ampere": {
        "BLOCK_Q": 16,
        "BLOCK_KV": 32,
        "BLOCK_S": 32,
    },
}


SUPPORTED_BITS = {1, 2, 3, 4}
DEFAULT_TOTAL_BITS = 3
DEFAULT_SEED = 42
DEFAULT_EPSILON = 1e-8

INV_LOG_2 = 1.0 / torch.log(torch.tensor(2.0))


def get_gpu_architecture() -> str:
    """Detect current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    
    if sm >= 100:
        return "blackwell"
    elif sm >= 90:
        return "hopper"
    elif sm >= 80:
        return "ampere"
    else:
        return "unknown"


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    model_name = model_name.lower().replace("-", "").replace("_", "")
    
    for key, config in MODEL_CONFIGS.items():
        if key in model_name:
            return config
    
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported: {list(MODEL_CONFIGS.keys())}"
    )
