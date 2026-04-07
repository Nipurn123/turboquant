"""Backend system for GPU-specific implementations."""

from .factory import get_backend, register_backend
from .base import BackendBase
from .pytorch import PyTorchBackend

__all__ = [
    "get_backend",
    "register_backend",
    "BackendBase",
    "PyTorchBackend",
]

try:
    from .vllm import TurboQuantKVCache, TurboQuantKVCacheConfig, create_turboquant_cache_for_vllm
    __all__.extend([
        "TurboQuantKVCache",
        "TurboQuantKVCacheConfig", 
        "create_turboquant_cache_for_vllm",
    ])
except ImportError:
    pass

try:
    from .sglang import TurboQuantKVPool, TurboQuantKVManager, create_turboquant_cache_for_sglang
    __all__.extend([
        "TurboQuantKVPool",
        "TurboQuantKVManager",
        "create_turboquant_cache_for_sglang",
    ])
except ImportError:
    pass

