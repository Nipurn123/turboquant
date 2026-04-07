"""
TurboQuant Universal - KV Cache Compression for LLM Inference

Supports:
- B200/B300 (Blackwell, SM 10.0)
- H200/H100 (Hopper, SM 9.0)  
- GLM-5 (MLA latent KV, 576-dim)
- Any model with configurable head dimensions

Based on: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
Paper: https://arxiv.org/abs/2504.19874
"""

from .core.engine import TurboQuantEngine
from .core.codebook import LloydMaxCodebook
from .core.rotation import RandomRotationMatrix
from .backends.factory import get_backend

__version__ = "1.0.0"
__author__ = "Community Implementation"

__all__ = [
    "TurboQuantEngine",
    "LloydMaxCodebook", 
    "RandomRotationMatrix",
    "get_backend",
]
