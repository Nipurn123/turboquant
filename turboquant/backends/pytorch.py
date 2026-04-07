"""
PyTorch fallback backend.

Works on any hardware (CPU, GPU, etc.) without specialized kernels.
"""

import torch
from typing import Dict
from .base import BackendBase


class PyTorchBackend(BackendBase):
    """
    PyTorch fallback backend for TurboQuant.
    
    Implements all operations using pure PyTorch, working on any device.
    This is the reference implementation and fallback for hardware without
    optimized kernels.
    """
    
    def __init__(self, engine):
        """Initialize with reference to engine."""
        self.engine = engine
    
    @torch.no_grad()
    def compress_keys(self, K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress keys using PyTorch operations."""
        return self.engine.compress_keys(K)
    
    @torch.no_grad()
    def compress_values(self, V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress values using PyTorch operations."""
        return self.engine.compress_values(V)
    
    @torch.no_grad()
    def decompress_values(self, compressed_v: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decompress values using PyTorch operations."""
        return self.engine.decompress_values(compressed_v)
    
    @torch.no_grad()
    def attention_scores(
        self, Q: torch.Tensor, compressed_k: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention scores using PyTorch operations."""
        return self.engine.attention_scores(Q, compressed_k)
    
    @torch.no_grad()
    def fused_attention(
        self,
        Q: torch.Tensor,
        compressed_k: Dict[str, torch.Tensor],
        compressed_v: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute fused attention using PyTorch operations."""
        return self.engine.fused_attention(Q, compressed_k, compressed_v)
    
    @property
    def name(self) -> str:
        return "pytorch"
    
    @property
    def is_available(self) -> bool:
        return True
