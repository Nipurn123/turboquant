"""
Base backend interface.

Defines the interface that all GPU-specific backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class BackendBase(ABC):
    """
    Abstract base class for TurboQuant backends.
    
    Each backend implements GPU-specific optimizations for:
    - Key compression
    - Value compression
    - Attention scoring
    - Fused attention
    """
    
    @abstractmethod
    def __init__(self, engine: "TurboQuantEngine"):
        """Initialize backend with engine configuration."""
        pass
    
    @abstractmethod
    def compress_keys(self, K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress key cache."""
        pass
    
    @abstractmethod
    def compress_values(self, V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress value cache."""
        pass
    
    @abstractmethod
    def decompress_values(self, compressed_v: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decompress value cache."""
        pass
    
    @abstractmethod
    def attention_scores(
        self, Q: torch.Tensor, compressed_k: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention scores."""
        pass
    
    @abstractmethod
    def fused_attention(
        self,
        Q: torch.Tensor,
        compressed_k: Dict[str, torch.Tensor],
        compressed_v: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute fused attention."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available on current hardware."""
        pass
