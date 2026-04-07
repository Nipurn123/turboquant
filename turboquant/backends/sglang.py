"""
SGLang integration for TurboQuant KV cache compression.

This module provides integration with SGLang's runtime for
KV cache compression using TurboQuant.

Usage:
    from turboquant.backends.sglang import TurboQuantKVManager
    
    # Initialize with SGLang's config
    kv_manager = TurboQuantKVManager.from_sglang_config(config)
"""

import torch
from typing import Dict, List, Optional, Tuple, Any

try:
    import sglang as sgl
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPool
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    ReqToTokenPool = object
    TokenToKVPool = object

from ..core.engine import TurboQuantEngine
from ..models.glm5 import GLM5Quantizer


class TurboQuantKVPool:
    """
    TurboQuant-compressed KV pool for SGLang.
    
    Replaces SGLang's standard TokenToKVPool with compression.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_total_tokens: int,
        total_bits: int = 3,
        device: str = "cuda",
        uses_mla: bool = False,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_total_tokens = max_total_tokens
        self.total_bits = total_bits
        self.device = device
        self.uses_mla = uses_mla
        
        if uses_mla:
            self._init_mla()
        else:
            self._init_standard()
    
    def _init_standard(self):
        """Initialize standard KV pool."""
        self.engine = TurboQuantEngine(
            head_dim=self.head_dim,
            total_bits=self.total_bits,
            device=self.device,
        )
        
        self.compressed_k_cache: List[Dict[str, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        self.compressed_v_cache: List[Dict[str, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        
        self.token_to_index = torch.zeros(
            self.max_total_tokens, dtype=torch.long, device=self.device
        )
        self.free_slots = list(range(self.max_total_tokens))
    
    def _init_mla(self):
        """Initialize MLA KV pool."""
        self.mla_engines = [
            GLM5Quantizer(
                num_heads=self.num_heads,
                device=self.device,
                latent_bits=self.total_bits,
            )
            for _ in range(self.num_layers)
        ]
        
        self.compressed_latent_cache: List[Dict] = [
            {} for _ in range(self.num_layers)
        ]
        self.compressed_indexer_cache: List[Dict] = [
            {} for _ in range(self.num_layers)
        ]
        
        self.token_to_index = torch.zeros(
            self.max_total_tokens, dtype=torch.long, device=self.device
        )
        self.free_slots = list(range(self.max_total_tokens))
    
    @torch.no_grad()
    def write(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Write KV pairs to the pool.
        
        Args:
            layer_idx: Layer index
            key: Key tensor (num_tokens, num_heads, head_dim)
            value: Value tensor (num_tokens, num_heads, head_dim)
            indices: Token indices to write to
        """
        if self.uses_mla:
            raise NotImplementedError("Use write_mla for MLA models")
        
        num_tokens, num_heads, head_dim = key.shape
        key_flat = key.reshape(num_tokens, num_heads * head_dim)
        value_flat = value.reshape(num_tokens, num_heads * head_dim)
        
        self.engine = TurboQuantEngine(
            head_dim=num_heads * head_dim,
            total_bits=self.total_bits,
            device=self.device,
        )
        
        compressed_k = self.engine.compress_keys(key_flat)
        compressed_v = self.engine.compress_values(value_flat)
        
        self.compressed_k_cache[layer_idx] = compressed_k
        self.compressed_v_cache[layer_idx] = compressed_v
        self.current_num_heads = num_heads
        self.current_head_dim = head_dim
        
        self.token_to_index[indices] = torch.arange(
            len(indices), device=self.device
        )
    
    @torch.no_grad()
    def write_mla(
        self,
        layer_idx: int,
        latent_k: torch.Tensor,
        latent_v: torch.Tensor,
        indexer_k: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Write MLA KV pairs to the pool.
        
        Args:
            layer_idx: Layer index
            latent_k: Latent key (num_tokens, latent_dim)
            latent_v: Latent value (num_tokens, latent_dim)
            indexer_k: Indexer key (num_tokens, num_heads, indexer_dim)
            indices: Token indices to write to
        """
        engine = self.mla_engines[layer_idx]
        
        compressed = engine.compress_latent_kv(latent_k, latent_v)
        compressed_idx = engine.compress_indexer(indexer_k)
        
        self.compressed_latent_cache[layer_idx] = compressed
        self.compressed_indexer_cache[layer_idx] = compressed_idx
        
        self.token_to_index[indices] = torch.arange(
            len(indices), device=self.device
        )
    
    def read(
        self,
        layer_idx: int,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read KV pairs from the pool.
        
        Args:
            layer_idx: Layer index
            indices: Token indices to read
            
        Returns:
            (key, value) tensors
        """
        if self.uses_mla:
            raise NotImplementedError("Use read_mla for MLA models")
        
        k_flat = self.engine.decompress_keys(self.compressed_k_cache[layer_idx])
        v_flat = self.engine.decompress_values(self.compressed_v_cache[layer_idx])
        
        kv_indices = self.token_to_index[indices]
        k_selected_flat = k_flat[kv_indices]
        v_selected_flat = v_flat[kv_indices]
        
        num_tokens = len(indices)
        num_heads = getattr(self, 'current_num_heads', self.num_heads)
        head_dim = getattr(self, 'current_head_dim', self.head_dim)
        
        k = k_selected_flat.reshape(num_tokens, num_heads, head_dim)
        v = v_selected_flat.reshape(num_tokens, num_heads, head_dim)
        return k, v
    
    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention with QJL correction.
        
        Args:
            layer_idx: Layer index
            query: Query tensor (seq_q, num_heads, head_dim)
            key_indices: Indices of keys to attend to
            
        Returns:
            Attention output
        """
        if self.uses_mla:
            raise NotImplementedError("Use compute_attention_mla for MLA models")
        
        seq_q, num_heads, head_dim = query.shape
        query_flat = query.reshape(seq_q, num_heads * head_dim).float()
        
        scores = self.engine.attention_scores(
            query_flat, self.compressed_k_cache[layer_idx]
        )
        
        kv_indices = self.token_to_index[key_indices]
        scores = scores[:, kv_indices]
        
        weights = torch.softmax(scores, dim=-1)
        
        v_flat = self.engine.decompress_values(self.compressed_v_cache[layer_idx]).float()
        v_selected_flat = v_flat[kv_indices]
        
        output_flat = torch.matmul(weights, v_selected_flat)
        output = output_flat.reshape(seq_q, num_heads, head_dim)
        return output.half()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if self.uses_mla:
            ratio = 5.24
        else:
            ratio = self.engine.compression_ratio
        
        return {
            'compression_ratio': ratio,
            'savings_percent': (1 - 1/ratio) * 100,
            'max_total_tokens': self.max_total_tokens,
        }


class TurboQuantKVManager:
    """
    KV cache manager for SGLang with TurboQuant compression.
    """
    
    def __init__(
        self,
        kv_pool: TurboQuantKVPool,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
    ):
        self.kv_pool = kv_pool
        self.req_to_token_pool = req_to_token_pool
    
    @classmethod
    def from_sglang_config(
        cls,
        config: Any,
        total_bits: int = 3,
    ) -> "TurboQuantKVManager":
        """
        Create KV manager from SGLang config.
        
        Args:
            config: SGLang server config
            total_bits: Bits per coordinate
            
        Returns:
            Configured TurboQuantKVManager
        """
        pool = TurboQuantKVPool(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            max_total_tokens=config.max_total_tokens,
            total_bits=total_bits,
            device=config.device,
            uses_mla=getattr(config, 'uses_mla', False),
        )
        
        if SGLANG_AVAILABLE:
            req_pool = ReqToTokenPool(
                max_num_reqs=config.max_num_requests,
                max_context_len=config.max_context_len,
                device=config.device,
            )
        else:
            req_pool = None
        
        return cls(pool, req_pool)
    
    def write_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        """Write KV to cache."""
        self.kv_pool.write(layer_idx, key, value, token_indices)
    
    def read_kv(
        self,
        layer_idx: int,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read KV from cache."""
        return self.kv_pool.read(layer_idx, token_indices)
    
    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention."""
        return self.kv_pool.compute_attention(layer_idx, query, key_indices)


def create_turboquant_cache_for_sglang(
    config: Any,
    total_bits: int = 3,
) -> TurboQuantKVManager:
    """
    Create TurboQuant cache for SGLang.
    
    Args:
        config: SGLang configuration
        total_bits: Bits per coordinate
        
    Returns:
        TurboQuantKVManager instance
    """
    return TurboQuantKVManager.from_sglang_config(config, total_bits)
