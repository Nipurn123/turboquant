"""
vLLM integration for TurboQuant KV cache compression.

This module provides a drop-in replacement for vLLM's KV cache
with TurboQuant compression enabled.

Usage:
    from turboquant.backends.vllm import TurboQuantKVCache
    
    # In vLLM config
    kv_cache_dtype = "turboquant_3bit"
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    AttentionBackend = object
    FlashAttentionBackend = object

from ..core.engine import TurboQuantEngine
from ..models.glm5 import GLM5Quantizer


@dataclass
class TurboQuantKVCacheConfig:
    """Configuration for TurboQuant KV cache."""
    
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    total_bits: int = 3
    device: str = "cuda"
    model_name: Optional[str] = None
    
    # GLM-5 MLA specific
    uses_mla: bool = False
    latent_dim: int = 576
    qk_head_dim: int = 192
    v_head_dim: int = 256


class TurboQuantKVCache:
    """
    TurboQuant-compressed KV cache for vLLM.
    
    Drop-in replacement for vLLM's standard KV cache with compression.
    """
    
    def __init__(self, config: TurboQuantKVCacheConfig):
        self.config = config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.device = config.device
        
        if config.uses_mla:
            self._init_mla_cache(config)
        else:
            self._init_standard_cache(config)
    
    def _init_standard_cache(self, config: TurboQuantKVCacheConfig):
        """Initialize standard (non-MLA) cache."""
        self.engine = TurboQuantEngine(
            head_dim=config.head_dim,
            total_bits=config.total_bits,
            device=config.device,
            model_name=config.model_name,
        )
        
        self.compressed_k: List[Dict[str, torch.Tensor]] = []
        self.compressed_v: List[Dict[str, torch.Tensor]] = []
        
        for _ in range(config.num_layers):
            self.compressed_k.append({})
            self.compressed_v.append({})
    
    def _init_mla_cache(self, config: TurboQuantKVCacheConfig):
        """Initialize MLA cache for GLM-5 style models."""
        self.mla_quantizers: List[GLM5Quantizer] = []
        
        for _ in range(config.num_layers):
            quantizer = GLM5Quantizer(
                num_heads=config.num_heads,
                device=config.device,
                latent_bits=config.total_bits,
            )
            self.mla_quantizers.append(quantizer)
        
        self.compressed_latent: List[Dict] = []
        self.compressed_indexer: List[Dict] = []
        
        for _ in range(config.num_layers):
            self.compressed_latent.append({})
            self.compressed_indexer.append({})
    
    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Update KV cache with new key-value pairs.
        
        Args:
            layer_idx: Layer index
            key: Key tensor (seq_len, num_heads, head_dim)
            value: Value tensor (seq_len, num_heads, head_dim)
        """
        if hasattr(self, 'engine'):
            seq_len, num_heads, head_dim = key.shape
            key_flat = key.reshape(seq_len, num_heads * head_dim)
            value_flat = value.reshape(seq_len, num_heads * head_dim)
            
            self.engine = TurboQuantEngine(
                head_dim=num_heads * head_dim,
                total_bits=self.config.total_bits,
                device=self.device,
            )
            
            compressed_k = self.engine.compress_keys(key_flat)
            compressed_v = self.engine.compress_values(value_flat)
            
            self.compressed_k[layer_idx] = compressed_k
            self.compressed_v[layer_idx] = compressed_v
            self.current_num_heads = num_heads
        else:
            raise NotImplementedError("MLA cache update requires projection matrices")
    
    @torch.no_grad()
    def update_mla(
        self,
        layer_idx: int,
        latent_k: torch.Tensor,
        latent_v: torch.Tensor,
        indexer_k: torch.Tensor,
    ) -> None:
        """
        Update MLA KV cache (for GLM-5 style models).
        
        Args:
            layer_idx: Layer index
            latent_k: Latent key (seq_len, latent_dim)
            latent_v: Latent value (seq_len, latent_dim)
            indexer_k: Indexer key (seq_len, num_heads, indexer_dim)
        """
        quantizer = self.mla_quantizers[layer_idx]
        
        compressed = quantizer.compress_latent_kv(latent_k, latent_v)
        compressed_idx = quantizer.compress_indexer(indexer_k)
        
        self.compressed_latent[layer_idx] = compressed
        self.compressed_indexer[layer_idx] = compressed_idx
    
    def get_kv(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full KV cache for a layer (decompressed).
        
        Args:
            layer_idx: Layer index
            
        Returns:
            (key, value) tensors
        """
        if hasattr(self, 'engine'):
            k_flat = self.engine.decompress_keys(self.compressed_k[layer_idx])
            v_flat = self.engine.decompress_values(self.compressed_v[layer_idx])
            
            seq_len = k_flat.shape[0]
            num_heads = getattr(self, 'current_num_heads', self.num_heads)
            head_dim = self.head_dim
            
            k = k_flat.reshape(seq_len, num_heads, head_dim)
            v = v_flat.reshape(seq_len, num_heads, head_dim)
            return k, v
        else:
            raise NotImplementedError("MLA get_kv not implemented for compressed format")
    
    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention with QJL correction.
        
        Args:
            layer_idx: Layer index
            query: Query tensor (seq_q, num_heads, head_dim)
            
        Returns:
            Attention output
        """
        if hasattr(self, 'engine'):
            seq_q, num_heads, head_dim = query.shape
            query_flat = query.reshape(seq_q, num_heads * head_dim).float()
            
            scores = self.engine.attention_scores(
                query_flat, self.compressed_k[layer_idx]
            )
            
            weights = torch.softmax(scores, dim=-1)
            
            v_flat = self.engine.decompress_values(self.compressed_v[layer_idx]).float()
            
            output_flat = torch.matmul(weights, v_flat)
            output = output_flat.reshape(seq_q, num_heads, head_dim)
            return output.half()
        else:
            raise NotImplementedError("MLA attention requires sparse attention API")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if hasattr(self, 'engine'):
            ratio = self.engine.compression_ratio
            original_mb = 0
            compressed_mb = 0
            
            for layer_idx in range(self.num_layers):
                if self.compressed_k[layer_idx]:
                    k_size = self.compressed_k[layer_idx]['indices_packed'].numel()
                    v_size = self.compressed_v[layer_idx]['indices_packed'].numel()
                    compressed_mb += (k_size + v_size) / (1024 * 1024)
            
            original_mb = compressed_mb * ratio
            
            return {
                'original_mb': original_mb,
                'compressed_mb': compressed_mb,
                'compression_ratio': ratio,
                'savings_percent': (1 - 1/ratio) * 100,
            }
        else:
            return {'compression_ratio': 0, 'savings_percent': 0}


def create_turboquant_cache_for_vllm(
    model_config: Any,
    total_bits: int = 3,
) -> TurboQuantKVCache:
    """
    Create TurboQuant KV cache from vLLM model config.
    
    Args:
        model_config: vLLM model configuration
        total_bits: Bits per coordinate (default: 3)
        
    Returns:
        Configured TurboQuantKVCache
    """
    config = TurboQuantKVCacheConfig(
        num_layers=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.head_dim,
        max_seq_len=model_config.max_model_len,
        total_bits=total_bits,
        device="cuda",
        model_name=getattr(model_config, 'model_type', None),
    )
    
    if hasattr(model_config, 'uses_mla') and model_config.uses_mla:
        config.uses_mla = True
        config.latent_dim = getattr(model_config, 'latent_dim', 576)
        config.qk_head_dim = getattr(model_config, 'qk_head_dim', 192)
        config.v_head_dim = getattr(model_config, 'v_head_dim', 256)
    
    return TurboQuantKVCache(config)
