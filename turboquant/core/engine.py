"""
TurboQuant Engine - Main interface for KV cache compression.

Paper-compliant implementation with bit-packing for actual compression.
Achieves 5.33x compression for 3-bit, 4.0x for 4-bit.
"""

import torch
import math
from typing import Dict, Optional, Tuple, Literal

from .constants import (
    DEFAULT_TOTAL_BITS,
    DEFAULT_SEED,
    DEFAULT_EPSILON,
    GPU_ARCHITECTURES,
    MODEL_CONFIGS,
    get_gpu_architecture,
    get_model_config,
)
from .rotation import RandomRotationMatrix, QJLProjectionMatrix
from .codebook import LloydMaxCodebook
from .bitpacking import (
    pack_bits,
    unpack_bits,
    pack_signs,
    unpack_signs,
    calculate_packed_size,
    get_compression_ratio,
)


class TurboQuantEngine:
    """
    Universal TurboQuant engine for KV cache compression.
    
    Paper-compliant implementation with bit-packing.
    
    The algorithm:
    1. Keys: MSE quantization + QJL correction for unbiased attention
    2. Values: MSE quantization only (no QJL needed)
    
    Storage with bit-packing:
    - Indices: mse_bits per coordinate (PACKED)
    - QJL signs: 1 bit per coordinate (PACKED)
    - Norms: 2 bytes per token
    - Residual norms: 2 bytes per token
    
    Compression ratios:
    - 3-bit: 5.33x (2-bit MSE + 1-bit QJL)
    - 4-bit: 4.00x (3-bit MSE + 1-bit QJL)
    
    Args:
        head_dim: Dimension of attention heads (default: 128)
        total_bits: Total bits per coordinate (default: 3)
        seed: Random seed for rotation matrices
        device: Device for computations ('cuda', 'cpu')
        gpu_arch: GPU architecture ('blackwell', 'hopper', 'auto')
        model_name: Model name for auto-configuration ('llama3', 'glm5', etc.)
    """
    
    def __init__(
        self,
        head_dim: int = 128,
        total_bits: int = DEFAULT_TOTAL_BITS,
        seed: int = DEFAULT_SEED,
        device: str = "cuda",
        gpu_arch: str = "auto",
        model_name: Optional[str] = None,
    ):
        self.head_dim = head_dim
        self.total_bits = total_bits
        self.mse_bits = max(total_bits - 1, 1)
        self.qjl_bits = 1
        self.seed = seed
        self.device = device
        
        if gpu_arch == "auto":
            gpu_arch = get_gpu_architecture()
        self.gpu_arch = gpu_arch
        
        if model_name is not None:
            model_config = get_model_config(model_name)
            if model_config.uses_mla and model_config.latent_dim > 0:
                self.head_dim = model_config.latent_dim
        
        self.rotation = RandomRotationMatrix(self.head_dim, seed, device)
        self.projection = QJLProjectionMatrix(self.head_dim, seed, device)
        
        self.key_codebook = LloydMaxCodebook(self.head_dim, self.mse_bits, device=device)
        self.val_codebook = LloydMaxCodebook(self.head_dim, total_bits, device=device)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.correction_scale = math.sqrt(math.pi / 2) / self.head_dim
        
        self.compression_ratio = get_compression_ratio(self.mse_bits, self.qjl_bits)
    
    @torch.no_grad()
    def compress_keys(self, K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress key cache with MSE + QJL using bit-packing.
        
        Args:
            K: Key tensor of shape (seq_len, num_heads, head_dim) or (seq_len, head_dim)
            
        Returns:
            Dictionary containing:
                - indices_packed: Packed quantization indices
                - signs_packed: Packed QJL signs
                - vec_norms: L2 norms of original vectors (float16)
                - residual_norms: L2 norms of residuals (float16)
                - shape: Original shape for unpacking
        """
        original_shape = K.shape
        if K.dim() == 3:
            num_heads = K.shape[1]
            K = K.reshape(-1, self.head_dim)
        
        K_f = K.float()
        
        vec_norms = torch.norm(K_f, dim=-1, keepdim=True)
        K_normed = K_f / (vec_norms + DEFAULT_EPSILON)
        
        rotated = self.rotation.rotate(K_normed, transpose=True)
        
        indices = self.key_codebook.quantize(rotated)
        
        y_hat = self.key_codebook.dequantize(indices)
        k_mse_normed = self.rotation.rotate(y_hat, transpose=False)
        k_mse = k_mse_normed * vec_norms
        
        residual = K_f - k_mse
        residual_norms = torch.norm(residual, dim=-1)
        
        projected = self.projection.project(residual)
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0
        
        indices_packed = pack_bits(indices, self.mse_bits)
        signs_packed = pack_signs(qjl_signs)
        
        return {
            "indices_packed": indices_packed,
            "signs_packed": signs_packed,
            "vec_norms": vec_norms.squeeze(-1).half(),
            "residual_norms": residual_norms.half(),
            "shape": torch.tensor(original_shape),
        }
    
    @torch.no_grad()
    def compress_values(self, V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress value cache with MSE only using bit-packing.
        
        Args:
            V: Value tensor of shape (seq_len, num_heads, head_dim) or (seq_len, head_dim)
            
        Returns:
            Dictionary containing:
                - indices_packed: Packed quantization indices
                - vec_norms: L2 norms of original vectors (float16)
                - shape: Original shape for unpacking
        """
        original_shape = V.shape
        if V.dim() == 3:
            V = V.reshape(-1, self.head_dim)
        
        V_f = V.float()
        
        vec_norms = torch.norm(V_f, dim=-1, keepdim=True)
        V_normed = V_f / (vec_norms + DEFAULT_EPSILON)
        
        rotated = self.rotation.rotate(V_normed, transpose=True)
        
        indices = self.val_codebook.quantize(rotated)
        
        indices_packed = pack_bits(indices, self.total_bits)
        
        return {
            "indices_packed": indices_packed,
            "vec_norms": vec_norms.squeeze(-1).half(),
            "shape": torch.tensor(original_shape),
        }
    
    @torch.no_grad()
    def decompress_keys(self, compressed_k: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decompress key cache (MSE only, no QJL correction).
        
        Args:
            compressed_k: Dictionary from compress_keys()
            
        Returns:
            Reconstructed key tensor
        """
        indices_packed = compressed_k["indices_packed"]
        norms = compressed_k["vec_norms"]
        original_shape = tuple(compressed_k["shape"].tolist())
        
        num_values = norms.shape[0] * self.head_dim
        indices = unpack_bits(indices_packed, num_values, self.mse_bits)
        indices = indices.reshape(-1, self.head_dim)
        
        y_hat = self.key_codebook.dequantize(indices)
        
        # Normalize before unrotation to maintain unit sphere property
        y_hat_norm = torch.norm(y_hat, dim=-1, keepdim=True)
        y_hat_normalized = y_hat / (y_hat_norm + DEFAULT_EPSILON)
        
        reconstructed_normed = self.rotation.rotate(y_hat_normalized, transpose=False)
        reconstructed = reconstructed_normed * norms.float().unsqueeze(-1)
        
        result = reconstructed.half()
        if len(original_shape) == 3:
            result = result.reshape(original_shape)
        
        return result
    
    @torch.no_grad()
    def decompress_values(self, compressed_v: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decompress value cache.
        
        Args:
            compressed_v: Dictionary from compress_values()
            
        Returns:
            Reconstructed value tensor
        """
        indices_packed = compressed_v["indices_packed"]
        norms = compressed_v["vec_norms"]
        original_shape = tuple(compressed_v["shape"].tolist())
        
        num_values = norms.shape[0] * self.head_dim
        indices = unpack_bits(indices_packed, num_values, self.total_bits)
        indices = indices.reshape(-1, self.head_dim)
        
        y_hat = self.val_codebook.dequantize(indices)
        
        # Normalize before unrotation to maintain unit sphere property
        y_hat_norm = torch.norm(y_hat, dim=-1, keepdim=True)
        y_hat_normalized = y_hat / (y_hat_norm + DEFAULT_EPSILON)
        
        reconstructed_normed = self.rotation.rotate(y_hat_normalized, transpose=False)
        reconstructed = reconstructed_normed * norms.float().unsqueeze(-1)
        
        result = reconstructed.half()
        if len(original_shape) == 3:
            result = result.reshape(original_shape)
        
        return result
    
    @torch.no_grad()
    def attention_scores(
        self,
        Q: torch.Tensor,
        compressed_k: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention scores with QJL correction.
        
        score(q, k) = <q, k_mse> + ||r|| * sqrt(pi/2)/d * <S*q, signs>
        
        Args:
            Q: Query tensor of shape (seq_q, num_heads, head_dim) or (seq_q, head_dim)
            compressed_k: Dictionary from compress_keys()
            
        Returns:
            Attention scores of shape (seq_q, seq_k)
        """
        original_q_shape = Q.shape
        if Q.dim() == 3:
            num_heads = Q.shape[1]
            Q = Q.reshape(-1, self.head_dim)
        
        Q_f = Q.float()
        
        indices_packed = compressed_k["indices_packed"]
        signs_packed = compressed_k["signs_packed"]
        norms = compressed_k["vec_norms"].float()
        r_norms = compressed_k["residual_norms"].float()
        
        num_tokens = norms.shape[0]
        num_values = num_tokens * self.head_dim
        
        indices = unpack_bits(indices_packed, num_values, self.mse_bits)
        indices = indices.reshape(-1, self.head_dim)
        
        signs = unpack_signs(signs_packed, num_values)
        signs = signs.reshape(-1, self.head_dim)
        
        y_hat = self.key_codebook.dequantize(indices)
        
        # Normalize before unrotation to maintain unit sphere property
        y_hat_norm = torch.norm(y_hat, dim=-1, keepdim=True)
        y_hat_normalized = y_hat / (y_hat_norm + DEFAULT_EPSILON)
        
        k_mse_normed = self.rotation.rotate(y_hat_normalized, transpose=False)
        k_mse = k_mse_normed * norms.unsqueeze(-1)
        
        term1 = Q_f @ k_mse.T
        
        q_proj = self.projection.project(Q_f)
        qjl_ip = q_proj @ signs.T
        
        term2 = self.correction_scale * qjl_ip * r_norms.unsqueeze(0)
        
        scores = (term1 + term2) * self.scale
        
        return scores
    
    @torch.no_grad()
    def fused_attention(
        self,
        Q: torch.Tensor,
        compressed_k: Dict[str, torch.Tensor],
        compressed_v: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute full fused attention: scores -> softmax -> weighted V.
        
        Args:
            Q: Query tensor
            compressed_k: Compressed keys
            compressed_v: Compressed values
            
        Returns:
            Attention output
        """
        scores = self.attention_scores(Q, compressed_k)
        
        weights = torch.softmax(scores, dim=-1)
        
        V_recon = self.decompress_values(compressed_v).float()
        
        output = weights @ V_recon
        
        return output.half()
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio vs FP16."""
        return self.compression_ratio
    
    def get_memory_savings(self, seq_len: int, num_heads: int = 1) -> Dict[str, int | float]:
        """
        Calculate memory savings for given sequence length.
        
        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            
        Returns:
            Dictionary with memory statistics
        """
        fp16_bytes = seq_len * num_heads * self.head_dim * 2
        
        num_coords = seq_len * num_heads * self.head_dim
        indices_bytes = calculate_packed_size(num_coords, self.mse_bits)
        signs_bytes = calculate_packed_size(num_coords, self.qjl_bits)
        norms_bytes = seq_len * num_heads * 2
        rnorms_bytes = seq_len * num_heads * 2
        
        compressed_bytes = indices_bytes + signs_bytes + norms_bytes + rnorms_bytes
        
        return {
            "fp16_bytes": fp16_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": fp16_bytes / compressed_bytes,
            "savings_percent": (1 - compressed_bytes / fp16_bytes) * 100,
        }
    
    def to(self, device: str) -> "TurboQuantEngine":
        """Move engine to specified device."""
        self.device = device
        self.rotation = self.rotation.to(device)
        self.projection = self.projection.to(device)
        self.key_codebook = self.key_codebook.to(device)
        self.val_codebook = self.val_codebook.to(device)
        return self
    
    def __repr__(self) -> str:
        return (
            f"TurboQuantEngine(head_dim={self.head_dim}, "
            f"total_bits={self.total_bits}, "
            f"compression_ratio={self.compression_ratio:.2f}x, "
            f"device='{self.device}')"
        )
