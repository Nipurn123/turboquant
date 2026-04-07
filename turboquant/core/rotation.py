"""
Random Rotation Matrix Generation.

Implements Haar-distributed random orthogonal matrices via QR decomposition.
Works for any dimension d (not hardcoded to 128).
"""

import torch
import math
from typing import Optional


class RandomRotationMatrix:
    """
    Generate and manage random rotation matrices for TurboQuant.
    
    The rotation matrix Π is generated via QR decomposition of a random
    Gaussian matrix, ensuring Haar-distributed random orthogonality.
    
    Args:
        dim: Dimension of the rotation matrix (head_dim or latent_dim)
        seed: Random seed for reproducibility
        device: Device to place the matrix on ('cpu', 'cuda', etc.)
        dtype: Data type for the matrix (default: float32)
    """
    
    def __init__(
        self,
        dim: int,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.seed = seed
        self.device = device
        self.dtype = dtype
        
        self.matrix = self._generate_rotation_matrix()
        self.matrix_T = self.matrix.T.contiguous()
    
    def _generate_rotation_matrix(self) -> torch.Tensor:
        """
        Generate Haar-distributed random orthogonal matrix.
        
        Algorithm:
        1. Sample G ~ N(0, I) from standard normal (dim x dim)
        2. Compute QR decomposition: G = QR
        3. Adjust signs: Q * diag(sign(diag(R))) to ensure uniform distribution
        
        Returns:
            Orthogonal matrix of shape (dim, dim)
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        
        G = torch.randn(self.dim, self.dim, generator=generator, dtype=torch.float64)
        
        Q, R = torch.linalg.qr(G)
        
        diag_R = torch.diag(R)
        diag_sign = torch.sign(diag_R)
        diag_sign[diag_sign == 0] = 1.0
        
        Pi = Q * diag_sign.unsqueeze(0)
        
        Pi = Pi.to(dtype=self.dtype, device=self.device)
        
        return Pi
    
    def rotate(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """
        Apply rotation to input tensor.
        
        Args:
            x: Input tensor of shape (..., dim) or (..., seq_len, dim)
            transpose: If True, apply Π^T instead of Π
            
        Returns:
            Rotated tensor of same shape as input
        """
        if x.dim() == 1:
            if transpose:
                return self.matrix_T @ x
            else:
                return self.matrix @ x
        elif x.dim() == 2:
            if transpose:
                return x @ self.matrix_T
            else:
                return x @ self.matrix
        else:
            if transpose:
                return torch.matmul(x, self.matrix_T)
            else:
                return torch.matmul(x, self.matrix)
    
    def to(self, device: str, dtype: Optional[torch.dtype] = None) -> "RandomRotationMatrix":
        """Move matrices to specified device/dtype."""
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        self.matrix = self.matrix.to(device=device, dtype=self.dtype)
        self.matrix_T = self.matrix_T.to(device=device, dtype=self.dtype)
        return self
    
    @property
    def shape(self) -> tuple:
        """Return shape of rotation matrix."""
        return self.matrix.shape
    
    def __repr__(self) -> str:
        return (
            f"RandomRotationMatrix(dim={self.dim}, seed={self.seed}, "
            f"device='{self.device}', dtype={self.dtype})"
        )


class QJLProjectionMatrix:
    """
    Random projection matrix for QJL (Quantized Johnson-Lindenstrauss) correction.
    
    Used for the second stage of key compression to achieve unbiased attention scores.
    
    Args:
        dim: Dimension of the projection matrix
        seed: Random seed (should differ from rotation matrix seed)
        device: Device to place the matrix on
        dtype: Data type for the matrix
    """
    
    def __init__(
        self,
        dim: int,
        seed: int = 42,
        device: str = "cpu", 
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.seed = seed + 10000
        self.device = device
        self.dtype = dtype
        
        self.matrix = self._generate_projection_matrix()
        self.matrix_T = self.matrix.T.contiguous()
    
    def _generate_projection_matrix(self) -> torch.Tensor:
        """
        Generate random Gaussian projection matrix for QJL.
        
        Uses JL-lemma normalization: entries ~ N(0, 1/d) to preserve norms.
        
        Returns:
            Projection matrix of shape (dim, dim)
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        
        S = torch.randn(self.dim, self.dim, generator=generator, dtype=torch.float64)
        
        # JL normalization: scale by 1/sqrt(d) to preserve norms
        S = S / math.sqrt(self.dim)
        
        S = S.to(dtype=self.dtype, device=self.device)
        
        return S
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply projection to input tensor.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Projected tensor of same shape
        """
        if x.dim() == 1:
            return self.matrix @ x
        elif x.dim() == 2:
            return x @ self.matrix_T
        else:
            return torch.matmul(x, self.matrix_T)
    
    def to(self, device: str, dtype: Optional[torch.dtype] = None) -> "QJLProjectionMatrix":
        """Move matrix to specified device/dtype."""
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        self.matrix = self.matrix.to(device=device, dtype=self.dtype)
        self.matrix_T = self.matrix_T.to(device=device, dtype=self.dtype)
        return self
    
    def __repr__(self) -> str:
        return f"QJLProjectionMatrix(dim={self.dim}, device='{self.device}')"
