"""
Lloyd-Max Scalar Quantization.

Implements optimal 1D quantization for the Beta-distributed coordinates
after random rotation. Works for any dimension d and bit-width b.

The key insight: after random rotation, coordinates follow a Beta distribution
with parameters determined by dimension d. Lloyd-Max finds optimal quantization
boundaries and centroids for this distribution.
"""

import torch
import math
from typing import Tuple, Optional
from scipy.special import gamma, betaln


def beta_pdf(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Compute Beta distribution PDF.
    
    Args:
        x: Points to evaluate (should be in [0, 1])
        alpha: First shape parameter
        beta: Second shape parameter
        
    Returns:
        PDF values at x
    """
    return (
        torch.pow(x, alpha - 1) 
        * torch.pow(1 - x, beta - 1) 
        / math.exp(betaln(alpha, beta))
    )


def compute_beta_distribution_params(dim: int) -> Tuple[float, float]:
    """
    Compute Beta distribution parameters for rotated coordinates.
    
    After random rotation, each coordinate follows Beta((d-1)/2, (d-1)/2)
    scaled to [-1, 1].
    
    Args:
        dim: Dimension of the original vectors
        
    Returns:
        (alpha, beta) parameters for Beta distribution
    """
    alpha = (dim - 1) / 2
    beta = (dim - 1) / 2
    return alpha, beta


class LloydMaxCodebook:
    """
    Lloyd-Max optimal scalar quantizer for Beta-distributed coordinates.
    
    The algorithm:
    1. Compute Beta distribution for given dimension
    2. Initialize boundaries uniformly
    3. Iteratively update:
       - Centroids as weighted means of intervals
       - Boundaries as midpoints of adjacent centroids
    4. Converge when change is below threshold
    
    Args:
        dim: Dimension of vectors (determines Beta distribution)
        bits: Number of bits (2^bits quantization levels)
        num_iterations: Max Lloyd-Max iterations
        tolerance: Convergence threshold
        device: Device for computations
    """
    
    def __init__(
        self,
        dim: int,
        bits: int,
        num_iterations: int = 100,
        tolerance: float = 1e-6,
        device: str = "cpu",
    ):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        if dim < 200:
            print(f"Warning: dim={dim} < 200, Beta→Gaussian convergence may be weak")
        
        self.dim = dim
        self.bits = bits
        self.num_levels = 2 ** bits
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.device = device
        
        self.alpha, self.beta = compute_beta_distribution_params(dim)
        
        self.centroids, self.boundaries = self._solve_lloyd_max()
    
    def _solve_lloyd_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve Lloyd-Max optimization.
        
        Returns:
            centroids: Quantization centroids (num_levels,)
            boundaries: Quantization boundaries (num_levels + 1,)
        """
        num_levels = self.num_levels
        
        boundaries = torch.linspace(-1, 1, num_levels + 1, device=self.device)
        centroids = torch.zeros(num_levels, device=self.device)
        
        for iteration in range(self.num_iterations):
            new_centroids = self._compute_centroids(boundaries)
            
            new_boundaries = torch.zeros(num_levels + 1, device=self.device)
            new_boundaries[0] = -1.0
            new_boundaries[-1] = 1.0
            for i in range(1, num_levels):
                new_boundaries[i] = (new_centroids[i-1] + new_centroids[i]) / 2
            
            centroid_change = torch.abs(new_centroids - centroids).max().item()
            
            centroids = new_centroids
            boundaries = new_boundaries
            
            if centroid_change < self.tolerance:
                break
        
        return centroids, boundaries
    
    def _compute_centroids(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Compute centroids as weighted means within each interval.
        
        Uses numerical integration to compute:
        c_i = integral_{b_i}^{b_{i+1}} x * p(x) dx / integral_{b_i}^{b_{i+1}} p(x) dx
        
        Args:
            boundaries: Quantization boundaries
            
        Returns:
            Centroids for each quantization level
        """
        num_samples = 1000
        num_levels = self.num_levels
        
        centroids = torch.zeros(num_levels, device=self.device)
        
        for i in range(num_levels):
            lower = boundaries[i].item()
            upper = boundaries[i + 1].item()
            
            x = torch.linspace(lower, upper, num_samples, device=self.device)
            
            x_scaled = (x + 1) / 2
            x_scaled = torch.clamp(x_scaled, 1e-10, 1 - 1e-10)
            
            pdf_vals = beta_pdf(x_scaled, self.alpha, self.beta)
            
            numerator = torch.trapz(x * pdf_vals, x)
            denominator = torch.trapz(pdf_vals, x)
            
            if denominator > 1e-10:
                centroids[i] = numerator / denominator
            else:
                centroids[i] = (lower + upper) / 2
        
        return centroids
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            indices: Quantization indices of shape (...,)
        """
        original_shape = x.shape
        x_flat = x.flatten()
        
        indices = torch.zeros_like(x_flat, dtype=torch.long)
        
        for i in range(self.num_levels - 1):
            mask = x_flat > self.boundaries[i + 1]
            indices[mask] = i + 1
        
        return indices.reshape(original_shape)
    
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize indices back to values.
        
        Args:
            indices: Quantization indices of shape (...,)
            
        Returns:
            Reconstructed values of shape (...,)
        """
        return self.centroids[indices]
    
    def to(self, device: str) -> "LloydMaxCodebook":
        """Move codebook to specified device."""
        self.device = device
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)
        return self
    
    @property
    def distortion(self) -> float:
        """
        Compute theoretical MSE distortion.
        
        From the paper: D ≈ 2.72 / 4^bits
        
        Returns:
            Expected MSE distortion
        """
        return 2.72 / (4 ** self.bits)
    
    @property
    def compression_ratio(self) -> float:
        """
        Compute compression ratio vs FP16.
        
        Returns:
            Compression ratio (e.g., 5.33 for 3-bit)
        """
        return 16.0 / self.bits
    
    def __repr__(self) -> str:
        return (
            f"LloydMaxCodebook(dim={self.dim}, bits={self.bits}, "
            f"levels={self.num_levels}, distortion={self.distortion:.4f})"
        )


def solve_lloyd_max(
    dim: int,
    bits: int,
    num_iterations: int = 100,
    tolerance: float = 1e-6,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to solve Lloyd-Max.
    
    Args:
        dim: Dimension of vectors
        bits: Number of bits
        num_iterations: Max iterations
        tolerance: Convergence threshold
        device: Device for computations
        
    Returns:
        (centroids, boundaries) tensors
    """
    codebook = LloydMaxCodebook(dim, bits, num_iterations, tolerance, device)
    return codebook.centroids, codebook.boundaries
