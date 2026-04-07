"""
DIAGNOSTIC: Find the bugs in our implementation.

The validation shows we're NOT actually compressing - we're using 2x MORE memory.
Let's figure out why.
"""

import torch
import numpy as np
from turboquant.core.engine import TurboQuantEngine


def diagnose_compression():
    """Diagnose why we're not actually compressing."""
    print("=" * 80)
    print("DIAGNOSTIC 1: Why is compression ratio 0.5x (EXPANSION, not compression)?")
    print("=" * 80)
    
    engine = TurboQuantEngine(head_dim=512, total_bits=3, device="cpu")
    
    K = torch.randn(100, 512)
    compressed = engine.compress_keys(K)
    
    print("\nOriginal KV cache:")
    print(f"  FP16 size: {K.numel() * 2} bytes")
    
    print("\nCompressed representation:")
    total_bytes = 0
    for key, tensor in compressed.items():
        bytes_per_element = {
            torch.uint8: 1,
            torch.int8: 1,
            torch.float16: 2,
            torch.float32: 4,
        }.get(tensor.dtype, 2)
        
        size_bytes = tensor.numel() * bytes_per_element
        total_bytes += size_bytes
        
        print(f"  {key:20s}: {str(tensor.shape):30s} {str(tensor.dtype):12s} = {size_bytes:8d} bytes")
    
    print(f"\n  TOTAL: {total_bytes} bytes")
    print(f"  Ratio: {(K.numel() * 2) / total_bytes:.2f}x")
    
    print("\n" + "=" * 80)
    print("PROBLEM IDENTIFIED:")
    print("=" * 80)
    print("We're storing k_mse (MSE-reconstructed keys) in the compressed cache!")
    print("This is FULL PRECISION and defeats the purpose of compression.")
    print("\nThe paper DOES NOT store k_mse - it reconstructs it on-the-fly from:")
    print("  - indices (quantized)")
    print("  - norms")
    print("  - rotation matrix (shared across all tokens)")
    
    print("\nSOLUTION:")
    print("  1. Remove k_mse from compressed representation")
    print("  2. Reconstruct on-demand using: k_mse = Pi.rotate(codebook.dequantize(indices)) * norms")
    print("  3. This requires the rotation matrix and codebook to be available at inference")


def diagnose_attention():
    """Diagnose why attention accuracy is so poor."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: Why is attention accuracy only 1% (cosine sim 0.01)?")
    print("=" * 80)
    
    engine = TurboQuantEngine(head_dim=512, total_bits=3, device="cpu")
    
    # Simple test: single query, single key
    q = torch.randn(512)
    q = q / torch.norm(q)
    
    k = torch.randn(512)
    k = k / torch.norm(k)
    
    # True score
    true_score = torch.dot(q, k).item() / np.sqrt(512)
    
    # Compress
    compressed = engine.compress_keys(k.unsqueeze(0))
    
    # Check what we're storing
    k_mse = compressed["k_mse"][0].float()
    
    # MSE-only score
    mse_score = torch.dot(q, k_mse).item() / np.sqrt(512)
    
    # QJL-corrected score
    qjl_score = engine.attention_scores(q.unsqueeze(0), compressed)[0, 0].item()
    
    print(f"\nTrue score: {true_score:.6f}")
    print(f"MSE score:  {mse_score:.6f} (error: {abs(mse_score - true_score):.6f})")
    print(f"QJL score:  {qjl_score:.6f} (error: {abs(qjl_score - true_score):.6f})")
    
    # Check reconstruction quality
    k_reconstruction_error = torch.mean((k - k_mse) ** 2).item()
    k_cosine = torch.nn.functional.cosine_similarity(k.unsqueeze(0), k_mse.unsqueeze(0)).item()
    
    print(f"\nKey reconstruction quality:")
    print(f"  MSE: {k_reconstruction_error:.6f}")
    print(f"  Cosine similarity: {k_cosine:.6f}")
    
    print("\n" + "=" * 80)
    print("PROBLEM IDENTIFIED:")
    print("=" * 80)
    print("The key reconstruction quality is POOR.")
    print("This means the MSE quantization is not working correctly.")
    print("\nPossible issues:")
    print("  1. Lloyd-Max codebook not converging properly")
    print("  2. Rotation matrix not orthogonal")
    print("  3. Quantization applied to wrong distribution")
    print("  4. Normalization issues")


def diagnose_quantization():
    """Diagnose Lloyd-Max quantization."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: Is Lloyd-Max codebook correct?")
    print("=" * 80)
    
    from turboquant.core.codebook import LloydMaxCodebook
    from turboquant.core.rotation import RandomRotationMatrix
    
    dim = 512
    bits = 3
    
    # Create codebook
    codebook = LloydMaxCodebook(dim, bits)
    
    print(f"\nCodebook for {dim}-dim, {bits}-bit:")
    print(f"  Levels: {codebook.num_levels}")
    print(f"  Centroids: {codebook.centroids}")
    print(f"  Boundaries: {codebook.boundaries}")
    
    # Test quantization on known distribution
    # Generate random vectors on unit sphere
    X = torch.randn(10000, dim)
    X_normed = X / torch.norm(X, dim=-1, keepdim=True)
    
    # Apply rotation
    rot = RandomRotationMatrix(dim, seed=42, device="cpu")
    X_rotated = rot.rotate(X_normed, transpose=True)
    
    # Quantize first coordinate
    coords = X_rotated[:, 0]
    indices = codebook.quantize(coords.unsqueeze(-1)).squeeze(-1)
    reconstructed = codebook.dequantize(indices)
    
    # Compute MSE
    mse = torch.mean((coords - reconstructed) ** 2).item()
    
    print(f"\nQuantization test on rotated coordinates:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Theoretical distortion: {codebook.distortion:.6f}")
    
    # Check if rotation is orthogonal
    product = rot.matrix @ rot.matrix.T
    identity_error = torch.mean((product - torch.eye(dim)) ** 2).item()
    
    print(f"\nRotation matrix orthogonality:")
    print(f"  ||R @ R.T - I||²: {identity_error:.8f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("The quantization MSE should be close to theoretical distortion.")
    print("If it's much higher, there's a bug in the quantization logic.")


def diagnose_qjl():
    """Diagnose QJL correction."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: Is QJL projection correct?")
    print("=" * 80)
    
    from turboquant.core.rotation import QJLProjectionMatrix
    
    dim = 512
    num_samples = 1000
    
    # Create QJL projection matrix
    S = QJLProjectionMatrix(dim, seed=42, device="cpu")
    
    print(f"\nQJL projection matrix:")
    print(f"  Shape: {S.matrix.shape}")
    print(f"  Output dimension: {S.output_dim}")
    
    # Test: projection should preserve norms on average
    X = torch.randn(num_samples, dim)
    projected = S.project(X)
    
    original_norms = torch.norm(X, dim=-1)
    projected_norms = torch.norm(projected, dim=-1)
    
    norm_ratio = projected_norms / original_norms
    
    print(f"\nNorm preservation test:")
    print(f"  Mean norm ratio: {norm_ratio.mean().item():.4f}")
    print(f"  Std norm ratio: {norm_ratio.std().item():.4f}")
    print(f"  Expected: ~{np.sqrt(S.output_dim / dim):.4f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("QJL projects from d dimensions to output_dim.")
    print("Norms should scale by sqrt(output_dim / d).")


if __name__ == "__main__":
    diagnose_compression()
    diagnose_attention()
    diagnose_quantization()
    diagnose_qjl()
    
    print("\n" + "=" * 80)
    print("SUMMARY OF BUGS")
    print("=" * 80)
    print("\n1. CRITICAL: We're storing k_mse (full precision) in compressed cache")
    print("   → This defeats compression entirely")
    print("   → FIX: Remove k_mse, reconstruct on-demand")
    print("\n2. CRITICAL: Key reconstruction quality is poor")
    print("   → MSE quantization not working correctly")
    print("   → FIX: Debug Lloyd-Max convergence")
    print("\n3. MAJOR: Attention computation is completely wrong")
    print("   → QJL correction not effective")
    print("   → FIX: Check QJL projection and correction formula")
    print("\n" + "=" * 80)
