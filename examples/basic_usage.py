"""
Example: Basic TurboQuant usage.

This example demonstrates:
1. Creating a TurboQuant engine
2. Compressing KV cache
3. Computing attention with compressed cache
4. Comparing memory usage
"""

import torch
from turboquant import TurboQuantEngine


def basic_example():
    """Basic usage example."""
    
    print("=" * 60)
    print("TurboQuant Basic Example")
    print("=" * 60)
    
    engine = TurboQuantEngine(
        head_dim=128,
        total_bits=3,
        device="cpu",
        gpu_arch="auto",
    )
    
    print(f"\nEngine: {engine}")
    print(f"Compression ratio: {engine.get_compression_ratio():.2f}x")
    
    seq_len = 1000
    num_heads = 8
    head_dim = 128
    
    K = torch.randn(seq_len, num_heads, head_dim)
    V = torch.randn(seq_len, num_heads, head_dim)
    Q = torch.randn(10, num_heads, head_dim)
    
    print(f"\nInput shapes:")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    print(f"  Q: {Q.shape}")
    
    print("\nCompressing KV cache...")
    compressed_k = engine.compress_keys(K)
    compressed_v = engine.compress_values(V)
    
    print(f"\nCompressed key components:")
    for key, tensor in compressed_k.items():
        print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    
    memory_stats = engine.get_memory_savings(seq_len, num_heads)
    print(f"\nMemory savings:")
    print(f"  FP16 bytes: {memory_stats['fp16_bytes']:,}")
    print(f"  Compressed bytes: {memory_stats['compressed_bytes']:,}")
    print(f"  Savings: {memory_stats['savings_bytes']:,} bytes ({memory_stats['savings_bytes'] / memory_stats['fp16_bytes'] * 100:.1f}%)")
    print(f"  Compression ratio: {memory_stats['compression_ratio']:.2f}x")
    
    print("\nComputing attention with compressed cache...")
    scores = engine.attention_scores(Q, compressed_k)
    print(f"  Attention scores shape: {scores.shape}")
    
    output = engine.fused_attention(Q, compressed_k, compressed_v)
    print(f"  Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def glm5_example():
    """GLM-5 MLA example."""
    
    print("\n" + "=" * 60)
    print("GLM-5 MLA Compression Example")
    print("=" * 60)
    
    from turboquant.models import GLM5Quantizer
    
    quantizer = GLM5Quantizer(
        device="cpu",
        latent_bits=3,
        indexer_bits=2,
    )
    
    print(f"\nQuantizer: {quantizer}")
    
    seq_len = 1000
    num_heads = 32
    
    latent_k = torch.randn(seq_len, 576)
    latent_v = torch.randn(seq_len, 576)
    indexer_k = torch.randn(seq_len, num_heads, 128)
    
    print(f"\nGLM-5 MLA shapes:")
    print(f"  Latent K: {latent_k.shape}")
    print(f"  Latent V: {latent_v.shape}")
    print(f"  Indexer K: {indexer_k.shape}")
    
    print("\nCompressing MLA latent KV...")
    compressed_latent = quantizer.compress_latent_kv(latent_k, latent_v)
    
    print("\nCompressing indexer...")
    compressed_indexer = quantizer.compress_indexer(indexer_k)
    
    stats = quantizer.get_compression_stats(seq_len, num_heads)
    print(f"\nGLM-5 compression statistics:")
    print(f"  Total FP16: {stats['total']['fp16_bytes']:,} bytes")
    print(f"  Total compressed: {stats['total']['compressed_bytes']:,} bytes")
    print(f"  Total compression ratio: {stats['total']['compression_ratio']:.2f}x")
    print(f"  Savings: {stats['total']['savings_percent']:.1f}%")
    
    print("\n" + "=" * 60)
    print("GLM-5 example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    basic_example()
    glm5_example()
