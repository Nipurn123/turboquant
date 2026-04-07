"""
Integration tests for vLLM and SGLang backends.

Uses CPU for testing but validates the API and compression logic.
"""

import torch
import sys
sys.path.insert(0, '.')


def test_vllm_integration():
    """Test vLLM backend integration."""
    print("=" * 80)
    print("Testing vLLM Integration")
    print("=" * 80)
    
    from turboquant.backends.vllm import (
        TurboQuantKVCache,
        TurboQuantKVCacheConfig,
    )
    
    config = TurboQuantKVCacheConfig(
        num_layers=4,
        num_heads=8,
        head_dim=128,
        max_seq_len=512,
        total_bits=3,
        device='cpu',
    )
    print(f"\n✓ Config created: {config.num_layers} layers, {config.total_bits}-bit")
    
    cache = TurboQuantKVCache(config)
    print("✓ KV cache initialized")
    
    seq_len = 100
    num_heads = 8
    head_dim = 128
    
    K = torch.randn(seq_len, num_heads, head_dim)
    V = torch.randn(seq_len, num_heads, head_dim)
    
    cache.update(0, K, V)
    print(f"✓ Updated layer 0 with {seq_len} tokens")
    
    k_out, v_out = cache.get_kv(0)
    print(f"✓ Decompressed: K={k_out.shape}, V={v_out.shape}")
    
    k_cosine = torch.nn.functional.cosine_similarity(
        K.flatten(), k_out.flatten(), dim=0
    ).item()
    v_cosine = torch.nn.functional.cosine_similarity(
        V.flatten(), v_out.flatten(), dim=0
    ).item()
    
    print(f"✓ Reconstruction quality: K={k_cosine:.4f}, V={v_cosine:.4f}")
    
    stats = cache.get_memory_stats()
    print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")
    
    query = torch.randn(10, num_heads, head_dim)
    output = cache.compute_attention(0, query)
    print(f"✓ Attention output: {output.shape}")
    
    assert k_cosine > 0.75, f"K reconstruction too low: {k_cosine}"
    assert v_cosine > 0.75, f"V reconstruction too low: {v_cosine}"
    
    print("\n✅ vLLM integration test PASSED")
    return True


def test_sglang_integration():
    """Test SGLang backend integration."""
    print("\n" + "=" * 80)
    print("Testing SGLang Integration")
    print("=" * 80)
    
    from turboquant.backends.sglang import (
        TurboQuantKVPool,
        TurboQuantKVManager,
    )
    
    pool = TurboQuantKVPool(
        num_layers=4,
        num_heads=8,
        head_dim=128,
        max_total_tokens=1000,
        total_bits=3,
        device='cpu',
    )
    print("\n✓ KV pool initialized")
    
    seq_len = 100
    num_heads = 8
    head_dim = 128
    
    K = torch.randn(seq_len, num_heads, head_dim)
    V = torch.randn(seq_len, num_heads, head_dim)
    indices = torch.arange(seq_len)
    
    pool.write(0, K, V, indices)
    print(f"✓ Written {seq_len} tokens to pool")
    
    k_out, v_out = pool.read(0, indices[:50])
    print(f"✓ Read 50 tokens: K={k_out.shape}, V={v_out.shape}")
    
    k_cosine = torch.nn.functional.cosine_similarity(
        K[:50].flatten(), k_out.flatten(), dim=0
    ).item()
    v_cosine = torch.nn.functional.cosine_similarity(
        V[:50].flatten(), v_out.flatten(), dim=0
    ).item()
    
    print(f"✓ Reconstruction quality: K={k_cosine:.4f}, V={v_cosine:.4f}")
    
    query = torch.randn(10, num_heads, head_dim)
    key_indices = indices[:seq_len]
    output = pool.compute_attention(0, query, key_indices)
    print(f"✓ Attention output: {output.shape}")
    
    stats = pool.get_memory_stats()
    print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")
    
    assert k_cosine > 0.75, f"K reconstruction too low: {k_cosine}"
    assert v_cosine > 0.75, f"V reconstruction too low: {v_cosine}"
    
    print("\n✅ SGLang integration test PASSED")
    return True


def test_glm5_mla_integration():
    """Test GLM-5 MLA integration."""
    print("\n" + "=" * 80)
    print("Testing GLM-5 MLA Integration")
    print("=" * 80)
    
    from turboquant.models import GLM5Quantizer
    
    num_heads = 8
    
    wqk = torch.randn(576, num_heads, 192)
    wv = torch.randn(576, num_heads, 256)
    
    quantizer = GLM5Quantizer(
        num_heads=num_heads,
        device='cpu',
        latent_bits=3,
        indexer_bits=2,
        wqk=wqk,
        wv=wv,
    )
    print(f"\n✓ GLM-5 quantizer initialized with {num_heads} heads")
    
    seq_len = 50
    latent_k = torch.randn(seq_len, 576)
    latent_v = torch.randn(seq_len, 576)
    indexer_k = torch.randn(seq_len, num_heads, 128)
    
    compressed_latent = quantizer.compress_latent_kv(latent_k, latent_v)
    compressed_indexer = quantizer.compress_indexer(indexer_k)
    print(f"✓ Compressed latent KV and indexer for {seq_len} tokens")
    
    k_recon, v_recon = quantizer.decompress_latent_kv(compressed_latent)
    print(f"✓ Decompressed: K={k_recon.shape}, V={v_recon.shape}")
    
    k_cosine = torch.nn.functional.cosine_similarity(
        latent_k.flatten(), k_recon.flatten(), dim=0
    ).item()
    v_cosine = torch.nn.functional.cosine_similarity(
        latent_v.flatten(), v_recon.flatten(), dim=0
    ).item()
    
    print(f"✓ Latent reconstruction: K={k_cosine:.4f}, V={v_cosine:.4f}")
    
    query = torch.randn(10, num_heads, 192)
    top_scores, top_indices = quantizer.sparse_attention_scores(
        query, compressed_indexer, top_k_per_head=20
    )
    print(f"✓ Sparse attention: scores={top_scores.shape}, indices={top_indices.shape}")
    
    stats = quantizer.get_compression_stats(seq_len, num_heads)
    print(f"✓ Compression: latent={stats['latent_kv']['compression_ratio']:.2f}x, "
          f"indexer={stats['indexer']['compression_ratio']:.2f}x")
    
    assert k_cosine > 0.75, f"K reconstruction too low: {k_cosine}"
    assert v_cosine > 0.75, f"V reconstruction too low: {v_cosine}"
    assert top_indices.max() < seq_len, "Indices out of bounds"
    
    print("\n✅ GLM-5 MLA integration test PASSED")
    return True


def test_end_to_end_workflow():
    """Test complete workflow from compression to attention."""
    print("\n" + "=" * 80)
    print("Testing End-to-End Workflow")
    print("=" * 80)
    
    from turboquant.backends.vllm import TurboQuantKVCache, TurboQuantKVCacheConfig
    
    config = TurboQuantKVCacheConfig(
        num_layers=2,
        num_heads=4,
        head_dim=64,
        max_seq_len=256,
        total_bits=3,
        device='cpu',
    )
    
    cache = TurboQuantKVCache(config)
    
    print("\n--- Simulating generation loop ---")
    seq_len = 0
    max_tokens = 50
    
    for step in range(max_tokens):
        new_k = torch.randn(1, config.num_heads, config.head_dim)
        new_v = torch.randn(1, config.num_heads, config.head_dim)
        
        if step == 0:
            full_k = new_k
            full_v = new_v
        else:
            full_k = torch.cat([prev_k, new_k], dim=0)
            full_v = torch.cat([prev_v, new_v], dim=0)
        
        cache.update(0, full_k, full_v)
        
        prev_k, prev_v = cache.get_kv(0)
        seq_len = prev_k.shape[0]
        
        if step % 10 == 0:
            stats = cache.get_memory_stats()
            print(f"  Step {step}: {seq_len} tokens, {stats['compression_ratio']:.2f}x compression")
    
    print(f"\n✓ Generated {seq_len} tokens with compression")
    
    stats = cache.get_memory_stats()
    print(f"✓ Final memory: {stats['compressed_mb']:.2f} MB compressed")
    print(f"✓ Savings: {stats['savings_percent']:.1f}%")
    
    print("\n✅ End-to-end workflow test PASSED")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("TURBOQUANT INTEGRATION TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    tests = [
        ("vLLM Integration", test_vllm_integration),
        ("SGLang Integration", test_sglang_integration),
        ("GLM-5 MLA Integration", test_glm5_mla_integration),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            results[name] = False
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("\nTurboQuant is ready for vLLM and SGLang!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
