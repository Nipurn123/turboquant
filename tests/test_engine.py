"""Tests for TurboQuant core engine."""

import torch
import pytest
from turboquant.core.engine import TurboQuantEngine
from turboquant.core.codebook import LloydMaxCodebook


class TestLloydMaxCodebook:
    """Test Lloyd-Max quantization."""
    
    def test_codebook_creation(self):
        """Test codebook initialization."""
        codebook = LloydMaxCodebook(dim=128, bits=3)
        
        assert codebook.num_levels == 8
        assert codebook.centroids.shape == (8,)
        assert codebook.boundaries.shape == (9,)
    
    def test_quantize_dequantize(self):
        """Test quantization and dequantization."""
        codebook = LloydMaxCodebook(dim=128, bits=3, device="cpu")
        
        x = torch.randn(100, 128)
        x_normed = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        
        indices = codebook.quantize(x_normed)
        reconstructed = codebook.dequantize(indices)
        
        assert indices.shape == (100, 128)
        assert reconstructed.shape == (100, 128)
        assert indices.dtype == torch.long
    
    def test_distortion_formula(self):
        """Test theoretical distortion formula."""
        for bits in [1, 2, 3, 4]:
            codebook = LloydMaxCodebook(dim=128, bits=bits)
            expected_distortion = 2.72 / (4 ** bits)
            assert abs(codebook.distortion - expected_distortion) < 0.01
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        for bits in [1, 2, 3, 4]:
            codebook = LloydMaxCodebook(dim=128, bits=bits)
            expected_ratio = 16.0 / bits
            assert abs(codebook.compression_ratio - expected_ratio) < 0.01


class TestTurboQuantEngine:
    """Test TurboQuant engine."""
    
    @pytest.fixture
    def engine(self):
        """Create test engine."""
        return TurboQuantEngine(
            head_dim=128,
            total_bits=3,
            device="cpu",
        )
    
    def test_engine_creation(self, engine):
        """Test engine initialization."""
        assert engine.head_dim == 128
        assert engine.total_bits == 3
        assert engine.mse_bits == 2
        assert engine.gpu_arch in ["blackwell", "hopper", "auto", "cpu"]
    
    def test_compress_keys(self, engine):
        """Test key compression."""
        K = torch.randn(100, 8, 128)
        
        compressed = engine.compress_keys(K)
        
        assert "indices" in compressed
        assert "k_mse" in compressed
        assert "qjl_signs" in compressed
        assert "vec_norms" in compressed
        assert "residual_norms" in compressed
        
        assert compressed["indices"].dtype == torch.uint8
        assert compressed["qjl_signs"].dtype == torch.int8
    
    def test_compress_values(self, engine):
        """Test value compression."""
        V = torch.randn(100, 8, 128)
        
        compressed = engine.compress_values(V)
        
        assert "indices" in compressed
        assert "vec_norms" in compressed
        
        assert compressed["indices"].dtype == torch.uint8
    
    def test_decompress_values(self, engine):
        """Test value decompression."""
        V = torch.randn(100, 128)
        
        compressed = engine.compress_values(V)
        reconstructed = engine.decompress_values(compressed)
        
        assert reconstructed.shape == V.shape
        
        mse = torch.mean((V - reconstructed) ** 2).item()
        assert mse < 3.0
    
    def test_attention_scores(self, engine):
        """Test attention score computation."""
        Q = torch.randn(10, 8, 128)
        K = torch.randn(100, 8, 128)
        
        compressed_k = engine.compress_keys(K)
        scores = engine.attention_scores(Q, compressed_k)
        
        assert scores.shape == (80, 800)
    
    def test_fused_attention(self, engine):
        """Test fused attention."""
        Q = torch.randn(10, 8, 128)
        K = torch.randn(100, 8, 128)
        V = torch.randn(100, 8, 128)
        
        compressed_k = engine.compress_keys(K)
        compressed_v = engine.compress_values(V)
        
        output = engine.fused_attention(Q, compressed_k, compressed_v)
        
        assert output.shape == (80, 128)
    
    def test_memory_savings(self, engine):
        """Test memory savings calculation."""
        stats = engine.get_memory_savings(seq_len=1000, num_heads=8)
        
        assert "fp16_bytes" in stats
        assert "compressed_bytes" in stats
        assert "savings_bytes" in stats
        assert "compression_ratio" in stats
        
        assert stats["fp16_bytes"] > stats["compressed_bytes"]
        assert stats["compression_ratio"] > 3.0
    
    def test_custom_head_dim(self):
        """Test engine with custom head dimension (GLM-5 MLA)."""
        engine = TurboQuantEngine(
            head_dim=576,
            total_bits=3,
            device="cpu",
        )
        
        K = torch.randn(100, 576)
        compressed = engine.compress_keys(K)
        
        assert compressed["k_mse"].shape == (100, 576)


class TestRotationMatrix:
    """Test random rotation matrices."""
    
    def test_rotation_orthogonality(self):
        """Test that rotation matrices are orthogonal."""
        from turboquant.core.rotation import RandomRotationMatrix
        
        rot = RandomRotationMatrix(dim=128, seed=42, device="cpu")
        
        product = rot.matrix @ rot.matrix.T
        identity = torch.eye(128)
        
        assert torch.allclose(product, identity, atol=1e-5)
    
    def test_rotation_determinant(self):
        """Test that rotation matrices have det=±1."""
        from turboquant.core.rotation import RandomRotationMatrix
        
        rot = RandomRotationMatrix(dim=128, seed=42, device="cpu")
        
        det = torch.linalg.det(rot.matrix)
        assert abs(abs(det.item()) - 1.0) < 1e-5
    
    def test_rotation_reproducibility(self):
        """Test that same seed gives same rotation."""
        from turboquant.core.rotation import RandomRotationMatrix
        
        rot1 = RandomRotationMatrix(dim=128, seed=42, device="cpu")
        rot2 = RandomRotationMatrix(dim=128, seed=42, device="cpu")
        
        assert torch.allclose(rot1.matrix, rot2.matrix)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
