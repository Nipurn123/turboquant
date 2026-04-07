# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- vLLM integration with `TurboQuantKVCache`
- SGLang integration with `TurboQuantKVPool` and `TurboQuantKVManager`
- GLM-5 Multi-Latent Attention (MLA) support with `GLM5Quantizer`
- Bit-packing for actual memory savings
- QJL correction for unbiased attention scores
- Comprehensive test suite with validation and integration tests
- Documentation with integration guide and API reference

### Changed

- Improved multi-head tensor handling in backends
- Fixed dtype mismatches in attention computation
- Enhanced reconstruction quality metrics

## [0.1.0] - 2024-01-15

### Added

- Initial release
- Core compression engine with Lloyd-Max quantization
- Random rotation for rotation-invariant distributions
- Support for 1-4 bit quantization
- GPU-optimized backends (PyTorch, Triton, cuTile)
- Basic documentation and examples

### Compression Performance

- 3-bit quantization: 5.33x compression ratio
- Reconstruction quality: 0.80+ cosine similarity
- Perplexity increase: <2%

[Unreleased]: https://github.com/anomaly/turboquant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anomaly/turboquant/releases/tag/v0.1.0
