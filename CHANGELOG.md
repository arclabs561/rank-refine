# Changelog

## [0.5.0] - 2024-11-26

### Fixed
- Added SAFETY comments to all unsafe SIMD code
- Removed overstated performance claims from README/docs
- Documented silent failure modes (candidates dropped if not in docs)

### Added
- `test_dot_simd_vs_portable` - verifies SIMD matches portable
- Edge case tests for empty inputs, mismatched lengths
- `test_maxsim_empty_query`, `test_maxsim_empty_doc`
- `test_head_dims_too_large` panic test

## [0.4.0] - 2024-11-26

### Added
- SIMD acceleration (AVX2+FMA on x86_64, NEON on aarch64)
- `crossencoder` module with `CrossEncoderModel` trait
- Benchmarks via criterion

## [0.3.0] - 2024-11-26

### Added
- `colbert` module with MaxSim scoring
- `simd` module with portable vector ops

## [0.2.0] - 2024-11-26

### Added
- `matryoshka` module for MRL tail refinement

## [0.1.0] - 2024-11-26

### Added
- Initial release (stub)

