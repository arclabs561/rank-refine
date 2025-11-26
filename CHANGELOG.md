# Changelog

## [0.7.4] - 2025-11-26

### Added
- MSRV 1.74 to Cargo.toml
- CI caching with `Swatinem/rust-cache`
- MSRV CI job
- Cross-link to rank-fusion in README

### Removed
- Unused `Embedding` and `TokenEmbeddings` newtypes (API uses raw slices)

### Fixed
- DESIGN.md trait signatures now match actual code

## [0.7.3] - 2025-11-26

### Added
- Complete pipeline example in lib.rs documentation
- `sort_scored_desc()` internal helper to deduplicate sorting logic
- Improved README with BYOM (Bring Your Own Model) explanation
- Features table in README

### Changed
- Clarified that tests use synthetic vectors (no model downloads required)
- Better module documentation emphasizing model-agnostic design

## [0.7.2] - 2025-11-26

### Added
- `pool_tokens_adaptive()`: Auto-selects pooling strategy based on factor
  - Factor 2-3: clustering (quality-focused)
  - Factor 4+: sequential (speed-focused)
- Pool factor guide in docs with storage/quality tradeoffs
- 26 new property tests (114 total)
- Additional SIMD property tests (Cauchy-Schwarz, norm scaling, bilinearity)
- Matryoshka property tests (alpha interpolation, config validation)

### Fixed
- Hierarchical pooling NaN handling (treats as max distance)
- Clippy `items_after_statements` lint in `cut_dendrogram`

## [0.7.1] - 2025-11-26

### Added
- `RefineConfig` builder with `with_alpha()`, `with_top_k()`, `refinement_only()`, `original_only()`
- `RefineError` enum for fallible operations (`InvalidHeadDims`, `EmptyQuery`, `DimensionMismatch`)
- `try_refine()` variants for all refinement functions
- `Embedding` and `TokenEmbeddings` newtypes for type safety
- `rank_with_top_k()` in colbert module
- `refine_with_config()` for full configuration control
- **`scoring` module**: Unified `Scorer` and `TokenScorer` traits for interoperability
  - `DenseScorer` (Dot, Cosine) for single-vector embeddings
  - `LateInteractionScorer` (MaxSimDot, MaxSimCosine) for multi-vector
  - `blend()` and `normalize_scores()` utilities
- Fuzz testing infrastructure (`fuzz/`) for SIMD operations
- Comprehensive property tests (20+ new tests) covering:
  - Token pooling quality (dimension preservation, score maintenance)
  - Score blending linearity
  - Normalization order preservation
  - Late interaction bounds
- Updated DESIGN.md with late interaction rationale and architecture diagram

### Changed
- Improved doc comments with usage examples and error conditions
- READMEs now include "When to Use What" guidance

## [0.7.0] - 2025-11-26

### Added
- **Optional `hierarchical` feature**: Uses [kodama](https://docs.rs/kodama) for proper
  hierarchical agglomerative clustering with Ward's method in token pooling.
  Matches scipy's quality, O(n²) complexity.
- Improved documentation matching kodama's attention to detail

### Changed
- Token pooling now has two implementations:
  - **With `hierarchical` feature**: kodama-based Ward's method clustering
  - **Default**: Greedy agglomerative clustering (zero dependencies)

## [0.6.0] - 2025-11-26

### Changed
- **Breaking**: `maxsim` and `maxsim_cosine` now return 0.0 for empty `doc_tokens` (was `NEG_INFINITY * query_len`)
- **Breaking**: Sorting uses `total_cmp` for deterministic NaN handling (NaN sorts first in descending order)
- **Breaking**: `dot_portable` is now `pub(crate)` instead of `pub`
- Removed unused `Config` struct from `crossencoder`
- `colbert::rank` no longer requires `Hash` bound (only `refine` needs it)

### Added
- **Token pooling** for `ColBERT`: `pool_tokens`, `pool_tokens_sequential`, `pool_tokens_with_protected`
  - Reduces storage 50-66% with minimal quality loss (pool factor 2-3)
  - Greedy agglomerative clustering (approximates hierarchical clustering)
  - Sequential windowed pooling for simpler use cases
  - Protected token support (preserves `[CLS]`/`[D]` markers)
- `#[must_use]` on all pure functions
- Property-based tests via proptest for all modules (22 new tests)
- Unit tests for NaN handling, empty docs, short embeddings, single candidates, pooling
- `matryoshka` now skips doc embeddings shorter than `head_dims` instead of panicking

### Fixed
- NaN scores no longer corrupt sort order (deterministic placement via `total_cmp`)
- Blending now uses `mul_add` for better floating-point precision
- All clippy pedantic warnings resolved
- Explicit SIMD intrinsic imports (no wildcards)

## [0.5.2] - 2025-11-26
- Simplified docs

## [0.5.1] - 2025-11-26
- CI setup, CHANGELOG

## [0.5.0] - 2025-11-26
- SAFETY docs for SIMD
- Edge case tests

## [0.4.0] - 2025-11-26
- SIMD, cross-encoder trait, benchmarks

## [0.3.0] - 2025-11-26
- ColBERT MaxSim

## [0.2.0] - 2025-11-26
- Matryoshka refinement

## [0.1.0] - 2025-11-26
- Initial stub
