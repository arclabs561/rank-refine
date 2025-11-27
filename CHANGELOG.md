# Changelog

## [0.7.28] - 2025-11-27

### Added
- **Token importance weighting** for MaxSim (based on [arXiv:2511.16106](https://arxiv.org/abs/2511.16106))
  - `simd::maxsim_weighted(&query, &doc, &weights)` — weighted dot-product MaxSim
  - `simd::maxsim_cosine_weighted` — weighted cosine MaxSim
  - `simd::maxsim_weighted_vecs` — convenience wrapper for owned vectors
  - `LateInteractionScorer::score_weighted` — trait method for weighted scoring
  - Formula: `score = Σᵢ wᵢ × maxⱼ(Qᵢ · Dⱼ)` where wᵢ is query token importance
  - Use case: IDF weighting, learned importance, rare term boosting
- 12 new tests for weighted MaxSim (unit + property tests)

## [0.7.27] - 2025-11-27

### Changed
- **Breaking**: `RefineConfig::with_alpha` now clamps to \[0, 1\] (was unchecked)
- **Breaking**: `MmrConfig::new` and `with_lambda` now clamp to \[0, 1\] (was unchecked)
- Simplified prelude: exports only core functions (`cosine`, `dot`, `maxsim`, `norm`, `pool_tokens`, `colbert_rank`, `mmr`, `MmrConfig`, `CrossEncoderModel`)
- Simplified lib.rs documentation

### Added
- Tests for parameter clamping (`refine_config_clamps_alpha`, `mmr_config_clamps_lambda`)

## [0.7.26] - 2025-11-27

### Added
- Concise examples for common use cases:
  - `rag_rerank.rs`: RAG chunk reranking
  - `search_diversity.rs`: MMR λ parameter demo
  - `colbert_pooling.rs`: Token compression
- README examples section

## [0.7.25] - 2025-11-27

### Added
- `TokenIndex`: Pre-computed token embeddings for repeated scoring
  - `score_all`, `top_k`, `rank` methods
  - Property tests for ordering, bounds, finite scores
- Non-commutativity warning in `maxsim` documentation
- MMR benchmark (`bench_mmr`)
- Integration tests: `e2e_clustering_pooling`, `e2e_mmr_diversity`

### Changed
- CI now tests clippy and docs with `--all-features`

### Removed
- Redundant `from_iter` method (use `FromIterator` trait instead)

## [0.7.24] - 2025-11-27

### Added
- Token importance weighting (Nov 2025) documented as future direction
- MMR lambda parameter guide with research citations
- Subnormal float performance warning in SIMD docs
- VRSD (2024) lambda-free diversity reference

### Fixed
- NaN ordering in REFERENCE.md (NaN sorts last, not first)

## [0.7.23] - 2025-11-27

### Changed
- Documentation refined based on research review:
  - Pool factor quality loss numbers aligned with Clavié et al. (2024)
  - Corrected complexity: greedy pooling is O(n^3 d), not O(n)
  - Added ColBERTv2 and 2D Matryoshka references
  - Documented protected tokens in DESIGN.md
  - Added 2D Matryoshka as future direction in REFERENCE.md
  - Mermaid diagrams and LaTeX math for GitHub rendering

## [0.7.21] - 2025-11-27

### Added
- Property tests for `diversity` module (10 new tests)
- `try_mmr_invalid_matrix` unit test

### Fixed
- Removed unused helper functions in diversity proptests

## [0.7.20] - 2025-11-27

### Added
- `try_mmr`: Fallible version of MMR that returns `Result` instead of panicking
- Complexity docs (`# Complexity` sections) for key functions
- Behavior notes for SIMD functions on mismatched vector lengths

### Changed
- Improved documentation with `# Complexity` and `# Note` sections

## [0.7.19] - 2025-11-27

### Added
- **`diversity` module**: MMR (Maximal Marginal Relevance) for diversity-aware reranking
  - `mmr`: MMR with precomputed similarity matrix
  - `mmr_cosine`: MMR with on-the-fly cosine similarity
  - `MmrConfig`: Builder pattern for λ (relevance-diversity tradeoff) and k
  - 7 unit tests for pure relevance, pure diversity, mixed, edge cases
- Updated prelude with `mmr`, `mmr_cosine`, `MmrConfig` exports

## [0.7.18] - 2025-11-26

### Changed
- **Simplified `embedding` module**: Removed `QueryEmbed`/`DocEmbed` types — instruction
  prefixes are the encoding model's concern, not this crate's. By the time embeddings
  reach here, they're just `&[f32]`.
- Kept `Normalized` (unit norm guarantee where dot=cosine) and `MaskedTokens` (batched
  late interaction with padding)
- Updated DESIGN.md to clarify scope: this crate scores embeddings, doesn't train them

## [0.7.17] - 2025-11-26

### Added
- **`embedding` module**: Type-safe embedding wrappers
  - `Normalized`: Guarantees unit L2 norm (dot = cosine)
  - `MaskedTokens`: Variable-length token batches with padding mask
  - `maxsim_masked`: MaxSim with mask support for batched processing
- `maxsim_batch` and `maxsim_cosine_batch`: Score one query against multiple documents

## [0.7.16] - 2025-11-26

### Changed
- **Reference implementation**: Comprehensive mathematical documentation for all traits
- `Scorer` trait: Documented symmetry, boundedness, scale invariants
- `TokenScorer` trait: Documented MaxSim asymmetry, complexity, degenerate cases
- `CrossEncoderModel` trait: Documented joint encoding semantics
- `Pooler` trait: Documented dimensionality preservation, cardinality invariants
- Module-level docs with taxonomy table and cascade diagram

## [0.7.15] - 2025-11-26

### Changed
- Clarified indexing vs query-time phases in README and DESIGN.md
- Added ASCII workflow diagram for late interaction pipeline
- Documented tricks from PyLate, ColPali implementations
- Added "When to Use Which Algorithm" indexing/query explanation

## [0.7.14] - 2025-11-26

### Changed
- Added "Why This Library?" section to README with research citations
- Expanded DESIGN.md with architectural motivation (DynamicRAG 2025)
- Added implementation comparison with qdrant, vindicator, fast-plaid
- Documented tricks gleaned from production codebases

## [0.7.13] - 2025-11-26

### Changed
- SIMD dispatch now skips vectors < 16 elements (matches qdrant threshold)
- Small vectors use portable fallback to avoid SIMD call overhead

### Added
- Comprehensive GitHub code comparison in DESIGN.md

## [0.7.12] - 2025-11-26

### Changed
- Updated DESIGN.md with academic citations (Clavié 2024, Wang 2024)
- Added clustering method comparison table
- Documented 2D Matryoshka as future direction
- Added references to fastembed-rs and ort for production usage

## [0.7.11] - 2025-11-26

### Added
- `FnPooler` for custom pooling strategies via closures
- Exported `FnPooler` in prelude
- Tests for custom pooling implementations

## [0.7.10] - 2025-11-26

### Added
- Mutation-killing unit tests in `simd` and `colbert` modules
- Tests for exact mathematical properties (cosine zero-norm, MaxSim sum-of-max)
- Tests for pooling count accuracy and refine alpha behavior

## [0.7.9] - 2025-11-26

### Fixed
- Escaped `[0,1]` in doc comments to fix rustdoc warnings

## [0.7.8] - 2025-11-26

### Changed
- Shorter `crossencoder` module and function docs
- Removed redundant `# Arguments` sections

## [0.7.7] - 2025-11-26

### Changed
- Simplified module documentation (less verbose, more direct)
- Cleaner error messages in `RefineError`
- Streamlined `RefineConfig` docs
- Reduced prelude exports to essentials
- Shorter trait documentation in `scoring` module

## [0.7.6] - 2025-11-26

### Added
- **Improved README**: Clear e2e guidance with embedding source table (fastembed, candle, ort)
- **Full fastembed example**: Shows real-world integration pattern
- **8 integration tests**: Realistic e2e workflows:
  - `e2e_two_stage_dense_then_colbert`: Dense → ColBERT pipeline
  - `e2e_matryoshka_refinement`: MRL head/tail workflow
  - `e2e_token_pooling_storage_workflow`: Pooling for storage efficiency
  - `e2e_cross_encoder_rerank`: Cross-encoder trait implementation
  - `e2e_hybrid_scoring_with_normalization`: Score normalization + blending
  - `e2e_full_pipeline_with_config`: Complete pipeline with RefineConfig
  - `e2e_trait_based_scoring`: Polymorphic scoring via traits
  - `e2e_edge_cases`: Empty inputs, single tokens, pooling edge cases

### Changed
- README now clearly explains "Bring Your Own Model" design
- Added embedding source comparison table

## [0.7.5] - 2025-11-26

### Added
- **`as_slices` helper**: Convert `&[Vec<f32>]` to `Vec<&[f32]>` ergonomically
- **`simd::maxsim_vecs` / `maxsim_cosine_vecs`**: Score owned token vectors directly
- **`TokenScorer::score_vecs` / `rank_vecs`**: Trait methods for owned vectors
- **`prelude` module**: Convenient re-exports for common imports
- 4 new tests for convenience functions

### Changed
- Updated quick start docs to showcase new convenience APIs
- Internal code now uses `as_slices` consistently (no manual conversions)

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
