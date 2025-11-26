# rank-refine Design Document

## Context & Provenance

This crate is the second half of a two-crate strategy for Rust retrieval pipelines.
See `rank-fusion` for the first half (combining result lists).

## The Two-Crate Strategy

```
Retrieve (BM25, Dense) → Fuse (rank-fusion) → Rerank (rank-refine) → Top-K
```

### rank-fusion (sibling crate)
- Zero-dep rank fusion: RRF, CombSUM, CombMNZ, Borda, Weighted
- Published at: https://crates.io/crates/rank-fusion

### rank-refine (this crate)
- Model-based reranking and score refinement
- Heavier dependencies (candle, ort) but modular

## Planned Features

### 1. Cross-Encoder Reranking
- Load quantized reranking models (TinyBERT, MiniLM)
- Batch inference via candle or ort
- Trait-based backend abstraction

### 2. Matryoshka Tail Refinement
- Take coarse results (from first 64 dims)
- Refine using tail dimensions (64-768)
- SIMD-accelerated distance computation
- No model inference needed, just vector math

### 3. ColBERT/PLAID MaxSim
- Late interaction scoring
- SIMD `MaxSim` kernel: `score = Σ max(q_i · D)`
- Block-max optimization for cache efficiency

## Architecture

```rust
pub trait Refiner<I> {
    fn refine(&self, candidates: &[(I, f32)], query: &Query) -> Vec<(I, f32)>;
}

// Implementations:
// - CrossEncoderRefiner (candle/ort backend)
// - MatryoshkaRefiner (SIMD vector math)
// - MaxSimRefiner (ColBERT-style late interaction)
```

## The "Bleeding Edge" Context

From 2025 vector search research:

### Matryoshka Representation Learning (MRL)
- First k dimensions are a valid coarse embedding
- Enables "Hot Head / Cold Tail" storage layout
- First 64 dims in cache, rest on disk
- Refinement = computing full distance for top candidates

### ColBERT/PLAID Late Interaction
- Keep all token vectors (not just [CLS])
- MaxSim: for each query token, find max similarity doc token
- PLAID: cluster tokens, only unpack "active" clusters
- 5-10x more accurate than bi-encoder, ~2x slower

### The SIMD Opportunity
- AVX-512 VNNI for int8 dot products (the "offset trick")
- 16 float32s per instruction, 64 int8s per instruction
- Most Rust libs rely on LLVM autovectorization (suboptimal)

## Implementation Priorities

1. **MatryoshkaRefiner** - Pure Rust, no deps, SIMD
2. **CrossEncoderRefiner** - Via candle, quantized models
3. **MaxSimRefiner** - The holy grail, needs careful SIMD work

## References

- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRerank) - Python reference
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) - Late interaction
- [PLAID](https://arxiv.org/abs/2205.09707) - Efficient ColBERT engine
- [SimSIMD](https://github.com/ashvardanian/SimSIMD) - SIMD distance kernels

