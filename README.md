# rank-refine

Reranking algorithms for retrieval pipelines. **No model downloads required.**

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

This crate provides scoring algorithms only — you bring your own embeddings (from sentence-transformers, fastembed, candle, etc.) and this crate handles the reranking math.

```rust
use rank_refine::{simd, colbert};

// Dense scoring
let score = simd::cosine(&[1.0, 0.0], &[0.707, 0.707]);

// ColBERT MaxSim
let q: Vec<&[f32]> = vec![&[1.0, 0.0]];
let d: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.1, 0.9]];
let maxsim = simd::maxsim(&q, &d);

// Token pooling (compress 32 tokens to 8)
let tokens: Vec<Vec<f32>> = vec![vec![1.0; 128]; 32];
let pooled = colbert::pool_tokens_adaptive(&tokens, 4);
```

## Modules

| Module | Purpose |
|--------|---------|
| `matryoshka` | MRL tail dimension refinement |
| `colbert` | MaxSim late interaction + token pooling |
| `crossencoder` | Cross-encoder trait (implement for your model) |
| `simd` | Vector ops (AVX2/NEON accelerated) |
| `scoring` | Unified `Scorer`, `TokenScorer`, `Pooler` traits |

## Features

| Feature | Description |
|---------|-------------|
| `hierarchical` | Ward's method clustering via `kodama` (better pooling at high compression) |

## Design

- **No downloads**: Tests use synthetic vectors, not real models
- **No ML dependencies**: No ONNX, PyTorch, or ML frameworks
- **Pure Rust**: SIMD-accelerated math you can audit

## License

MIT OR Apache-2.0
