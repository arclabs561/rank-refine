# rank-refine

Reranking for retrieval pipelines.

[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

## Usage

```toml
[dependencies]
rank-refine = "0.7"

# Optional: hierarchical clustering for better token pooling
rank-refine = { version = "0.7", features = ["hierarchical"] }
```

## Example

```rust
use rank_refine::{colbert, matryoshka};

// Matryoshka refinement (MRL embeddings)
let candidates = vec![("d1", 0.9), ("d2", 0.8)];
let query = vec![0.1, 0.2, 0.3, 0.4];
let docs = vec![
    ("d1", vec![0.1, 0.2, 0.3, 0.4]),
    ("d2", vec![0.1, 0.2, 0.3, 0.4]),
];
let refined = matryoshka::refine(&candidates, &query, &docs, 2);

// ColBERT MaxSim ranking
let query_tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let doc_tokens = vec![
    ("d1", vec![vec![0.9, 0.1], vec![0.1, 0.9]]),
    ("d2", vec![vec![0.5, 0.5], vec![0.5, 0.5]]),
];
let ranked = colbert::rank(&query_tokens, &doc_tokens);

// Token pooling (reduces storage 50-66%)
let pooled = colbert::pool_tokens(&doc_tokens[0].1, 2);
```

## Modules

| Module | Method |
|--------|--------|
| `matryoshka` | MRL tail dimension refinement |
| `colbert` | MaxSim late interaction + token pooling |
| `crossencoder` | Cross-encoder trait (BYOM) |
| `simd` | Vector ops (AVX2/NEON) |
| `scoring` | Unified `Scorer` and `TokenScorer` traits |

## Features

- `hierarchical`: Use [kodama](https://docs.rs/kodama) for Ward's method clustering
  in token pooling (better quality, O(n²) vs O(n³))

## Development

```bash
# Run tests
cargo test --all-features

# Run benchmarks
cargo bench

# Fuzz testing (requires nightly)
rustup run nightly cargo fuzz run fuzz_simd
```

## License

MIT OR Apache-2.0
