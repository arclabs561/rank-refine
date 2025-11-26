# rank-refine

Model-based reranking for retrieval pipelines.

## Install

```toml
[dependencies]
rank-refine = "0.2"
```

## Two-Crate Approach

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

| Crate | Purpose | Dependencies |
|-------|---------|--------------|
| [rank-fusion](https://crates.io/crates/rank-fusion) | Combine result lists | None |
| rank-refine | Re-score with expensive methods | None (candle/ort optional) |

## Available: Matryoshka Refinement

MRL embeddings have a useful property: the first k dimensions are a valid coarse embedding.

Two-stage retrieval:
1. **Coarse**: Search with first 64 dims (fast, cache-friendly)
2. **Refine**: Re-score top candidates using tail dims (accurate)

```rust
use rank_refine::matryoshka;

// Candidates from coarse retrieval (first 64 dims)
let candidates = vec![("doc1", 0.9), ("doc2", 0.85), ("doc3", 0.7)];

// Full 128-dim embeddings for refinement
let query: Vec<f32> = vec![0.1; 128];
let docs = vec![
    ("doc1", vec![0.2; 128]),
    ("doc2", vec![0.15; 128]),
    ("doc3", vec![0.1; 128]),
];

// Refine using dimensions 64..128
let refined = matryoshka::refine(&candidates, &query, &docs, 64);
```

### Blending Options

```rust
// Custom blend: 70% original score, 30% tail similarity
let refined = matryoshka::refine_with_alpha(&candidates, &query, &docs, 64, 0.7);

// Pure tail similarity (ignore original scores)
let refined = matryoshka::refine_tail_only(&candidates, &query, &docs, 64);
```

## Planned

- **Cross-encoder** — Score pairs with transformers (candle or ort)
- **ColBERT/PLAID MaxSim** — Late interaction scoring

## License

MIT OR Apache-2.0
