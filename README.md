# rank-refine

Rescore search candidates with embeddings. SIMD-accelerated.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

**Zero dependencies** by default. Adds ~0.2s to compile, ~50KB to binary.

## When to Use

You have candidates from initial retrieval and want better ranking using embeddings.

```rust
use rank_refine::simd::cosine;

let score = cosine(&query_emb, &doc_emb);
```

## When NOT to Use

- Vector DB handles reranking → use theirs
- No embeddings available → use [`rank-fusion`](https://crates.io/crates/rank-fusion) with scores only
- Need model inference → use fastembed, candle, ort (we only score, not embed)

## The Problem

Initial retrieval is fast but imprecise. Expensive models are accurate but slow.

```
10M docs → ANN (10ms) → 100 candidates → rerank (5ms) → 10 results
```

This crate is the reranking step.

## Methods

| Method | Speed | Quality | Input |
|--------|-------|---------|-------|
| `cosine` / `dot` | ~0.1μs | Baseline | Dense embeddings |
| `maxsim` | ~50μs | +5-15% | Token embeddings |
| `CrossEncoderModel` | ~10ms | +10-20% | Text pairs |

### Dense (cosine/dot)

```rust
use rank_refine::simd::{cosine, dot};

let score = cosine(&query, &doc);  // normalized
let score = dot(&query, &doc);     // unnormalized
```

SIMD auto-dispatches: AVX2+FMA on x86_64, NEON on ARM.

### ColBERT (MaxSim)

Per-token embeddings, finer-grained matching:

```math
\text{MaxSim}(Q, D) = \sum_{q \in Q} \max_{d \in D} \text{sim}(q, d)
```

```rust
use rank_refine::colbert::{maxsim, rank};

let score = maxsim(&query_tokens, &doc_tokens);
```

Note: `maxsim(Q, D) ≠ maxsim(D, Q)` — argument order matters.

### TokenIndex

Pre-compute embeddings for repeated scoring:

```rust
use rank_refine::colbert::TokenIndex;

// Build index once
let index = TokenIndex::new(vec![
    ("doc1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
    ("doc2", vec![vec![0.5, 0.5]]),
]);

// Query many times
let results = index.top_k(&query_tokens, 10);
```

### Cross-encoder

Implement the trait with your inference backend:

```rust
impl CrossEncoderModel for MyModel {
    fn score(&self, query: &str, doc: &str) -> f32 { /* ... */ }
}
```

## Diversity (MMR)

Select diverse results, not just top-k most similar:

```math
\text{MMR} = \arg\max_{d} \left[ \lambda \cdot \text{rel}(d) - (1-\lambda) \cdot \max_{s \in S} \text{sim}(d, s) \right]
```

```rust
use rank_refine::diversity::{mmr_cosine, MmrConfig};

// Candidates: (id, relevance_score)
let candidates = vec![("d1", 0.9), ("d2", 0.85), ("d3", 0.8)];
// Their embeddings (same order)
let embeddings = vec![
    vec![1.0, 0.0],  // d1
    vec![0.9, 0.1],  // d2: similar to d1
    vec![0.0, 1.0],  // d3: different
];

let config = MmrConfig::default().with_k(2);
let diverse = mmr_cosine(&candidates, &embeddings, config);
// Returns d1 (most relevant) and d3 (diverse from d1)
```

- λ=1.0: pure relevance
- λ=0.5: balanced (default)
- λ=0.0: maximum diversity

## Token Pooling

Reduce ColBERT storage by clustering similar tokens:

```rust
use rank_refine::colbert::pool_tokens;

// 32 tokens × 128 dims
let doc_tokens: Vec<Vec<f32>> = vec![vec![0.1; 128]; 32];
let pooled = pool_tokens(&doc_tokens, 4);  // → 8 tokens (75% smaller)
```

## Features

| Feature | Adds | Purpose |
|---------|------|---------|
| `hierarchical` | kodama | Better pooling at 4x+ compression |

## Documentation

| Document | Audience | Contents |
|----------|----------|----------|
| [README](README.md) | Getting started | Quick examples, when to use |
| [DESIGN.md](DESIGN.md) | Architecture | Module design, API rationale |
| [REFERENCE.md](REFERENCE.md) | Deep dive | Algorithms, math, pseudocode |

**New to this?** Start with the examples above, then read REFERENCE.md
to understand how MaxSim and token pooling work under the hood.

## License

MIT OR Apache-2.0
