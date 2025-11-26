# Design

Two crates for retrieval pipelines:

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

## Why Late Interaction?

Dense retrieval compresses a document to a single vector, losing token-level semantics.
Late interaction preserves this information:

| Approach | Representation | Score Cost | Quality |
|----------|----------------|------------|---------|
| Dense | Single `Vec<f32>` | O(d) dot | Good |
| Late Interaction | `Vec<Vec<f32>>` | O(q × d) MaxSim | Better |
| Cross-encoder | Full text pair | O(n) inference | Best |

For example, "What is the capital of France?" with dense retrieval might match documents
about capitals in general. Late interaction's token-level matching finds documents where
"capital" and "France" co-occur with high similarity.

**Trade-off**: Late interaction costs more storage (128 tokens × 128 dims vs 1 × 768)
but yields 5-15% better recall on complex queries.

## Token Pooling

Storage can be reduced via token pooling without significant quality loss:

- **Factor 2**: 50% storage reduction, ~1% recall drop
- **Factor 3**: 66% storage reduction, ~2% recall drop

Two strategies:

1. **Hierarchical clustering** (Ward's method via `kodama`): Groups semantically similar
   tokens. Higher quality but O(n²) complexity.

2. **Sequential pooling**: Averages adjacent tokens in sliding windows. Faster but
   less semantically aware.

Protected tokens (like `[CLS]` and `[D]` markers) can be preserved during pooling.

## Modules

- **matryoshka** — Two-stage retrieval: coarse with head dims, refine with tail
- **colbert** — MaxSim late interaction: `Σ_q max_d(q·d)`, with token pooling
- **crossencoder** — BYOM trait for transformer-based scoring
- **simd** — AVX2+FMA (x86_64), NEON (aarch64), portable fallback
- **scoring** — Unified `Scorer` and `TokenScorer` traits for interoperability

## Architecture

```
                      ┌──────────────────┐
                      │   Your Model     │
                      │  (embeddings)    │
                      └────────┬─────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  matryoshka   │  │   colbert     │  │ crossencoder  │
    │  (MRL tail)   │  │  (MaxSim)     │  │    (BYOM)     │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │      simd        │
                      │  (dot, cosine,   │
                      │   maxsim)        │
                      └──────────────────┘
```

## Trait Design

The `scoring` module provides unified traits:

```rust
// Dense scoring
pub trait Scorer {
    fn score(&self, query: &[f32], doc: &[f32]) -> f32;
    fn rank<I: Clone>(&self, query: &[f32], docs: &[(I, &[f32])]) -> Vec<(I, f32)>;
}

// Late interaction scoring
pub trait TokenScorer {
    fn score_tokens(&self, query: &[&[f32]], doc: &[&[f32]]) -> f32;
    fn rank_tokens<I: Clone>(&self, query: &[&[f32]], docs: &[(I, Vec<&[f32]>)]) -> Vec<(I, f32)>;
}
```

This enables swapping scoring strategies without changing pipeline code.

## NaN Handling

Sorting uses `f32::total_cmp` for deterministic ordering:

- NaN values sort before all other values
- This ensures consistent behavior regardless of input quality
- Users should filter NaN scores if they appear (indicates bad embeddings)

## References

- [Matryoshka](https://arxiv.org/abs/2205.13147) — MRL embeddings
- [ColBERT](https://arxiv.org/abs/2004.12832) — Late interaction retrieval
- [PLAID](https://arxiv.org/abs/2205.09707) — Efficient ColBERT serving
- [ColBERT-serve](https://arxiv.org/abs/2501.02065) — Memory-mapped ColBERT (ECIR 2025)
- [Jina-ColBERT-v2](https://arxiv.org/abs/2408.16672) — Multilingual + Matryoshka ColBERT
- [kodama](https://docs.rs/kodama) — Hierarchical clustering for token pooling
