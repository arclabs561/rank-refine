# rank-refine

Model-based reranking for retrieval pipelines.

## Context

This crate is part of a two-crate approach:

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

**rank-fusion** combines multiple result lists. Zero dependencies.

**rank-refine** re-scores a single list with expensive methods. This crate.

## Planned Refiners

### 1. Matryoshka Refinement

MRL embeddings have the property that the first k dimensions are a valid
coarse embedding. This enables:

1. Initial retrieval with first 64 dims (fast, fits in cache)
2. Refinement with full 768 dims (accurate, for top candidates only)

No model inference needed — just SIMD vector math.

### 2. Cross-Encoder

Score (query, document) pairs with a transformer model. More accurate than
bi-encoder similarity but O(n) inference cost.

Backends: candle (pure Rust) or ort (ONNX Runtime).

### 3. ColBERT/PLAID MaxSim

Late interaction: keep all token embeddings, not just [CLS].

```
score = Σ max(q_i · D)  for each query token q_i
```

PLAID optimization: cluster tokens, only decompress active clusters.

## Architecture

```rust
pub trait Refiner<I, Q> {
    fn refine(&self, candidates: &[(I, f32)], query: &Q) -> Vec<(I, f32)>;
}
```

Implementations will be feature-gated:
- `matryoshka` — no deps, SIMD
- `cross-encoder` — requires candle or ort
- `colbert` — requires candle or ort

## References

- Kusupati et al. (2022) — [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- Khattab & Zaharia (2020) — [ColBERT](https://arxiv.org/abs/2004.12832)
- Santhanam et al. (2022) — [PLAID](https://arxiv.org/abs/2205.09707)
