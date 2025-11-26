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

Storage can be reduced via token pooling without significant quality loss. From
[Clavié et al. (2024)](https://arxiv.org/abs/2409.14683):

- **Factor 2 (50% reduction)**: Virtually no retrieval degradation
- **Factor 3-4 (66-75% reduction)**: <5% degradation on most datasets

Key insight: Pooling happens at **indexing time only** — no query-time processing needed.

### Clustering Methods

Research shows **Ward's hierarchical clustering** outperforms k-means for embedding pooling:

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| Ward (hierarchical) | Highest | O(n²) | Aggressive compression (4x+) |
| Greedy clustering | Good | O(n) | Moderate compression (2-3x) |
| Sequential | Lowest | O(n) | Speed-critical, position-aware |

Ward produces more **compact, homogeneous clusters** and handles ambiguous tokens better.

### Implementation

```rust
// Aggressive compression with highest quality
#[cfg(feature = "hierarchical")]
let pooled = pool_tokens_hierarchical(&tokens, 4);

// Balanced (default)
let pooled = pool_tokens(&tokens, 2);

// Preserve special tokens
let pooled = pool_tokens_with_protected(&tokens, 2, 2); // protect first 2
```

Protected tokens (like `[CLS]` and `[D]` markers) can be preserved during pooling.

## Modules

- **matryoshka** — Two-stage retrieval: coarse with head dims, refine with tail
- **colbert** — MaxSim late interaction: `Σ_q max_d(q·d)`, with token pooling
- **crossencoder** — BYOM trait for transformer-based scoring
- **simd** — AVX2+FMA (x86_64), NEON (aarch64), portable fallback
- **scoring** — Unified `Scorer` and `TokenScorer` traits for interoperability

### Matryoshka Note

Current implementation supports **dimension truncation** (standard MRL). The emerging
**2D Matryoshka** approach (Wang et al., 2024) adds **layer truncation** — using embeddings
from intermediate transformer layers for even more flexibility. This would require:

1. Models trained with 2D Matryoshka loss
2. API to specify both `target_dims` and `target_layer`

The current `matryoshka::refine` function handles dimension-based refinement correctly
and is compatible with any MRL-trained model (e.g., Nomic, Cohere v3, OpenAI text-embedding-3).

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
pub trait Scorer {
    fn score(&self, query: &[f32], doc: &[f32]) -> f32;
    fn rank<I: Clone>(&self, query: &[f32], docs: &[(I, &[f32])]) -> Vec<(I, f32)>;
}

pub trait TokenScorer {
    fn score_tokens(&self, query: &[&[f32]], doc: &[&[f32]]) -> f32;
    fn rank_tokens<I: Clone>(&self, query: &[&[f32]], docs: &[(I, &[&[f32]])]) -> Vec<(I, f32)>;
}
```

Implementations: `DenseScorer` (Dot, Cosine) and `LateInteractionScorer` (MaxSimDot, MaxSimCosine).

## NaN Handling

Sorting uses `f32::total_cmp` for deterministic ordering:

- NaN values sort before all other values
- This ensures consistent behavior regardless of input quality
- Users should filter NaN scores if they appear (indicates bad embeddings)

## Implementation Comparison

Based on review of production Rust implementations (qdrant, fast-plaid, vecstore):

| Feature | qdrant | fast-plaid | vecstore | rank-refine |
|---------|--------|------------|----------|-------------|
| SIMD dispatch | Runtime | GPU (tch) | None | Runtime |
| Min dim threshold | 16/32 | N/A | None | 16 |
| NaN handling | N/A | N/A | `partial_cmp` | `total_cmp` |
| Token pooling | None | Quantization | None | Clustering |
| Metric traits | `Metric<T>` | N/A | Config | `Scorer`/`TokenScorer` |

Key differences:
- **qdrant**: Full vector database; SIMD optimized with dimension thresholds
- **fast-plaid**: GPU-based PLAID engine; uses PyTorch tensors
- **vecstore**: Alpha-stage; simpler MaxSim but lacks NaN safety
- **rank-refine**: CPU reranking library; unique hierarchical token pooling

## References

### Core Techniques
- [Matryoshka](https://arxiv.org/abs/2205.13147) — MRL embeddings (Kusupati et al., 2022)
- [2D Matryoshka](https://arxiv.org/abs/2411.17299) — Layer + dimension truncation (Wang et al., 2024)
- [ColBERT](https://arxiv.org/abs/2004.12832) — Late interaction retrieval (Khattab & Zaharia, 2020)
- [PLAID](https://arxiv.org/abs/2205.09707) — Efficient ColBERT serving

### Token Pooling
- [Token Pooling](https://arxiv.org/abs/2409.14683) — 50-75% storage reduction (Clavié et al., 2024)
- [kodama](https://docs.rs/kodama) — Ward's hierarchical clustering for Rust

### Production Systems
- [ColBERT-serve](https://arxiv.org/abs/2501.02065) — Memory-mapped ColBERT (ECIR 2025)
- [Jina-ColBERT-v2](https://arxiv.org/abs/2408.16672) — Multilingual + Matryoshka ColBERT
- [fastembed-rs](https://docs.rs/fastembed) — Rust embedding generation via ONNX
- [ort](https://docs.rs/ort) — ONNX Runtime bindings for cross-encoder inference
