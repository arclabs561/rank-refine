# Design

Two crates for retrieval pipelines:

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

## Why a Dedicated Reranking Library?

Reranking is architecturally distinct from both retrieval and storage. Research and
industry practice (2024) converge on separating these concerns:

| Concern | Optimized For | Where |
|---------|---------------|-------|
| Storage | Billions of vectors, fast ANN | Vector database (qdrant, milvus) |
| Inference | Transformer models, batching | Embedding service (fastembed, ort) |
| **Reranking** | Small candidate sets, SIMD scoring | **Dedicated library** |

**Why not embed reranking in the vector database?**

1. **Computational mismatch**: Databases optimize for ANN over millions; reranking
   optimizes for precise scoring of 20-100 candidates.

2. **Vendor independence**: Swap databases without rewriting reranking logic.

3. **Pluggable strategies**: Easily switch between dense/late-interaction/cross-encoder
   without touching infrastructure.

4. **Independent scaling**: Run reranking on CPU while database uses GPU/specialized
   hardware.

From [DynamicRAG (Sun et al., 2025)](https://arxiv.org/abs/2505.07233): reranking is
"crucial but often under-explored" — separating it enables focused optimization.

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

## Indexing vs Query Time

Late interaction has distinct **indexing** and **query** phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  INDEXING (Offline, One-time)                                   │
│                                                                 │
│  Document → Encode → Reduce Dims → Pool Tokens → Quantize → DB  │
│             (model)  (768→128)     (cluster)    (2-bit)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  QUERY (Online, Per-request)                                    │
│                                                                 │
│  Query → Encode → Candidate Gen → MaxSim Score → Top-K          │
│          (model)  (ANN/PLAID)     (this crate)                  │
└─────────────────────────────────────────────────────────────────┘
```

**What this crate provides**:
- Query-time: `maxsim`, `rank`, `refine` (fast, CPU-optimized)
- Indexing-time: `pool_tokens`, `pool_tokens_hierarchical` (run once, store results)

**What you bring**:
- Embeddings (from fastembed, ort, sentence-transformers, etc.)
- Storage (qdrant, milvus, in-memory Vec, etc.)
- Candidate generation (ANN from your vector DB)

## Token Pooling

Storage can be reduced via token pooling without significant quality loss. From
[Clavié et al. (2024)](https://arxiv.org/abs/2409.14683):

- **Factor 2 (50% reduction)**: Virtually no retrieval degradation
- **Factor 3-4 (66-75% reduction)**: <5% degradation on most datasets

Key insight: **Pooling happens at indexing time only** — documents are pooled once,
stored, and then scored against un-pooled queries. No query-time processing needed.

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

## Unit Normalization

Many embedding models (especially contrastively trained ones) output unit-normalized
vectors. For unit vectors: `dot(a, b) = cosine(a, b)`.

The `embedding` module provides:

- **`Normalized`** — Wrapper for unit-normalized vectors where `dot = cosine`
- **`MaskedTokens`** — Token embeddings with padding mask for batched MaxSim

```rust
// Normalized embeddings: dot = cosine
let q = normalize(&[3.0, 4.0]).unwrap();
let d = normalize(&[1.0, 0.0]).unwrap();
let sim = q.dot(&d); // This IS cosine similarity

// Masked tokens for batched processing with padding
let masked = MaskedTokens::new(tokens, mask);
let score = maxsim_masked(&query_tokens, &masked);
```

**Scope note**: Training paradigm (contrastive, instruction-tuned, etc.) is the
embedding model's concern, not this crate's. By the time embeddings reach here,
they're just `&[f32]`.

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

Based on review of production Rust implementations:

| Feature | qdrant | vindicator | fast-plaid | rank-refine |
|---------|--------|------------|------------|-------------|
| Focus | Vector DB | Rank fusion | PLAID GPU | Reranking |
| SIMD | AVX, 4-way unroll | N/A | tch (GPU) | AVX+FMA |
| Dim threshold | 16/32 | N/A | N/A | 16 |
| NaN scores | N/A | noisy_float | N/A | `total_cmp` |
| Token pooling | N/A | N/A | Quantization | Clustering |
| Score collection | Vec | SmallVec<4> | Tensor | Vec |

### Tricks Gleaned from Production Code

**From qdrant** (22k stars):
- Four-way loop unrolling in AVX (32 elements/iteration for dot product)
- Separate `hsum256_ps` for horizontal sum
- Dimension threshold to skip SIMD overhead for small vectors

**From PyLate** (LightOn):
- `torch.einsum("ash,bth->abst", Q, D)` for batched MaxSim
- Attention mask support for padding tokens
- Separate `colbert_scores_pairwise` for 1:1 scoring

**From ColPali** (illuin-tech):
- `pad_sequence` for variable-length token batches
- FastPLAID integration for indexed search
- Both single-vector and multi-vector scoring in same API

**From vindicator**:
- `SmallVec<[_; 4]>` for score collection (avoids heap for small fusions)
- `noisy_float::n32` for NaN-safe arithmetic (alternative to `total_cmp`)
- Generic `SearchEntry` trait for flexible ID types

**From fast-plaid**:
- Memory-mapped ColBERT indices reduce RAM from 86GB to 8GB
- Quantization-based compression as alternative to clustering

### Our Design Choices

1. **`total_cmp` over `noisy_float`**: Zero dependencies; works on stable Rust
2. **Clustering over quantization**: Higher quality at same storage (research-backed)
3. **Slice-based API**: Zero-copy where possible; user controls allocation
4. **BYOM**: No inference code = smaller binary, flexible model choice

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
