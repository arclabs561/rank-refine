# rank-refine

**Reranking algorithms for retrieval pipelines.** SIMD-accelerated, zero dependencies.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

## Why This Library?

Modern search pipelines are two-stage:

```
10M docs → fast retrieval (ANN, BM25) → 100 candidates → reranking → 10 results
```

**This crate is the reranking step.** It takes embeddings and scores them.

| What | This Crate | Other Tools |
|------|------------|-------------|
| **Input** | Embeddings (f32 vectors) | Text, raw documents |
| **Output** | Relevance scores | Model inference |
| **Dependencies** | 0 (or 1 with `hierarchical`) | 50+ (transformers, torch) |
| **Compile time** | ~0.2s | Minutes |

**Bring Your Own Model (BYOM)**: We score embeddings — we don't generate them.
Use [fastembed](https://github.com/Anush008/fastembed-rs), [candle](https://github.com/huggingface/candle), 
[ort](https://github.com/pykeio/ort), or any embedding source.

### vs rank-fusion

| | rank-refine | [rank-fusion](https://crates.io/crates/rank-fusion) |
|-|-------------|-------------|
| **Input** | Embeddings | Ranked lists with scores |
| **Use case** | Rerank with semantic similarity | Combine multiple retrievers |
| **Example** | Score query vs documents with cosine | Merge BM25 + dense results |

Use **rank-fusion** when you have multiple retriever outputs to combine.
Use **rank-refine** when you have embeddings and want better ranking.

## Quick Start

```rust
use rank_refine::simd::cosine;

// Score query against each candidate
let query = vec![0.1, 0.2, 0.3, 0.4];
let docs = vec![
    vec![0.1, 0.2, 0.3, 0.4],  // similar
    vec![0.9, 0.0, 0.0, 0.1],  // different
];

let scores: Vec<f32> = docs.iter()
    .map(|d| cosine(&query, d))
    .collect();
// [1.0, 0.28] — first doc most similar
```

## Methods

### Dense Scoring

Single-vector embeddings. SIMD auto-dispatches (AVX2+FMA on x86_64, NEON on ARM).

```rust
use rank_refine::simd::{cosine, dot, norm};

let score = cosine(&a, &b);   // -1 to 1, normalized
let score = dot(&a, &b);      // unbounded, for pre-normalized vectors
let len = norm(&v);           // L2 norm
```

**When to use**: Fast baseline. Most embedding models output single vectors.

### Late Interaction (MaxSim)

Per-token embeddings for finer-grained matching. Used by ColBERT, ColPali, Jina-ColBERT-v2.

```rust
use rank_refine::simd::{maxsim, maxsim_cosine, maxsim_batch};

// Query: 3 tokens × 128 dims, Doc: 10 tokens × 128 dims
let query: Vec<Vec<f32>> = vec![vec![0.1; 128]; 3];
let doc: Vec<Vec<f32>> = vec![vec![0.2; 128]; 10];

let score = maxsim_vecs(&query, &doc);

// Batch scoring: one query, many docs
let docs: Vec<Vec<Vec<f32>>> = vec![doc; 100];
let scores = maxsim_batch(&query, &docs);
```

**Formula**: For each query token, find best-matching doc token, then sum.

$$\text{MaxSim}(Q, D) = \sum_{q \in Q} \max_{d \in D} (q \cdot d)$$

**Note**: `maxsim(Q, D) ≠ maxsim(D, Q)` — query must be first argument.

### Token Importance Weighting

Weight query tokens by importance (e.g., IDF). Research shows +2-5% quality.

```rust
use rank_refine::simd::maxsim_weighted;

let weights = vec![1.0, 0.5, 2.0];  // per-token weights
let score = maxsim_weighted(&query, &doc, &weights);
```

### Score Normalization

MaxSim scores are unbounded. Normalize for comparison:

```rust
use rank_refine::simd::{normalize_maxsim, softmax_scores, top_k_indices};

// Divide by query length (ColBERT convention: 32)
let normalized = normalize_maxsim(score, 32);

// Softmax for relative comparison
let probs = softmax_scores(&scores);

// Get top-k indices
let top_10 = top_k_indices(&scores, 10);
```

### Token Pooling

Reduce ColBERT storage by clustering similar tokens:

```rust
use rank_refine::colbert::{pool_tokens, pool_tokens_adaptive, pool_tokens_with_protected};

// 32 tokens → 8 tokens (factor 4 = 75% compression)
let pooled = pool_tokens(&doc_tokens, 4);

// Adaptive: switches method based on token count
let pooled = pool_tokens_adaptive(&doc_tokens, 4);

// Preserve special tokens ([CLS], [D])
let pooled = pool_tokens_with_protected(&doc_tokens, 4, 2);
```

| Factor | Compression | Quality Loss |
|--------|-------------|--------------|
| 2 | 50% | ~0% |
| 3 | 66% | ~1% |
| 4+ | 75%+ | 2-5% (use `hierarchical` feature) |

### TokenIndex

Pre-indexed documents for repeated queries:

```rust
use rank_refine::colbert::TokenIndex;

let index = TokenIndex::new(vec![
    ("doc1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
    ("doc2", vec![vec![0.5, 0.5]]),
]);

let results = index.top_k(&query_tokens, 10);
```

### Diversity Selection (MMR & DPP)

Avoid returning near-duplicates:

```rust
use rank_refine::diversity::{mmr_cosine, dpp, MmrConfig, DppConfig};

// Maximal Marginal Relevance
let config = MmrConfig::default().with_k(5).with_lambda(0.5);
let diverse = mmr_cosine(&candidates, &embeddings, config);

// Determinantal Point Process (better theoretical guarantees)
let config = DppConfig::new(5, 0.5);
let diverse = dpp(&candidates, &embeddings, config);
```

- λ=1.0: pure relevance (top-k)
- λ=0.5: balanced (default)
- λ=0.0: maximum diversity

### Matryoshka Embeddings

Two-stage refinement with truncated embeddings:

```rust
use rank_refine::matryoshka::refine;

// Stage 1: coarse ranking with first 128 dims
// Stage 2: fine-tune with remaining dims
let results = refine(&candidates, &query, 128, &docs);
```

### Cross-Encoder

Trait for transformer-based scoring:

```rust
use rank_refine::crossencoder::{CrossEncoderModel, rerank};

struct MyModel { /* ort, candle, etc. */ }

impl CrossEncoderModel for MyModel {
    fn score(&self, query: &str, doc: &str) -> f32 {
        // Your inference code
    }
}

let ranked = rerank(&model, "query", &[("id1", "doc text")]);
```

### Type-Safe Wrappers

Encode mathematical invariants at compile time:

```rust
use rank_refine::embedding::{Normalized, MaskedTokens, normalize};

// Normalized: guarantees unit length
let n = normalize(&vec![3.0, 4.0]).unwrap();
assert!((n.dot(&n) - 1.0).abs() < 1e-6);

// MaskedTokens: exclude padding from MaxSim
let masked = MaskedTokens::with_mask(&tokens, &attention_mask);
```

## Features

| Feature | Adds | Purpose |
|---------|------|---------|
| `hierarchical` | kodama | Ward's clustering for 4x+ pooling |

## Examples

```bash
cargo run --example rerank           # Dense scoring
cargo run --example rag_rerank       # RAG pipeline
cargo run --example search_diversity # MMR diversity
cargo run --example colbert_pooling  # Token compression
```

## Documentation

- **[DESIGN.md](DESIGN.md)**: Architecture, API rationale, research citations
- **[REFERENCE.md](REFERENCE.md)**: Algorithms, pseudocode, mathematical properties

## Performance

- **SIMD**: Auto-dispatches AVX2+FMA (x86_64) or NEON (aarch64)
- **Zero-copy**: Slice-based APIs avoid allocation
- **Minimal threshold**: SIMD skipped for dim < 16 (overhead exceeds benefit)

## License

MIT OR Apache-2.0
