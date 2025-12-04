# Getting Started with rank-refine

This guide walks you through using `rank-refine` in real-world scenarios, from basic similarity scoring to complete RAG reranking pipelines.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [RAG Reranking Pipeline](#rag-reranking-pipeline)
3. [Integration with rank-fusion](#integration-with-rank-fusion)
4. [Diversity Selection](#diversity-selection)
5. [Token Pooling](#token-pooling)
6. [Python Integration](#python-integration)

---

## Quick Start (5 minutes)

### Installation

```bash
cargo add rank-refine
```

### Basic Example: Dense Similarity Scoring

```rust
use rank_refine::simd::cosine;

// Query and document embeddings (from your embedding model)
let query = vec![0.8, 0.6, 0.4, ...];  // 384-dimensional embedding
let doc = vec![0.7, 0.5, 0.5, ...];    // 384-dimensional embedding

// Compute cosine similarity
let score = cosine(&query, &doc);
// Result: 0.92 (high similarity)
```

### Late Interaction (ColBERT MaxSim)

```rust
use rank_refine::simd::maxsim_vecs;

// Query tokens (one embedding per token)
let query_tokens = vec![
    vec![0.8, 0.6, ...],  // "capital" token embedding
    vec![0.7, 0.5, ...],  // "of" token embedding
    vec![0.9, 0.4, ...],  // "France" token embedding
];

// Document tokens (one embedding per token)
let doc_tokens = vec![
    vec![0.79, 0.61, ...],  // "Paris" token embedding
    vec![0.72, 0.48, ...],  // "is" token embedding
    vec![0.88, 0.42, ...],  // "capital" token embedding
    // ... more tokens
];

// MaxSim finds best match for each query token
let score = maxsim_vecs(&query_tokens, &doc_tokens);
// Result: sum of best matches per query token
```

**Why MaxSim?** Unlike dense embeddings that pool all tokens into one vector, MaxSim preserves token-level alignment. This enables precise matching: "capital" matches "capital" even if they appear in different positions.

---

## RAG Reranking Pipeline

Complete example: Dense retrieval → Rerank with MaxSim → Top-K

```rust
use rank_refine::simd::maxsim_vecs;

// Step 1: Get initial candidates from dense retrieval (e.g., Qdrant, Pinecone)
// In production, this would come from your vector database
let candidates = vec![
    ("doc_1", get_doc_tokens("doc_1")),
    ("doc_2", get_doc_tokens("doc_2")),
    ("doc_3", get_doc_tokens("doc_3")),
    // ... 100-1000 candidates
];

// Step 2: Rerank with MaxSim (late interaction)
let query_tokens = get_query_tokens("capital of France");

let mut scored: Vec<(&str, f32)> = candidates
    .iter()
    .map(|(doc_id, doc_tokens)| {
        let score = maxsim_vecs(&query_tokens, doc_tokens);
        (*doc_id, score)
    })
    .collect();

// Step 3: Sort by score (descending)
scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

// Step 4: Take top-K
let top_k = scored.into_iter().take(10).collect::<Vec<_>>();
```

**Performance**: Reranking 100 candidates with MaxSim takes ~5ms on Apple M3 Max. Suitable for real-time RAG pipelines.

---

## Integration with rank-fusion

Combine `rank-refine` (scoring) with `rank-fusion` (fusion) for hybrid search:

```rust
use rank_refine::simd::maxsim_vecs;
// use rank_fusion::rrf;  // Add rank-fusion as dependency

// Step 1: Retrieve from multiple sources
let bm25_candidates = get_bm25_candidates(query);  // Elasticsearch
let dense_candidates = get_dense_candidates(query);  // Vector DB

// Step 2: Score with rank-refine (MaxSim)
let query_tokens = get_query_tokens(query);

let bm25_scored: Vec<(String, f32)> = bm25_candidates
    .iter()
    .map(|(id, doc_tokens)| {
        let score = maxsim_vecs(&query_tokens, doc_tokens);
        (id.clone(), score)
    })
    .collect();

let dense_scored: Vec<(String, f32)> = dense_candidates
    .iter()
    .map(|(id, doc_tokens)| {
        let score = maxsim_vecs(&query_tokens, doc_tokens);
        (id.clone(), score)
    })
    .collect();

// Step 3: Fuse with rank-fusion (RRF)
// let fused = rrf(&bm25_scored, &dense_scored);
```

See [`examples/refine_to_fusion_pipeline.rs`](examples/refine_to_fusion_pipeline.rs) for a complete example.

---

## Diversity Selection

Use MMR (Maximal Marginal Relevance) to diversify results:

```rust
use rank_refine::diversity::{mmr_cosine, MmrConfig};

// Candidates with their embeddings
let candidates = vec![
    ("doc_1", vec![0.9, 0.1, ...]),
    ("doc_2", vec![0.8, 0.2, ...]),
    ("doc_3", vec![0.7, 0.3, ...]),
    // ... more candidates
];

// Query embedding
let query = vec![1.0, 0.0, ...];

// MMR config: λ = 0.5 (balanced relevance and diversity)
let config = MmrConfig::new(0.5);

// Select diverse top-K
let diverse = mmr_cosine(&candidates, &query, config, 10);
```

**When to use**: RAG pipelines where you want diverse context for LLMs, avoiding near-duplicate documents.

---

## Token Pooling

Compress token embeddings to reduce storage:

```rust
use rank_refine::colbert::pool_tokens;

// Original: 100 tokens × 128 dim = 12,800 floats
let doc_tokens = vec![
    vec![0.1, 0.2, ...],  // token 1
    vec![0.2, 0.3, ...],  // token 2
    // ... 98 more tokens
];

// Pool with factor 2: keep 50 tokens (50% reduction)
let pooled = pool_tokens(&doc_tokens, 2);
// Result: 50 tokens × 128 dim = 6,400 floats (50% storage reduction)
```

**Trade-off**: Factor 2 pooling reduces storage by 50% with only 0.1-0.3% MRR@10 loss (MS MARCO benchmarks).

---

## Python Integration

```python
import rank_refine

# Dense cosine similarity
query = [0.8, 0.6, 0.4]
doc = [0.7, 0.5, 0.5]
score = rank_refine.cosine(query, doc)

# MaxSim (late interaction)
query_tokens = [[0.8, 0.6], [0.7, 0.5], [0.9, 0.4]]
doc_tokens = [[0.79, 0.61], [0.72, 0.48], [0.88, 0.42]]
score = rank_refine.maxsim_vecs(query_tokens, doc_tokens)
```

See [`rank-refine-python/README.md`](../rank-refine-python/README.md) for more Python examples.

---

## Performance Considerations

### Latency

- **Dense cosine (128 dim)**: ~40ns per pair
- **MaxSim (32 query × 128 doc × 128 dim)**: ~49μs per document
- **Reranking 100 candidates**: ~5ms total

### Memory

- **Dense embeddings**: 1 vector per document (~512 bytes for 128-dim f32)
- **Token embeddings**: ~10-50 vectors per document (depends on document length)
- **Storage**: Token embeddings are 10-50x larger than dense (use pooling to reduce)

### When to Use

**Use dense (cosine)**:
- First-stage retrieval (millions of docs)
- Latency-critical applications
- Storage-constrained deployments

**Use MaxSim (late interaction)**:
- Second-stage reranking (100-1000 candidates)
- Precision-critical applications
- Token-level alignment needed

---

## Next Steps

- **See [API Documentation](https://docs.rs/rank-refine)** for all functions
- **See [DESIGN.md](DESIGN.md)** for algorithm details
- **See [INTEGRATION.md](INTEGRATION.md)** for framework-specific examples
- **See [examples/](examples/)** for complete runnable examples

---

## Common Pitfalls

### 1. Using MaxSim for first-stage retrieval

**Problem**: MaxSim is too slow for millions of documents.

**Solution**: Use dense embeddings + ANN search for first stage, then rerank top-100 with MaxSim.

### 2. Not pooling token embeddings

**Problem**: Token embeddings are 10-50x larger than dense, causing storage issues.

**Solution**: Use `pool_tokens` with factor 2-3 (50-66% reduction, minimal quality loss).

### 3. Mismatched embedding dimensions

**Problem**: Query and document embeddings must have the same dimension.

**Solution**: Ensure your embedding model produces consistent dimensions.

---

## Getting Help

- **GitHub Issues**: https://github.com/arclabs561/rank-refine/issues
- **Documentation**: https://docs.rs/rank-refine
- **Examples**: See `examples/` directory

