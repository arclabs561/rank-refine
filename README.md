# rank-refine

SIMD-accelerated similarity scoring for vector search and RAG. Provides MaxSim (ColBERT), cosine similarity, diversity selection (MMR, DPP), and token pooling.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

```
cargo add rank-refine
```

## Why Late Interaction?

Dense retrieval encodes each document as a single vector. This works for broad matching but loses token-level alignment.

**Problem**: A query like "capital of France" might match a document about "France's economic capital" with high similarity, even if it never mentions "capital" in the geographic sense.

**Solution**: Late interaction (ColBERT-style) keeps one vector per token instead of pooling. At query time, each query token finds its best-matching document token, then we sum those matches.

```
Dense:           "the quick brown fox" → [0.1, 0.2, ...]  (1 vector)
Late Interaction: "the quick brown fox" → [[...], [...], [...], [...]]  (4 vectors)
```

This preserves token-level semantics that single-vector embeddings lose. Useful for reranking in RAG pipelines.

## What This Is

Scoring primitives for retrieval systems:

| You need | This crate provides |
|----------|---------------------|
| Score pre-computed embeddings | `cosine`, `dot`, `maxsim` |
| ColBERT/late interaction | `maxsim_vecs`, `maxsim_batch` |
| Diversity selection | `mmr_cosine`, `dpp` |
| Compress token embeddings | `pool_tokens`, `pool_tokens_adaptive` |

**What this is NOT**: embedding generation, model inference, storage. See [fastembed-rs](https://github.com/Anush008/fastembed-rs) for inference.

## Usage

```rust
use rank_refine::simd::{cosine, maxsim_vecs};

// Dense similarity
let score = cosine(&query_embedding, &doc_embedding);

// Late interaction (ColBERT)
let score = maxsim_vecs(&query_tokens, &doc_tokens);
```

### Realistic Example

```rust
use rank_refine::colbert;

// Query: "capital of France" (32 tokens, 128-dim embeddings)
let query = vec![
    vec![0.12, -0.45, 0.89, ...],  // "capital" token
    vec![0.34, 0.67, -0.23, ...],  // "of" token
    vec![0.78, -0.12, 0.45, ...],  // "France" token
    // ... 29 more tokens
];

// Document: "Paris is the capital of France" (100 tokens)
let doc = vec![
    vec![0.11, -0.44, 0.90, ...],  // "Paris" token
    vec![0.35, 0.66, -0.24, ...],  // "is" token
    // ... 98 more tokens
];

// MaxSim finds best matches for each query token
let score = colbert::maxsim_vecs(&query, &doc);
// "capital" matches "capital" (0.95), "France" matches "France" (0.92)
// Score = 0.95 + 0.92 + ... (sum of best matches per query token)
```

## API

### Similarity (SIMD-accelerated)

| Function | Input | Notes |
|----------|-------|-------|
| `cosine(a, b)` | `&[f32]` | Normalized, -1 to 1 |
| `dot(a, b)` | `&[f32]` | Unnormalized |
| `maxsim(q, d)` | `&[&[f32]]` | Sum of max similarities |
| `maxsim_cosine(q, d)` | `&[&[f32]]` | Cosine variant |
| `maxsim_weighted(q, d, w)` | `&[&[f32]], &[f32]` | Per-token weights |
| `maxsim_batch(q, docs)` | `&[Vec<f32>], &[Vec<Vec<f32>>]` | Batch scoring |

### Token Pooling

| Function | Tokens Kept | Notes |
|----------|-------------|-------|
| `pool_tokens(t, 2)` | 50% | Safe default |
| `pool_tokens(t, 4)` | 25% | Use `hierarchical` feature |
| `pool_tokens_adaptive(t, f)` | varies | Auto-selects greedy vs ward |
| `pool_tokens_with_protected(t, f, n)` | varies | Keeps first n tokens unpooled |

### Diversity

| Function | Algorithm |
|----------|-----------|
| `mmr_cosine(candidates, embeddings, config)` | Maximal Marginal Relevance |
| `dpp(candidates, embeddings, config)` | Determinantal Point Process |

### Utilities

| Function | Purpose |
|----------|---------|
| `normalize_maxsim(score, qlen)` | Scale to [0,1] |
| `softmax_scores(scores)` | Probability distribution |
| `top_k_indices(scores, k)` | Top-k by score |
| `blend(a, b, α)` | Linear interpolation |

## How It Works

### MaxSim (Late Interaction)

MaxSim scores token-level alignment. For each query token, find its best-matching document token, then sum:

$$\text{score}(Q, D) = \sum_{i} \max_{j} (q_i \cdot d_j)$$

**Visual example**:

```
Query tokens:     [q1]  [q2]  [q3]
                    \    |    /
                     \   |   /     Each query token searches
                      v  v  v      for its best match
Document tokens:  [d1] [d2] [d3] [d4]
                   ↑         ↑
                  0.9       0.8    (best matches)

MaxSim = 0.9 + 0.8 + ... (sum of best matches)
```

**Example**: Query "capital of France" (2 tokens) vs document "Paris is the capital of France" (6 tokens):
- Query token "capital" finds best match: `dot("capital", "capital") = 0.95`
- Query token "France" finds best match: `dot("France", "France") = 0.92`
- MaxSim = 0.95 + 0.92 = 1.87

This captures token-level alignment: "capital" and "France" both have strong matches, even if they appear in different parts of the document. Single-vector embeddings average these signals and lose precision.

**When to use**:
- Second-stage reranking (after dense retrieval)
- Precision-critical applications (legal, medical)
- Queries with multiple important terms

**When not to use**:
- First-stage retrieval (too slow for millions of docs)
- Storage-constrained (10-100x larger than dense)

### MMR (Diversity)

**Problem**: Top-k by relevance returns near-duplicates. A search for "async programming" might return 10 Python asyncio tutorials instead of examples in Python, Rust, JavaScript, and Go.

**Solution**: Balance relevance with diversity by penalizing similarity to already-selected items:

$$\text{MMR} = \arg\max_d \left[ \lambda \cdot \text{rel}(d) - (1-\lambda) \cdot \max_{s \in S} \text{sim}(d,s) \right]$$

**Lambda parameter**:
- `λ = 1.0`: Pure relevance (equivalent to top-k)
- `λ = 0.5`: Balanced (common default for RAG)
- `λ = 0.3`: Strong diversity (exploration mode)
- `λ = 0.0`: Maximum diversity (ignore relevance)

**When to use**:
- RAG pipelines (diverse context helps LLMs)
- Recommendation systems (avoid redundancy)
- Search result diversification

**When not to use**:
- Known-item search (single correct answer)
- When relevance is paramount

### Token Pooling

**Storage**: ColBERT stores one vector per token. For 10M documents with 100 tokens each:
- Storage = 10M × 100 × 128 × 4 bytes = 512 GB

**Token pooling**: Cluster similar document tokens and store only cluster centroids:

```
Before:  [tok1] [tok2] [tok3] [tok4] [tok5] [tok6]  (6 vectors)
         similar--^       similar------^
After:   [mean(1,2)]  [tok3] [mean(4,5)] [tok6]    (4 vectors = 33% reduction)
```

**Why it works**: Many tokens are redundant. Function words cluster together, as do related content words. Merging similar tokens loses little discriminative information.

| Factor | Tokens Kept | MRR@10 Loss | When to Use |
|--------|-------------|-------------|-------------|
| 2 | 50% | 0.1–0.3% | Default choice |
| 3 | 33% | 0.5–1.0% | Good tradeoff |
| 4 | 25% | 1.5–3.0% | Storage-constrained (use `hierarchical` feature) |
| 8+ | 12% | 5–10% | Extreme storage constraints only |

Numbers from MS MARCO dev (Clavie et al., 2024). Pool at index time. Re-score with unpooled embeddings at query time if needed.

**When pooling hurts more**:
- Long documents with many distinct concepts
- Queries that need to distinguish between similar tokens
- Short passages (less redundancy to exploit)

## Benchmarks

Measured on Apple M3 Max with `cargo bench`:

| Operation | Dim | Time |
|-----------|-----|------|
| `dot` | 128 | 13ns |
| `dot` | 768 | 126ns |
| `cosine` | 128 | 40ns |
| `cosine` | 768 | 380ns |
| `maxsim` | 32q×128d×128dim | 49μs |
| `maxsim` (pooled 2x) | 32q×64d×128dim | 25μs |
| `maxsim` (pooled 4x) | 32q×32d×128dim | 12μs |

These timings enable real-time reranking of 100-1000 candidates.

## Vendoring

If you prefer not to add a dependency:

- `src/simd.rs` is self-contained (~600 lines)
- AVX2+FMA / NEON with portable fallback
- No dependencies

## Quick Decision Guide

What problem are you solving?

1. **Scoring embeddings** → Use `cosine` or `dot`
   - Dense embeddings: `cosine(&query, &doc)`
   - Pre-normalized: `dot(&query, &doc)`

2. **Reranking with token-level precision** → Use `maxsim_vecs`
   - After dense retrieval (100-1000 candidates)
   - ColBERT-style token embeddings
   - Precision-critical applications

3. **Diversity selection** → Use `mmr_cosine` or `dpp`
   - RAG pipelines (diverse context)
   - Recommendation systems
   - Search result diversification

4. **Compressing token embeddings** → Use `pool_tokens`
   - Storage-constrained deployments
   - Factor 2-3 recommended (50-66% reduction)

**When not to use**:
- First-stage retrieval over millions of docs (use dense + ANN)
- Very short documents (<10 tokens, little benefit over dense)
- Latency-critical first-stage (dense embeddings are faster)
- Storage-constrained without pooling (10-100x larger than dense)

See [REFERENCE.md](REFERENCE.md) for algorithm details and edge cases.

## Features

| Feature | Dependency | Purpose |
|---------|------------|---------|
| `hierarchical` | kodama | Ward's clustering (better at 4x+) |

## See Also

- [rank-fusion](https://crates.io/crates/rank-fusion): merge ranked lists (no embeddings)
- [fastembed-rs](https://github.com/Anush008/fastembed-rs): generate embeddings
- [DESIGN.md](DESIGN.md): architecture decisions
- [REFERENCE.md](REFERENCE.md): algorithm reference

## License

MIT OR Apache-2.0
