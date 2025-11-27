# rank-refine

SIMD-accelerated similarity scoring for vector search and RAG.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

```
cargo add rank-refine
```

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

For each query token, find its best-matching document token, then sum:

$$\text{score}(Q, D) = \sum_{i} \max_{j} (q_i \cdot d_j)$$

This preserves token-level semantics that single-vector embeddings lose.

### MMR (Diversity)

Balance relevance with diversity by penalizing similarity to already-selected items:

$$\text{MMR} = \arg\max_d \left[ \lambda \cdot \text{rel}(d) - (1-\lambda) \cdot \max_{s \in S} \text{sim}(d,s) \right]$$

### Token Pooling

Cluster similar document tokens to reduce storage:

| Factor | Tokens Kept | MRR@10 Loss |
|--------|-------------|-------------|
| 2 | 50% | 0.1–0.3% |
| 3 | 33% | 0.5–1.0% |
| 4 | 25% | 1.5–3.0% |

Numbers from MS MARCO dev (Clavie et al., 2024). Pooling merges similar tokens; it hurts when queries need to distinguish between them.

## Benchmarks

Apple M3 Max, `cargo bench`:

| Operation | Dim | Time |
|-----------|-----|------|
| `dot` | 128 | 13ns |
| `dot` | 768 | 126ns |
| `cosine` | 128 | 40ns |
| `cosine` | 768 | 380ns |
| `maxsim` | 32q×128d×128dim | 49μs |
| `maxsim` (pooled 2x) | 32q×64d×128dim | 25μs |
| `maxsim` (pooled 4x) | 32q×32d×128dim | 12μs |

## Vendoring

If you prefer not to add a dependency:

- `src/simd.rs` is self-contained (~600 lines)
- AVX2+FMA / NEON with portable fallback
- No dependencies

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
