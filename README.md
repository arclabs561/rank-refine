# rank-refine

Reranking with embeddings. SIMD-accelerated.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

```
cargo add rank-refine
```

## Usage

```rust
use rank_refine::simd::{cosine, maxsim_vecs};

// Dense
let score = cosine(&query, &doc);

// Late interaction (ColBERT)
let score = maxsim_vecs(&query_tokens, &doc_tokens);
```

## API

### Similarity

| Function | Input | Notes |
|----------|-------|-------|
| `cosine(a, b)` | `&[f32]` | Normalized, -1 to 1 |
| `dot(a, b)` | `&[f32]` | Unnormalized |
| `maxsim(q, d)` | `&[&[f32]]` | Sum of max similarities |
| `maxsim_cosine(q, d)` | `&[&[f32]]` | Cosine variant |
| `maxsim_weighted(q, d, w)` | `&[&[f32]], &[f32]` | Per-token weights |
| `maxsim_batch(q, docs)` | `&[Vec<f32>], &[Vec<Vec<f32>>]` | Batch scoring |

### Token Pooling

| Function | Compression | Quality |
|----------|-------------|---------|
| `pool_tokens(t, 2)` | 50% | ~0% loss |
| `pool_tokens(t, 4)` | 75% | 2-5% loss |
| `pool_tokens_adaptive(t, f)` | varies | auto-selects method |
| `pool_tokens_with_protected(t, f, n)` | varies | preserves first n tokens |

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

### Traits

```rust
// Cross-encoder: implement this
trait CrossEncoderModel {
    fn score(&self, query: &str, doc: &str) -> f32;
}

// Token pooling: implement this or use builtins
trait Pooler {
    fn pool(&self, tokens: &[Vec<f32>], target: usize) -> Vec<Vec<f32>>;
}
```

### Types

```rust
// Compile-time normalized guarantee
let n: Normalized = normalize(&v)?;
n.dot(&n) == 1.0

// Masked tokens (attention mask applied)
let m = MaskedTokens::with_mask(&tokens, &mask);
```

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

| Factor | Storage Saved | Quality Loss |
|--------|---------------|--------------|
| 2 | 50% | ~0.1% |
| 3 | 66% | ~0.8% |
| 4 | 75% | ~2.1% |

Quality loss measured on MS MARCO (Clavie et al., 2024).

## Benchmarks

M3 Max, `cargo bench`:

| Operation | Dim | Time |
|-----------|-----|------|
| `dot` | 128 | 13ns |
| `dot` | 768 | 126ns |
| `cosine` | 128 | 40ns |
| `cosine` | 768 | 380ns |
| `maxsim` | 32q×128d×128dim | 49μs |
| `maxsim` (pooled 2x) | 32q×64d×128dim | 25μs |
| `maxsim` (pooled 4x) | 32q×32d×128dim | 12μs |

## Features

| Feature | Dependency | Purpose |
|---------|------------|---------|
| `hierarchical` | kodama | Ward's clustering (better at 4x+) |

## See Also

- [rank-fusion](https://crates.io/crates/rank-fusion): merge ranked lists (no embeddings)
- [DESIGN.md](DESIGN.md): architecture
- [REFERENCE.md](REFERENCE.md): algorithms

## License

MIT OR Apache-2.0
