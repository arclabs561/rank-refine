# rank-refine

Reranking for retrieval pipelines. Re-scores candidates with expensive methods.

[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

## Usage

```toml
[dependencies]
rank-refine = "0.7"
```

## When to Use What

| Method | Best For | Latency | Storage |
|--------|----------|---------|---------|
| `crossencoder` | Highest accuracy | High (model inference) | Low (text only) |
| `colbert` | Token-level precision | Medium (SIMD ops) | High (token vectors) |
| `matryoshka` | Two-stage retrieval | Low (vector tail) | Medium (full embedding) |

**Cross-encoders** are most accurate but expensive. Use for top-K refinement (K ≤ 100).

**ColBERT/MaxSim** provides token-level matching without encoding query-doc pairs together. Good for longer documents where specific passages matter.

**Matryoshka** is ideal when you already have MRL embeddings. Coarse retrieval uses the prefix; refinement uses the tail. Zero model calls.

## Modules

| Module | Method |
|--------|--------|
| `matryoshka` | Refine with MRL tail dimensions |
| `colbert` | MaxSim late interaction |
| `crossencoder` | Cross-encoder trait (BYOM) |
| `simd` | Vector ops (AVX2/NEON) |

## Matryoshka Refinement

For [Matryoshka embeddings](https://arxiv.org/abs/2205.13147) where prefixes are valid lower-dim embeddings.

```rust
use rank_refine::matryoshka;

// Stage 1: coarse retrieval with truncated embeddings (e.g., first 64 dims)
// Stage 2: refine with tail dimensions
let refined = matryoshka::refine(&candidates, &query, &docs, 64);

// With custom blending (0.0 = all tail, 1.0 = all original)
let refined = matryoshka::refine_with_alpha(&candidates, &query, &docs, 64, 0.3);

// Fallible version for validated inputs
let result = matryoshka::try_refine(&candidates, &query, &docs, 64, RefineConfig::default());
```

Supported models: `nomic-embed-text-v1.5`, `gte-large-en-v1.5`, `mxbai-embed-large-v1`, `jina-embeddings-v3`.

## ColBERT / MaxSim

For [ColBERT](https://arxiv.org/abs/2004.12832) token-level embeddings.

```rust
use rank_refine::colbert;

// query_tokens: Vec<Vec<f32>> - one embedding per token
// docs: Vec<(id, Vec<Vec<f32>>)> - token embeddings per doc
let ranked = colbert::rank(&query_tokens, &docs);

// With top_k limit
let ranked = colbert::rank_with_top_k(&query_tokens, &docs, Some(10));

// Token pooling reduces storage 50-66% with minimal quality loss
let pooled = colbert::pool_tokens(&doc_tokens, 2); // pool_factor=2

// Protected tokens (e.g., [CLS]) preserved
let pooled = colbert::pool_tokens_with_protected(&doc_tokens, 2, 1);
```

Supported models: `colbertv2.0`, `answerai-colbert-small-v1`, `jina-colbert-v2`.

## Cross-Encoder

Trait-based. Bring your own inference backend.

```rust
use rank_refine::crossencoder::{CrossEncoderModel, rerank};

impl CrossEncoderModel for YourModel {
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        // your inference code
    }
}

let ranked = rerank(&model, "query", &[("id1", "text1"), ("id2", "text2")]);
```

Models: `ms-marco-MiniLM-L-6-v2`, `bge-reranker-base`, `bge-reranker-v2-m3`.

## Configuration

```rust
use rank_refine::RefineConfig;

// Builder pattern
let config = RefineConfig::default()
    .with_alpha(0.7)   // 70% original, 30% refined
    .with_top_k(10);   // return top 10

// Presets
let config = RefineConfig::refinement_only();  // alpha = 0.0
let config = RefineConfig::original_only();    // alpha = 1.0
```

## Error Handling

Functions have `try_*` variants that return `Result<T, RefineError>`:

```rust
use rank_refine::{matryoshka, RefineConfig, RefineError};

match matryoshka::try_refine(&candidates, &query, &docs, 64, RefineConfig::default()) {
    Ok(refined) => { /* use refined */ }
    Err(RefineError::InvalidHeadDims { head_dims, query_len }) => {
        eprintln!("head_dims ({head_dims}) must be < query.len() ({query_len})");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

## SIMD

Vector ops use AVX2+FMA (x86_64) or NEON (aarch64) when available. No configuration needed.

## Caveats

- Candidates not in `docs` are dropped silently
- Empty query in MaxSim returns 0
- Short doc embeddings (< head_dims) are skipped in matryoshka

## Related

- [rank-fusion](https://crates.io/crates/rank-fusion) — Combine results from multiple retrievers (RRF, CombMNZ, Borda)

## License

MIT OR Apache-2.0
