# rank-refine

Reranking algorithms for retrieval pipelines.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)
[![MSRV](https://img.shields.io/badge/MSRV-1.74-blue)](https://blog.rust-lang.org/2023/11/16/Rust-1.74.0.html)

## Why This Library?

Reranking improves RAG quality by **up to 48%** (Pinecone benchmarks) but requires
different optimization than vector databases:

| Step | Optimizes For | Scale |
|------|---------------|-------|
| Retrieval | Millions of vectors, ANN | Fast, approximate |
| **Reranking** | 20-100 candidates, exact | Precise, SIMD-optimized |

A dedicated reranking library lets you:
- **Swap databases** without rewriting scoring logic
- **Mix strategies** (dense + late interaction + cross-encoder)
- **Scale independently** from your vector infrastructure

## Design: Bring Your Own Model (BYOM)

This crate provides **scoring algorithms only** — no model weights, no inference.
You bring embeddings from your preferred source:

| Embedding Source | Crate | Notes |
|------------------|-------|-------|
| [fastembed](https://crates.io/crates/fastembed) | `fastembed` | ONNX, many models |
| [candle](https://github.com/huggingface/candle) | `candle-transformers` | Pure Rust |
| [ort](https://crates.io/crates/ort) | `ort` | ONNX Runtime |
| sentence-transformers | Python → JSON/bincode | Via serde |

## Quick Start

```rust
use rank_refine::prelude::*;

// 1. Get embeddings from your model (fastembed, candle, etc.)
let query_emb: Vec<f32> = /* your_model.embed("query") */;
let doc_embs: Vec<Vec<f32>> = /* your_model.embed_batch(docs) */;

// 2. Score with rank-refine
let score = cosine(&query_emb, &doc_embs[0]);

// For ColBERT token embeddings:
let query_tokens: Vec<Vec<f32>> = /* colbert_model.encode_query("query") */;
let doc_tokens: Vec<Vec<f32>> = /* colbert_model.encode_doc("doc") */;
let maxsim = maxsim_vecs(&query_tokens, &doc_tokens);
```

## Prelude

Import common types with one line:

```rust
use rank_refine::prelude::*;
// Traits: DenseScorer, LateInteractionScorer, Pooler, Scorer, TokenScorer
// SIMD: cosine, dot, maxsim, maxsim_cosine, maxsim_vecs, norm
// Matryoshka: mrl_refine, mrl_try_refine, RefineConfig
// ColBERT: pool_tokens, colbert_rank
// Cross-encoder: CrossEncoderModel
// Utilities: as_slices, RefineError
```

## End-to-End Example with fastembed

```toml
# Cargo.toml
[dependencies]
rank-refine = "0.7"
fastembed = "4"
```

```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use rank_refine::scoring::{DenseScorer, Scorer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize model (downloads on first run, ~30MB)
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        ..Default::default()
    })?;

    // Embed
    let query = model.embed(vec!["What is Rust?"], None)?;
    let docs = model.embed(vec![
        "Rust is a systems programming language.",
        "Python is great for data science.",
    ], None)?;

    // Rank with rank-refine
    let scorer = DenseScorer::Cosine;
    let doc_refs: Vec<(usize, &[f32])> = docs.iter()
        .enumerate()
        .map(|(i, e)| (i, e.as_slice()))
        .collect();
    let ranked = scorer.rank(&query[0], &doc_refs);

    println!("Best match: doc {}", ranked[0].0);
    Ok(())
}
```

## When to Use Which Algorithm

| You Have | Use | Why |
|----------|-----|-----|
| Dense embeddings (bi-encoder) | `cosine` / `dot` | Fast, SIMD-accelerated |
| Token embeddings (ColBERT) | `maxsim_vecs` | Late interaction captures fine-grained similarity |
| Matryoshka (MRL) embeddings | `mrl_refine` | Tail dimensions refine coarse prefix scores |
| Model you can call | `CrossEncoderModel` trait | Most accurate, but requires inference |

### Indexing vs Query Time

Late interaction (ColBERT) requires pre-computed token embeddings. The workflow:

```
INDEXING (once per document):
  doc → model.encode() → pool_tokens() → store in DB

QUERY (per request):
  query → model.encode() → rank-refine → top-K
  candidates from DB  ────────┘
```

Functions like `pool_tokens` are for **indexing time** — run them once and store results.
Functions like `maxsim_vecs` and `rank` are for **query time** — run them per query on candidates.

## Modules

| Module | Purpose |
|--------|---------|
| `matryoshka` | MRL tail dimension refinement |
| `colbert` | MaxSim late interaction + token pooling |
| `crossencoder` | Cross-encoder trait (implement for your model) |
| `simd` | Vector ops (AVX2/NEON accelerated) |
| `scoring` | Unified `Scorer`, `TokenScorer`, `Pooler` traits |

## Features

| Feature | Description |
|---------|-------------|
| `hierarchical` | Ward's method clustering via `kodama` (better pooling at 4x+ compression) |

## Related

- [`rank-fusion`](https://crates.io/crates/rank-fusion) — Combine ranked lists (RRF, CombMNZ, Borda)

## License

MIT OR Apache-2.0
