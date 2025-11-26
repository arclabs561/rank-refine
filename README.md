# rank-refine

Reranking algorithms for retrieval pipelines.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)
[![MSRV](https://img.shields.io/badge/MSRV-1.74-blue)](https://blog.rust-lang.org/2023/11/16/Rust-1.74.0.html)

## The Retrieval Funnel Problem

You want to run expensive, accurate models on your documents. But you have millions of them.

```
                    ┌───────────────────┐
                    │  10M documents    │  Can't run BERT on all of these
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
   Initial          │  ANN / BM25       │  Fast, approximate
   Retrieval        │  ~10ms            │  Recall: 90%, Precision: 30%
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Top 100          │  Small enough for expensive ops
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
   Reranking        │  Cross-encoder    │  Slow, accurate
   (rank-refine)    │  ColBERT MaxSim   │  
                    │  MRL refinement   │  
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Top 10           │  High precision for LLM context
                    └───────────────────┘
```

The compute-accuracy tradeoff is fundamental:
- **Bi-encoders** (initial retrieval): Encode query and docs separately, compare with dot product. O(1) per doc.
- **Cross-encoders** (reranking): Encode query+doc together, attend across both. O(n) per doc, but much more accurate.

You can't afford cross-encoders on millions of docs. But you can afford them on 100.

## What This Crate Does

rank-refine provides the **scoring algorithms** for the reranking stage. No model weights,
no inference runtime—you bring embeddings from fastembed, candle, ort, or serialize from Python.

```rust
use rank_refine::prelude::*;

// Dense scoring (SIMD-accelerated)
let score = cosine(&query_emb, &doc_emb);

// Late interaction (ColBERT MaxSim)
let score = maxsim_vecs(&query_tokens, &doc_tokens);
```

## Algorithms: When to Use What

| You Have | Method | Accuracy | Speed | When to Use |
|----------|--------|----------|-------|-------------|
| Dense embeddings | `cosine` / `dot` | Baseline | 0.1μs | Fast re-scoring |
| Token embeddings (ColBERT) | `maxsim_vecs` | +5-10% | 50μs | Fine-grained matching |
| Matryoshka embeddings | `mrl_refine` | +2-5% | 1μs | Two-stage MRL |
| Any model callable | `CrossEncoderModel` | +10-20% | 10ms | Maximum accuracy |

**Dense (cosine/dot)**: Your baseline. SIMD-accelerated, works with any bi-encoder output.

**ColBERT MaxSim**: Instead of one embedding per doc, you have one per token. MaxSim finds
the best token-level match for each query token, then sums. Captures fine-grained similarity
that single-vector embeddings miss.

**Matryoshka (MRL)**: Some embedding models (e.g., `nomic-embed-text-v1.5`) are trained so
that prefixes of the embedding are useful. You store 256-dim, retrieve with first 64 dims,
refine with full 256. The tail dimensions add signal without re-embedding.

**Cross-encoder**: The accuracy ceiling. You implement the `CrossEncoderModel` trait with
your inference backend (ONNX, candle, etc.). Query and doc are encoded together with
full cross-attention.

## Token Pooling (ColBERT Storage Reduction)

ColBERT's per-token embeddings are expensive to store. If your docs average 200 tokens
at 128 dims, that's 100KB per doc. Pooling reduces this:

```rust
use rank_refine::colbert::pool_tokens;

let tokens: Vec<Vec<f32>> = embed_doc(doc);  // 200 x 128
let pooled = pool_tokens(&tokens, 4);         // 50 x 128 (4x reduction)
```

Pooling clusters similar tokens. For 4x+ compression, enable the `hierarchical` feature
for Ward's method clustering instead of greedy agglomerative.

## SIMD Acceleration

Vector operations auto-dispatch to:
- **x86_64**: AVX2 + FMA when available
- **aarch64**: NEON

No runtime flags needed. Fallback to portable code on other platforms.

## Quick Start

```rust
use rank_refine::prelude::*;

// Get embeddings from your model (fastembed, candle, etc.)
let query_emb: Vec<f32> = model.embed("query");
let doc_embs: Vec<Vec<f32>> = model.embed_batch(&docs);

// Score with SIMD-accelerated cosine
let scores: Vec<f32> = doc_embs.iter()
    .map(|d| cosine(&query_emb, d))
    .collect();

// Or use the Scorer trait
let scorer = DenseScorer::Cosine;
let ranked = scorer.rank(&query_emb, &doc_refs);
```

## Modules

| Module | Purpose |
|--------|---------|
| `simd` | Vector ops (dot, cosine, maxsim) — AVX2/NEON accelerated |
| `colbert` | MaxSim late interaction + token pooling |
| `matryoshka` | MRL tail dimension refinement |
| `crossencoder` | Trait for cross-encoder models |
| `scoring` | Unified `Scorer`, `TokenScorer`, `Pooler` traits |

## Features

| Feature | Description |
|---------|-------------|
| `hierarchical` | Ward's method clustering for better pooling at 4x+ compression |

## Related

- [`rank-fusion`](https://crates.io/crates/rank-fusion) — Combine ranked lists (RRF, CombMNZ, Borda)

## License

MIT OR Apache-2.0
