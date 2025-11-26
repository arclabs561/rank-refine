# rank-refine

Reranking for retrieval pipelines. Re-scores candidates with expensive methods.

[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

## Usage

```toml
[dependencies]
rank-refine = "0.5"
```

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
```

Supported models: `nomic-embed-text-v1.5`, `gte-large-en-v1.5`, `mxbai-embed-large-v1`, `jina-embeddings-v3`.

## ColBERT / MaxSim

For [ColBERT](https://arxiv.org/abs/2004.12832) token-level embeddings.

```rust
use rank_refine::colbert;

// query_tokens: Vec<Vec<f32>> - one embedding per token
// docs: Vec<(id, Vec<Vec<f32>>)> - token embeddings per doc
let ranked = colbert::rank(&query_tokens, &docs);
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

## SIMD

Vector ops use AVX2+FMA (x86_64) or NEON (aarch64) when available. No configuration needed.

## Caveats

- Candidates not in `docs` are dropped silently
- Empty query in MaxSim returns 0

## License

MIT OR Apache-2.0
