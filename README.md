# rank-refine

Rerank retrieval results with expensive-but-accurate methods.

[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

## When to Use This

You have top-K candidates from initial retrieval and want to re-score them for better precision:

- **Matryoshka refinement**: You used truncated embeddings for fast retrieval, now refine with full dimensions
- **ColBERT/MaxSim**: You have token-level embeddings and want late interaction scoring
- **Cross-encoder**: You want transformer-based pairwise scoring (BYOM)

## Quick Start

```toml
[dependencies]
rank-refine = "0.5"
```

## Matryoshka Refinement

### What It Is

[Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) (MRL) trains embeddings so that **prefixes are valid lower-dimensional embeddings**. A 768-dim embedding's first 64 dims work as a coarse 64-dim embedding.

### Supported Models

These Hugging Face models support Matryoshka:

| Model | Full Dims | Recommended Coarse |
|-------|-----------|-------------------|
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 64, 128, 256 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | 256, 512 |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 64, 128, 256, 512 |
| `jinaai/jina-embeddings-v3` | 1024 | 32, 64, 128, 256, 512 |

### Two-Stage Retrieval Pattern

```rust
use rank_refine::matryoshka;

// Stage 1: Fast coarse retrieval with first 64 dims
// (done in your vector DB - configure index for 64 dims)
let candidates = vector_db.search(&query_embedding[..64], top_k: 100);

// Stage 2: Refine with tail dimensions (64..768)
let full_query: Vec<f32> = embed("your query"); // 768 dims
let full_docs: Vec<(&str, Vec<f32>)> = candidates
    .iter()
    .map(|id| (*id, get_full_embedding(id)))
    .collect();

let refined = matryoshka::refine(
    &candidates,      // (id, coarse_score) from stage 1
    &full_query,      // full 768-dim query
    &full_docs,       // full 768-dim docs
    64,               // head_dims used in coarse retrieval
);
```

### Why This Works

- **Stage 1**: Search 64-dim index → 4-8× faster, fits more in memory
- **Stage 2**: Refine top-100 using dims 64..768 → higher precision

The tail dimensions capture fine-grained semantics that coarse retrieval misses.

### Tuning Alpha

```rust
use rank_refine::matryoshka::refine_with_alpha;

// alpha controls blend: final = alpha * original + (1-alpha) * tail_similarity
let refined = refine_with_alpha(&candidates, &query, &docs, 64, 0.3);
// 0.3 = 30% original score, 70% tail similarity
```

- `alpha=0.0`: Ignore original scores, pure tail refinement
- `alpha=0.5`: Equal blend (default)
- `alpha=1.0`: Keep original scores unchanged

## ColBERT / MaxSim

### What It Is

[ColBERT](https://arxiv.org/abs/2004.12832) represents queries and documents as bags of token embeddings, then scores via MaxSim:

```
score = Σ (max similarity between each query token and any doc token)
```

This "late interaction" captures token-level alignment that single-vector models miss.

### Supported Models

| Model | Dims | Notes |
|-------|------|-------|
| `colbert-ir/colbertv2.0` | 128 | Original ColBERT v2 |
| `answerdotai/answerai-colbert-small-v1` | 96 | Smaller, faster |
| `jinaai/jina-colbert-v2` | 128 | Multilingual |

### Usage

```rust
use rank_refine::colbert;

// Query: 32 tokens × 128 dims (from ColBERT tokenization)
let query_tokens: Vec<Vec<f32>> = colbert_encode_query("what is rust?");

// Documents with their token embeddings
let docs: Vec<(&str, Vec<Vec<f32>>)> = candidates
    .iter()
    .map(|id| (*id, colbert_encode_doc(&get_text(id))))
    .collect();

// Rank by MaxSim
let ranked = colbert::rank(&query_tokens, &docs);
```

### Refining with Blend

```rust
// Blend MaxSim with original retrieval scores
let refined = colbert::refine(
    &candidates,      // (id, original_score)
    &query_tokens,
    &docs,
    0.5,              // alpha: 50% original, 50% MaxSim
);
```

### Getting Token Embeddings

ColBERT requires token-level embeddings, not pooled. With the `candle` ecosystem:

```rust
// Pseudocode - actual implementation depends on your setup
fn colbert_encode(text: &str, model: &ColBertModel) -> Vec<Vec<f32>> {
    let tokens = tokenizer.encode(text);
    let outputs = model.forward(&tokens);  // [seq_len, hidden_dim]
    
    // Project to ColBERT dimension and normalize
    let projected = linear_projection(outputs);  // [seq_len, 128]
    projected.iter().map(|tok| normalize(tok)).collect()
}
```

## Cross-Encoder (Bring Your Own Model)

### What It Is

Cross-encoders score query-document pairs directly with a transformer, giving the highest precision but requiring O(n) forward passes.

### The Trait

```rust
use rank_refine::crossencoder::{CrossEncoderModel, rerank, Score};

trait CrossEncoderModel {
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<Score>;
}
```

### Example with candle

```rust
use rank_refine::crossencoder::{CrossEncoderModel, rerank, Score};

struct MiniLMCrossEncoder {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl CrossEncoderModel for MiniLMCrossEncoder {
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<Score> {
        documents.iter().map(|doc| {
            // Concatenate: [CLS] query [SEP] doc [SEP]
            let input = format!("{} [SEP] {}", query, doc);
            let tokens = self.tokenizer.encode(&input);
            let logits = self.model.forward(&tokens);
            logits[0]  // Classification head output
        }).collect()
    }
}

// Usage
let model = MiniLMCrossEncoder::load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;
let candidates = vec![
    ("doc1", "Rust is a systems programming language."),
    ("doc2", "Python is popular for data science."),
    ("doc3", "Rust prevents memory safety bugs."),
];

let ranked = rerank(&model, "what is rust?", &candidates);
// doc1 and doc3 rank higher than doc2
```

### Recommended Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Medium | Better |
| `BAAI/bge-reranker-base` | 278M | Slow | Best |
| `BAAI/bge-reranker-v2-m3` | 568M | Slowest | SOTA |

## Full Pipeline Example

```rust
use rank_fusion::{rrf, RrfConfig};
use rank_refine::{matryoshka, crossencoder::{CrossEncoderModel, rerank}};

async fn search(query: &str, top_k: usize) -> Vec<Document> {
    // 1. Embed query (Matryoshka model)
    let query_embedding = embed(query);  // 768 dims

    // 2. Coarse retrieval (parallel BM25 + truncated vectors)
    let (bm25, vectors) = tokio::join!(
        bm25_search(query, 100),
        vector_search(&query_embedding[..64], 100),  // 64-dim coarse
    );

    // 3. Fuse
    let fused = rrf(bm25, vectors, RrfConfig::default());

    // 4. Matryoshka refinement on top 50
    let top_50: Vec<_> = fused.into_iter().take(50).collect();
    let full_docs = load_full_embeddings(&top_50);
    let refined = matryoshka::refine(&top_50, &query_embedding, &full_docs, 64);

    // 5. Cross-encoder rerank on top 20
    let top_20: Vec<_> = refined.into_iter().take(20).collect();
    let texts = load_texts(&top_20);
    let reranked = rerank(&cross_encoder, query, &texts);

    // 6. Return top_k
    reranked.into_iter().take(top_k).collect()
}
```

## SIMD Acceleration

Vector operations (dot product, cosine, MaxSim) use SIMD when available:
- **x86_64**: AVX2 + FMA (runtime detection)
- **aarch64**: NEON (always enabled)

No configuration needed — dispatch is automatic.

## Caveats

- **Candidates not in docs are dropped**: `matryoshka::refine` and `colbert::refine` silently skip candidates whose embeddings aren't provided
- **Empty inputs**: MaxSim with empty query returns 0; with empty docs returns negative infinity per query token
- **Memory**: ColBERT stores per-token embeddings — a 512-token doc × 128 dims = 256KB per doc

## License

MIT OR Apache-2.0
