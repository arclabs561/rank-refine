# rank-refine

Fast reranking for retrieval pipelines.

```toml
[dependencies]
rank-refine = "0.4"
```

## Pipeline

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

## Modules

| Module | Purpose | Notes |
|--------|---------|-------|
| `matryoshka` | Refine with MRL tail dimensions | Zero deps |
| `colbert` | MaxSim late interaction | Zero deps |
| `crossencoder` | Transformer scoring | Trait-based |
| `simd` | Vector ops | AVX2/NEON auto-dispatch |

## Performance

SIMD-accelerated on x86_64 (AVX2+FMA) and aarch64 (NEON):

| Operation | Speedup |
|-----------|---------|
| dot/cosine | 3× |
| MaxSim 32×128×128 | 3.8× |
| ColBERT 100 docs | 3.2× |

## Matryoshka Refinement

```rust
use rank_refine::matryoshka;

let candidates = vec![("doc1", 0.9), ("doc2", 0.8)];
let query = vec![0.1; 128];
let docs = vec![("doc1", vec![0.2; 128]), ("doc2", vec![0.15; 128])];

let refined = matryoshka::refine(&candidates, &query, &docs, 64);
```

## ColBERT MaxSim

```rust
use rank_refine::colbert;

let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let docs = vec![
    ("doc1", vec![vec![0.9, 0.1], vec![0.1, 0.9]]),
    ("doc2", vec![vec![0.5, 0.5]]),
];

let ranked = colbert::rank(&query, &docs);
```

## Cross-Encoder (BYOM)

Bring your own model via the `CrossEncoderModel` trait:

```rust
use rank_refine::crossencoder::{CrossEncoderModel, rerank, Score};

struct MyEncoder;
impl CrossEncoderModel for MyEncoder {
    fn score_batch(&self, query: &str, docs: &[&str]) -> Vec<Score> {
        // Your inference here
        docs.iter().map(|_| 0.5).collect()
    }
}

let ranked = rerank(&MyEncoder, "query", &[("d1", "doc text")]);
```

## License

MIT OR Apache-2.0
