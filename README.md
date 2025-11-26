# rank-refine

Reranking for retrieval pipelines.

```toml
[dependencies]
rank-refine = "0.5"
```

## Pipeline

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

## Modules

| Module | Purpose |
|--------|---------|
| `matryoshka` | Refine using MRL tail dimensions |
| `colbert` | MaxSim late interaction scoring |
| `crossencoder` | Cross-encoder trait (BYOM) |
| `simd` | Vector ops (AVX2/NEON auto-dispatch) |

## Matryoshka Refinement

```rust
use rank_refine::matryoshka;

let candidates = vec![("doc1", 0.9), ("doc2", 0.8)];
let query = vec![0.1; 128];
let docs = vec![("doc1", vec![0.2; 128]), ("doc2", vec![0.15; 128])];

// Coarse retrieval used dims 0..64, refine with 64..128
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

```rust
use rank_refine::crossencoder::{CrossEncoderModel, rerank, Score};

struct MyEncoder;
impl CrossEncoderModel for MyEncoder {
    fn score_batch(&self, query: &str, docs: &[&str]) -> Vec<Score> {
        docs.iter().map(|_| 0.5).collect()
    }
}

let ranked = rerank(&MyEncoder, "query", &[("d1", "doc text")]);
```

## License

MIT OR Apache-2.0
