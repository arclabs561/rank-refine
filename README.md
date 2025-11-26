# rank-refine

Reranking for retrieval pipelines.

```toml
[dependencies]
rank-refine = "0.3"
```

## Pipeline

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

## Modules

| Module | Use Case |
|--------|----------|
| `matryoshka` | Refine with tail dimensions of MRL embeddings |
| `colbert` | MaxSim scoring for token-level embeddings |
| `simd` | Vector ops (dot, cosine, maxsim) |

## Matryoshka Refinement

Two-stage retrieval with MRL embeddings:

```rust
use rank_refine::matryoshka;

let candidates = vec![("doc1", 0.9), ("doc2", 0.8)];
let query = vec![0.1; 128];
let docs = vec![
    ("doc1", vec![0.2; 128]),
    ("doc2", vec![0.15; 128]),
];

// Coarse retrieval used dims 0..64, refine with 64..128
let refined = matryoshka::refine(&candidates, &query, &docs, 64);
```

## ColBERT MaxSim

Late interaction scoring with token embeddings:

```rust
use rank_refine::colbert;

let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];  // 2 tokens
let docs = vec![
    ("doc1", vec![vec![0.9, 0.1], vec![0.1, 0.9]]),
    ("doc2", vec![vec![0.5, 0.5]]),
];

let ranked = colbert::rank(&query, &docs);
```

## License

MIT OR Apache-2.0
