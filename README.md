# rank-refine

Reranking for retrieval pipelines.

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)
[![MSRV](https://img.shields.io/badge/MSRV-1.74-blue)](https://blog.rust-lang.org/2023/11/16/Rust-1.74.0.html)

```rust
use rank_refine::matryoshka;

let candidates = vec![("d1", 0.9), ("d2", 0.8)];
let query = vec![0.1, 0.2, 0.3, 0.4];
let docs = vec![("d1", vec![0.1, 0.2, 0.3, 0.4]), ("d2", vec![0.1, 0.2, 0.3, 0.4])];
let refined = matryoshka::refine(&candidates, &query, &docs, 2);
```

## Modules

| Module | Method |
|--------|--------|
| `matryoshka` | MRL tail dimension refinement |
| `colbert` | MaxSim late interaction |
| `crossencoder` | Cross-encoder trait (BYOM) |
| `simd` | Vector ops (AVX2/NEON) |

## Related

See [`rank-fusion`](https://crates.io/crates/rank-fusion) for combining ranked lists (RRF, CombMNZ, Borda).

## License

MIT OR Apache-2.0
