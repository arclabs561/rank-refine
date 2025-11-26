# rank-refine

Model-based reranking for retrieval pipelines.

[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**Status:** Early development. See [DESIGN.md](DESIGN.md) for roadmap.

## The Two-Crate Strategy

```
Retrieve (BM25, Dense) → Fuse (rank-fusion) → Rerank (rank-refine) → Top-K
```

| Crate | Purpose | Dependencies |
|-------|---------|--------------|
| [rank-fusion](https://crates.io/crates/rank-fusion) | Combine result lists | Zero |
| rank-refine | Re-score with models | candle/ort (optional) |

## Planned Features

- **Matryoshka refinement** — Refine coarse results using tail dimensions (SIMD)
- **Cross-encoder** — Score pairs with quantized transformers
- **ColBERT/PLAID MaxSim** — Late interaction scoring

## License

MIT OR Apache-2.0

