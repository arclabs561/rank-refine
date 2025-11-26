# rank-refine

Model-based reranking for retrieval pipelines.

**Status:** Stub. See [DESIGN.md](DESIGN.md) for roadmap.

## Two-Crate Approach

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

| Crate | Purpose | Dependencies |
|-------|---------|--------------|
| [rank-fusion](https://crates.io/crates/rank-fusion) | Combine result lists | None |
| rank-refine | Re-score with models | candle/ort (optional) |

## Planned Features

- **Matryoshka refinement** — Refine coarse results using tail dimensions (SIMD, no deps)
- **Cross-encoder** — Score pairs with transformers (candle or ort)
- **ColBERT/PLAID MaxSim** — Late interaction scoring

## License

MIT OR Apache-2.0
