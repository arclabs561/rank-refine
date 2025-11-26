# Design

Two crates for retrieval pipelines:

```
Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
```

## Modules

- **matryoshka** — Two-stage retrieval: coarse with head dims, refine with tail
- **colbert** — MaxSim late interaction: `Σ_q max_d(q·d)`
- **crossencoder** — BYOM trait for transformer-based scoring
- **simd** — AVX2+FMA (x86_64), NEON (aarch64), portable fallback

## References

- [Matryoshka](https://arxiv.org/abs/2205.13147) — MRL embeddings
- [ColBERT](https://arxiv.org/abs/2004.12832) — late interaction
