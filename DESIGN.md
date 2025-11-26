# rank-refine Design

## What This Is

Reranking algorithms for retrieval pipelines. Takes candidates from initial retrieval and re-scores them.

## Implemented (v0.3.0)

### Matryoshka Refinement
- Uses tail dimensions of MRL embeddings for cheap refinement
- No model inference needed, just vector math
- Blends original score with tail similarity

### ColBERT MaxSim
- Late interaction scoring: `score = Σ_q max_d(q · d)`
- Works with any token-level embeddings
- `rank()` for fresh scoring, `refine()` to blend with existing scores

### SIMD Module
- Portable implementations that auto-vectorize
- `dot`, `norm`, `cosine`, `maxsim`
- Future: explicit AVX2/NEON intrinsics

## Architecture

```
rank-fusion (combine lists) → rank-refine (rescore single list) → Top-K
```

These are separate concerns:
- Fusion: zero-dep, pure rank manipulation
- Refinement: may need vectors, models, heavier compute

## Future Directions

1. **Cross-encoder** — Score pairs with transformers (candle feature flag)
2. **SIMD intrinsics** — Explicit AVX2/NEON for 3-5x speedup
3. **Batch scoring** — Amortize overhead for many candidates

## References

- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [PLAID](https://arxiv.org/abs/2205.09707)
