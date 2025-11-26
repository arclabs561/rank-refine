# rank-refine Design

## What This Is

Fast reranking for retrieval pipelines. Takes candidates and re-scores them.

## Implemented (v0.4.0)

### Matryoshka Refinement
- Uses tail dimensions of MRL embeddings
- Blends original score with tail similarity
- Zero external dependencies

### ColBERT MaxSim
- Late interaction: `score = Σ_q max_d(q · d)`
- `rank()` for fresh scoring
- `refine()` to blend with existing scores

### Cross-Encoder (Trait)
- `CrossEncoderModel` trait for BYOM
- `rerank()` and `refine()` generic functions
- Bring your own inference backend

### SIMD Acceleration
- AVX2+FMA on x86_64 (runtime detection)
- NEON on aarch64 (always enabled)
- ~3× speedup on vector operations

## Benchmarks (M-series Mac)

| Operation | Time |
|-----------|------|
| dot 128-dim | 13ns |
| cosine 768-dim | 380ns |
| MaxSim 32×128×128 | 48µs |
| ColBERT 100 docs | 2.9ms |

## Architecture

```
rank-fusion (combine lists) → rank-refine (rescore) → Top-K
```

## Future

1. **Candle integration** — Optional feature for cross-encoder inference
2. **Batch SIMD** — Process multiple vectors at once
3. **Quantization** — int8/binary dot products

## References

- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [ColBERT](https://arxiv.org/abs/2004.12832)
