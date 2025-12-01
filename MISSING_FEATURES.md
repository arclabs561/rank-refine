# Missing Features from Research Papers

This document tracks features mentioned in ColBERT/ColPali research papers that are not yet implemented in `rank-refine`.

## Residual Compression (ColBERTv2)

**Paper**: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)

**What it is**: A compression technique that stores token embeddings as:
- Centroid index (which cluster the token belongs to)
- Quantized residual (difference from centroid, stored in 1-2 bits per dimension)

**Benefits**: 6-10× storage reduction with minimal quality loss

**Current status**: Mentioned in documentation but not implemented. We have token pooling instead, which is simpler and doesn't require centroid training.

**Implementation complexity**: High - requires:
1. K-means clustering to find centroids
2. Quantization of residuals (1-2 bit per dimension)
3. Decompression at query time

**Trade-off**: Token pooling (what we have) is simpler and works with any model without retraining. Residual compression requires training centroids on your corpus.

## Query Augmentation with [MASK] Tokens

**Paper**: [ColBERT's [MASK]-based Query Augmentation: Effects of Quadrupling the Query Input Length](https://arxiv.org/abs/2408.13672)

**What it is**: Adding special `[MASK]` tokens to queries during encoding. These tokens act as soft query expansion, allowing the model to learn which query terms should be emphasized.

**Benefits**: Improved retrieval quality, especially for short queries

**Current status**: ✅ **DOCUMENTED** - Added comprehensive example (`examples/mask_token_weighting.rs`) and documentation showing how to use `maxsim_weighted()` with [MASK] token embeddings and their learned importance weights.

**Implementation complexity**: Medium - would require:
1. Support for masked token embeddings in queries ✅ (encoders provide these)
2. Weighted MaxSim ✅ (we have `maxsim_weighted()`)
3. Documentation on how to use [MASK] tokens with encoders ✅ (added example and docs)

**Note**: We already have `maxsim_weighted()` which can be used with [MASK] token embeddings. The example shows how to weight [MASK] tokens lower (typically 0.2-0.4) than original query tokens.

## Batch Alignment Functions

**What it is**: Batch versions of `maxsim_alignments()` and `highlight_matches()` for processing multiple documents at once.

**Current status**: ✅ **IMPLEMENTED** - Added `maxsim_alignments_batch()`, `maxsim_alignments_cosine_batch()`, and `highlight_matches_batch()`.

**Implementation complexity**: Low - straightforward extension of existing batch functions.

**Priority**: Medium - useful for reranking pipelines but not critical.

## Query Expansion Utilities

**What it is**: Helper functions for common query expansion techniques:
- Synonym expansion
- Term weighting (IDF-based)
- Query rewriting

**Current status**: ✅ **PARTIALLY IMPLEMENTED** - Added `idf_weights()` and `bm25_weights()` for computing term importance weights. These can be used with `maxsim_weighted()` to boost rare terms and improve retrieval quality.

**Implementation complexity**: Low-Medium - depends on how much we want to support.

**Note**: We have `maxsim_weighted()` which can be used with IDF weights. Synonym expansion and query rewriting are typically done before encoding, not in the scoring layer.

## Advanced Token Pooling Strategies

**What we have**: Greedy agglomerative and Ward's hierarchical clustering.

**What's missing**:
- Learnable token importance weights (Archish et al., 2025)
- Adaptive pooling based on query distribution
- Token pruning (removing uninformative tokens entirely)

**Priority**: Low - current pooling is sufficient for most use cases.

## Multimodal-Specific Features

**What we have**: Alignment and highlighting work for both text and multimodal (ColPali).

**What's missing**:
- Patch-level visualization utilities
- Snippet extraction helpers (extracting image regions from patch indices) ✅ **IMPLEMENTED** - Added `patches_to_regions()` and `extract_snippet_indices()`
- Cross-modal attention visualization

**Priority**: Low - users can implement these on top of our alignment functions. Basic snippet extraction helpers are now available.

## Summary

| Feature | Complexity | Priority | Notes |
|---------|-----------|----------|-------|
| Residual compression | High | Low | Token pooling is simpler alternative |
| [MASK] token support | Medium | ✅ **DONE** | Documented with example |
| Batch alignment | Low | ✅ **DONE** | Implemented with utilities |
| Alignment utilities | Low | ✅ **DONE** | top_k, filter, stats, query/doc filtering |
| Query expansion utils | Low-Medium | ✅ **DONE** | IDF and BM25 weighting added |
| Advanced pooling | Medium | Low | Current methods sufficient |
| Multimodal helpers | Low | ✅ **DONE** | Patch-to-region and snippet extraction |

## Recommendations

1. **Focus on testing and correctness** (current priority) ✅
2. **Document how to use weighted MaxSim with [MASK] tokens** (low effort, high value)
3. **Add batch alignment functions** (if users request it)
4. **Residual compression**: Only if users specifically need 6-10× compression beyond token pooling

The core functionality (MaxSim, alignment, highlighting, pooling, diversity) is complete and well-tested. Missing features are either:
- Encoder-level concerns (query augmentation)
- Nice-to-have optimizations (batch alignment)
- Alternative approaches to problems we already solve (residual compression vs token pooling)

