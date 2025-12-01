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

**Current status**: Not implemented. This is typically done at the encoder level (during model inference), not in the scoring layer.

**Implementation complexity**: Medium - would require:
1. Support for masked token embeddings in queries
2. Weighted MaxSim (we have this!)
3. Documentation on how to use [MASK] tokens with encoders

**Note**: We already have `maxsim_weighted()` which could be used if encoders provide weights for [MASK] tokens.

## Batch Alignment Functions

**What it is**: Batch versions of `maxsim_alignments()` and `highlight_matches()` for processing multiple documents at once.

**Current status**: We have `maxsim_batch()` for scoring, but not for alignments.

**Implementation complexity**: Low - straightforward extension of existing batch functions.

**Priority**: Medium - useful for reranking pipelines but not critical.

## Query Expansion Utilities

**What it is**: Helper functions for common query expansion techniques:
- Synonym expansion
- Term weighting (IDF-based)
- Query rewriting

**Current status**: Not implemented. This is typically done before encoding, not in the scoring layer.

**Implementation complexity**: Low-Medium - depends on how much we want to support.

**Note**: We have `maxsim_weighted()` which can be used with IDF weights.

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
- Snippet extraction helpers (extracting image regions from patch indices)
- Cross-modal attention visualization

**Priority**: Low - users can implement these on top of our alignment functions.

## Summary

| Feature | Complexity | Priority | Notes |
|---------|-----------|----------|-------|
| Residual compression | High | Low | Token pooling is simpler alternative |
| [MASK] token support | Medium | Medium | Encoder-level, but weighted MaxSim helps |
| Batch alignment | Low | Medium | Nice-to-have optimization |
| Query expansion utils | Low-Medium | Low | Typically done before encoding |
| Advanced pooling | Medium | Low | Current methods sufficient |
| Multimodal helpers | Low | Low | Can be built on top |

## Recommendations

1. **Focus on testing and correctness** (current priority) ✅
2. **Document how to use weighted MaxSim with [MASK] tokens** (low effort, high value)
3. **Add batch alignment functions** (if users request it)
4. **Residual compression**: Only if users specifically need 6-10× compression beyond token pooling

The core functionality (MaxSim, alignment, highlighting, pooling, diversity) is complete and well-tested. Missing features are either:
- Encoder-level concerns (query augmentation)
- Nice-to-have optimizations (batch alignment)
- Alternative approaches to problems we already solve (residual compression vs token pooling)

