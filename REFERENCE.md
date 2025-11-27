# Technical Reference

This document provides deep technical details for those who want to understand
exactly how this library works. For getting started, see [README.md](README.md).

## Table of Contents

1. [MaxSim Algorithm](#maxsim-algorithm)
2. [Token Pooling](#token-pooling)
3. [Matryoshka Embeddings](#matryoshka-embeddings)
4. [SIMD Implementation](#simd-implementation)
5. [Mathematical Properties](#mathematical-properties)
6. [Glossary](#glossary)
7. [References](#references)

---

## MaxSim Algorithm

### Intuition

Imagine you have a query "What is the capital of France?" and a document about
Paris. Dense retrieval compresses both into single vectors and computes one
similarity score. But what if "capital" appears early in the document and
"France" appears late? A single vector might not capture both well.

Late interaction keeps one vector per token:

```
Query:  ["what", "is", "the", "capital", "of", "France", "?"]
         ↓       ↓     ↓      ↓         ↓      ↓        ↓
        [q1]   [q2]  [q3]   [q4]      [q5]   [q6]     [q7]

Doc:    ["Paris", "is", "the", "capital", "city", "of", "France", "."]
         ↓        ↓     ↓      ↓         ↓       ↓      ↓        ↓
        [d1]     [d2]  [d3]   [d4]      [d5]    [d6]   [d7]     [d8]
```

For each query token, we find its best-matching document token:
- q4 ("capital") matches best with d4 ("capital") → high score
- q6 ("France") matches best with d7 ("France") → high score

### Formal Definition

Given:
- Query tokens: Q = [q₁, q₂, ..., qₘ] where each qᵢ ∈ ℝᵈ
- Document tokens: D = [d₁, d₂, ..., dₙ] where each dⱼ ∈ ℝᵈ

The MaxSim score is:

```
MaxSim(Q, D) = Σᵢ₌₁ᵐ max_{j∈[1,n]} (qᵢ · dⱼ)
```

In words: for each query token, find the document token with highest dot product,
then sum all these maxima.

### Pseudocode

```python
def maxsim(query_tokens, doc_tokens):
    """
    query_tokens: list of m vectors, each d-dimensional
    doc_tokens: list of n vectors, each d-dimensional
    returns: scalar score
    """
    if len(query_tokens) == 0 or len(doc_tokens) == 0:
        return 0.0
    
    total = 0.0
    for q in query_tokens:
        best_match = -infinity
        for d in doc_tokens:
            similarity = dot(q, d)
            if similarity > best_match:
                best_match = similarity
        total += best_match
    
    return total
```

### Complexity

- **Time**: O(m × n × d) where m=query tokens, n=doc tokens, d=dimension
- **Space**: O(1) additional (streaming computation)

### Key Properties

1. **Asymmetric**: MaxSim(Q, D) ≠ MaxSim(D, Q) in general
2. **Non-negative** (for non-negative dot products): Adding more doc tokens can
   only increase or maintain the score
3. **Permutation invariant**: Token order doesn't affect the score

---

## Token Pooling

### The Storage Problem

ColBERT stores ~128 vectors per document (one per token). For a 10M document
corpus with 128-dimensional vectors:

```
Storage = 10M docs × 128 tokens × 128 dims × 4 bytes = 655 GB
```

Token pooling reduces this by clustering similar tokens.

### How It Works

1. **Compute pairwise similarities** between all tokens in a document
2. **Cluster similar tokens** using hierarchical clustering
3. **Replace each cluster** with its centroid (mean of member vectors)

```
Original tokens:    [t1] [t2] [t3] [t4] [t5] [t6] [t7] [t8]
                     │    │    │    │    │    │    │    │
Similarity matrix:  ┌─────────────────────────────────────┐
                    │ High similarity between t1,t2,t3    │
                    │ High similarity between t4,t5       │
                    │ t6,t7,t8 are distinct               │
                    └─────────────────────────────────────┘
                     │              │         │    │    │
Clusters:           [  cluster1  ] [cluster2] │    │    │
                     │              │         │    │    │
Pooled:             [mean123]     [mean45]  [t6] [t7] [t8]

Result: 8 tokens → 5 tokens (37.5% reduction)
```

### Pool Factor

The **pool factor** controls compression aggressiveness:

| Factor | Reduction | Quality Impact | Recommendation |
|--------|-----------|----------------|----------------|
| 2 | 50% | ~0% | Safe default |
| 3 | 66% | <1% | Good tradeoff |
| 4 | 75% | 2-3% | Use hierarchical |
| 8+ | 87%+ | 5%+ | Only if desperate |

### Greedy vs Ward's Clustering

**Greedy Agglomerative** (default):
```python
while num_clusters > target:
    find most similar pair of clusters
    merge them
```
- Simple, O(n³) worst case
- Good for factors 2-3

**Ward's Method** (with `hierarchical` feature):
```python
while num_clusters > target:
    find pair whose merge minimizes within-cluster variance
    merge them
```
- Produces more homogeneous clusters
- Better for aggressive pooling (factor 4+)
- Uses [kodama](https://docs.rs/kodama) library

### Why Ward's Is Better for Aggressive Pooling

Ward's method minimizes the increase in total within-cluster variance at each
merge. This produces clusters where members are genuinely similar, not just
"the least bad pairing available."

For moderate pooling (factor 2-3), the difference is small. For aggressive
pooling (factor 4+), Ward's maintains significantly better retrieval quality.

**Research**: [Clavié et al., 2024](https://arxiv.org/abs/2409.14683) shows
Ward's matches scipy's implementation and achieves research-paper results.

---

## Matryoshka Embeddings

### The Key Insight

Standard embedding models are trained to optimize the full vector. Matryoshka
models are trained to optimize **multiple nested prefixes simultaneously**:

```
Training loss = λ₁·loss(dims[0:64]) + λ₂·loss(dims[0:128]) + λ₃·loss(dims[0:256]) + ...
```

This forces the model to pack the most important information into the first
dimensions.

### Why This Matters for Retrieval

You can trade off quality vs speed/storage by using different prefix lengths:

| Dimensions | Quality | Speed | Use Case |
|------------|---------|-------|----------|
| 64 | 85% | Fastest | First-pass filtering |
| 128 | 92% | Fast | Balanced retrieval |
| 256 | 97% | Medium | High-quality search |
| 768 (full) | 100% | Slowest | Final refinement |

### Two-Stage Refinement

This library implements two-stage refinement:

```
Full embedding: [████████████████████████████████]
                     head (first k)    tail (remaining)

Stage 1: Search using head only (fast, lower quality)
Stage 2: Re-score top candidates using tail (slow, high quality)
```

The `alpha` parameter controls blending:

```
final_score = α × original_score + (1-α) × tail_similarity
```

### Why Tail Dimensions Help

The tail contains "fine-grained" information that can disambiguate between
candidates that looked similar using only the head:

```
Query:   "jaguar car"
Doc A:   "jaguar animal in rainforest"  (head similarity: 0.8)
Doc B:   "jaguar automobile review"     (head similarity: 0.8)

After tail refinement:
Doc A tail similarity: 0.3  →  final: 0.55
Doc B tail similarity: 0.9  →  final: 0.85  ← winner
```

**Paper**: [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)

---

## SIMD Implementation

### Why SIMD?

Single Instruction Multiple Data (SIMD) processes multiple values in parallel:

```
Scalar dot product (4 operations):
  a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3], then sum

SIMD dot product (1 operation + reduction):
  [a[0],a[1],a[2],a[3]] × [b[0],b[1],b[2],b[3]] = [p0,p1,p2,p3], then sum
```

This library uses:
- **AVX2+FMA** on x86_64 (8 floats per instruction)
- **NEON** on aarch64/ARM (4 floats per instruction)
- **Portable fallback** for other platforms

### Minimum Dimension Threshold

SIMD has setup overhead. For very small vectors, the portable implementation
is faster. We use a threshold of 16 dimensions:

```rust
const MIN_DIM_SIMD: usize = 16;

if a.len() >= MIN_DIM_SIMD {
    // Use SIMD
} else {
    // Use portable
}
```

### Safety

All `unsafe` SIMD code has explicit safety comments:

```rust
// SAFETY: We've verified AVX2 and FMA are available via runtime detection.
// The function handles mismatched lengths by using min(a.len(), b.len()).
unsafe { dot_avx2(a, b) }
```

---

## Mathematical Properties

### Dot Product Properties

For vectors a, b ∈ ℝᵈ:

1. **Commutative**: a·b = b·a
2. **Bilinear**: (αa)·b = α(a·b)
3. **Self-product**: a·a = ||a||²
4. **Cauchy-Schwarz**: |a·b| ≤ ||a|| × ||b||

### Cosine Similarity Properties

```
cos(a, b) = (a·b) / (||a|| × ||b||)
```

1. **Bounded**: cos(a, b) ∈ [-1, 1]
2. **Scale-invariant**: cos(αa, βb) = cos(a, b) for α,β > 0
3. **Self-similarity**: cos(a, a) = 1

### MaxSim Properties

1. **Asymmetric**: MaxSim(Q, D) ≠ MaxSim(D, Q)
2. **Monotonic in docs**: Adding doc tokens can only increase score
3. **Linear in query tokens**: Scales with number of query tokens

### NaN Handling

This library uses `f32::total_cmp` for sorting, which provides:
- Deterministic ordering: NaN < -∞ < ... < 0 < ... < +∞
- Consistent behavior across platforms
- No panics on NaN input

---

## Glossary

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor — fast but imprecise search |
| **ColBERT** | Contextualized Late Interaction over BERT |
| **Cross-encoder** | Model that processes query and document together |
| **Dense retrieval** | One vector per document |
| **Dot product** | Sum of element-wise products: Σᵢ aᵢbᵢ |
| **Late interaction** | One vector per token, scored at query time |
| **MaxSim** | Sum over query tokens of max similarity to any doc token |
| **MRL** | Matryoshka Representation Learning |
| **Pool factor** | Compression ratio for token pooling |
| **SIMD** | Single Instruction Multiple Data — parallel processing |
| **Token** | A word or subword unit in text |
| **Ward's method** | Clustering that minimizes within-cluster variance |

---

## References

### Core Papers

- **ColBERT**: Khattab & Zaharia, 2020. [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
- **PLAID**: Santhanam et al., 2022. [arxiv.org/abs/2205.09707](https://arxiv.org/abs/2205.09707)
- **Matryoshka**: Kusupati et al., 2022. [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147)
- **Token Pooling**: Clavié et al., 2024. [arxiv.org/abs/2409.14683](https://arxiv.org/abs/2409.14683)

### Implementation References

- **kodama** (Rust hierarchical clustering): [docs.rs/kodama](https://docs.rs/kodama)
- **answer.ai pooling blog**: [answer.ai/posts/colbert-pooling.html](https://www.answer.ai/posts/colbert-pooling.html)

### Production Systems

- **ColBERT-serve** (ECIR 2025): [arxiv.org/abs/2501.02065](https://arxiv.org/abs/2501.02065)
- **Jina-ColBERT-v2**: [arxiv.org/abs/2408.16672](https://arxiv.org/abs/2408.16672)

