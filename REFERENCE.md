# Technical Reference

Deep technical details for understanding the implementation.
For getting started, see [README.md](README.md).

## Table of Contents

1. [MaxSim Algorithm](#maxsim-algorithm)
2. [Token Pooling](#token-pooling)
3. [Matryoshka Embeddings](#matryoshka-embeddings)
4. [Maximal Marginal Relevance (MMR)](#maximal-marginal-relevance-mmr)
5. [Determinantal Point Processes (DPP)](#determinantal-point-processes-dpp)
6. [SIMD Implementation](#simd-implementation)
7. [Mathematical Properties](#mathematical-properties)
8. [When to Use What](#when-to-use-what)
9. [Glossary](#glossary)
10. [References](#references)

---

## MaxSim Algorithm

### Intuition

Dense retrieval compresses an entire document into one vector. Late interaction keeps **one vector per token**.

Think of it like this: dense retrieval asks "is this document about the same topic?" Late interaction asks "does this document have good matches for each part of my query?"

```
Query tokens:     [q1]  [q2]  [q3]
                    \    |    /
                     \   |   /     Each query token searches
                      v  v  v      for its best match
Document tokens:  [d1] [d2] [d3] [d4]
                   ↑         ↑
                  0.9       0.8    (best matches)

MaxSim = 0.9 + 0.8 + ... (sum of best matches)
```

For each query token, find its best-matching document token, then sum.

### Formal Definition

Given query tokens $Q = [q_1, q_2, \ldots, q_m]$ and document tokens $D = [d_1, d_2, \ldots, d_n]$
where each $q_i, d_j \in \mathbb{R}^d$:

$$\text{MaxSim}(Q, D) = \sum_{i=1}^{m} \max_{j \in [1,n]} (q_i \cdot d_j)$$

### Pseudocode

```python
def maxsim(Q, D):
    if len(Q) == 0 or len(D) == 0:
        return 0.0
    total = 0.0
    for q in Q:
        total += max(dot(q, d) for d in D)
    return total
```

### Worked Example

Query: "best pizza" (2 tokens), Document: "great italian pizza restaurant" (4 tokens)

Using 3-dimensional embeddings (real embeddings are typically 128-768 dim):

```
Query tokens:
  q1 "best"  = [0.8, 0.3, 0.1]
  q2 "pizza" = [0.2, 0.9, 0.4]

Document tokens:
  d1 "great"      = [0.7, 0.2, 0.1]
  d2 "italian"    = [0.1, 0.5, 0.8]
  d3 "pizza"      = [0.2, 0.95, 0.3]
  d4 "restaurant" = [0.4, 0.3, 0.6]
```

**Step 1:** For q1 ("best"), compute dot products with all doc tokens:
- q1 . d1 = 0.8(0.7) + 0.3(0.2) + 0.1(0.1) = 0.56 + 0.06 + 0.01 = **0.63**
- q1 . d2 = 0.8(0.1) + 0.3(0.5) + 0.1(0.8) = 0.08 + 0.15 + 0.08 = 0.31
- q1 . d3 = 0.8(0.2) + 0.3(0.95) + 0.1(0.3) = 0.16 + 0.285 + 0.03 = 0.475
- q1 . d4 = 0.8(0.4) + 0.3(0.3) + 0.1(0.6) = 0.32 + 0.09 + 0.06 = 0.47
- max = **0.63** (matched "great")

**Step 2:** For q2 ("pizza"), compute dot products:
- q2 . d1 = 0.2(0.7) + 0.9(0.2) + 0.4(0.1) = 0.14 + 0.18 + 0.04 = 0.36
- q2 . d2 = 0.2(0.1) + 0.9(0.5) + 0.4(0.8) = 0.02 + 0.45 + 0.32 = 0.79
- q2 . d3 = 0.2(0.2) + 0.9(0.95) + 0.4(0.3) = 0.04 + 0.855 + 0.12 = **1.015**
- q2 . d4 = 0.2(0.4) + 0.9(0.3) + 0.4(0.6) = 0.08 + 0.27 + 0.24 = 0.59
- max = **1.015** (matched "pizza")

**Final:** MaxSim = 0.63 + 1.015 = **1.645**

### Complexity

- **Time**: $O(m \times n \times d)$ where m=query tokens, n=doc tokens, d=embedding dim
- **Space**: $O(1)$ additional (streaming computation)

### Properties

1. **Asymmetric**: $\text{MaxSim}(Q, D) \neq \text{MaxSim}(D, Q)$
2. **Monotonic**: Adding doc tokens cannot decrease score
3. **Permutation invariant**: Token order doesn't affect score

### Score Normalization

MaxSim scores are **unbounded**: $\text{score} \in [0, m]$ where $m$ is query token count.

**To normalize to ~\[0, 1\]:**

```rust
let query_maxlen = 32; // Standard ColBERT default
let normalized = maxsim(&q, &d) / query_maxlen as f32;
```

**For relative comparison** (comparing candidates for a single query):

```rust
// Softmax normalization - scores sum to 1
let scores: Vec<f32> = docs.iter().map(|d| maxsim(&q, d)).collect();
let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
let sum: f32 = exp_scores.iter().sum();
let normalized: Vec<f32> = exp_scores.iter().map(|s| s / sum).collect();
```

**Why 32?** ColBERT models are trained with `query_maxlen=32`. Even if your query has fewer
tokens, normalizing by 32 keeps scores comparable across different query lengths.

### Edge Cases

| Case | Behavior | Note |
|------|----------|------|
| Empty query | Returns 0.0 | — |
| Empty document | Returns 0.0 | — |
| Single query token | Reduces to max(dot(q, d)) | No aggregation benefit |
| Identical tokens | Sum of self-similarities | May inflate score |
| All-padding query | Score reflects padding | Use `MaskedTokens` to exclude |

### Out of Scope

This crate **does not handle**:
- **Tokenization** — Use your model's tokenizer
- **Role markers** (`[Q]`/`[D]`) — Applied during encoding, not scoring
- **Dimensionality reduction** — Pre-process before calling this crate

---

## Token Pooling

### The Storage Problem

ColBERT stores ~128 vectors per document. For 10M documents:

$$\text{Storage} = 10^7 \times 128 \times 128 \times 4 = 655\text{ GB}$$

That's a lot of disk space! Can we compress without losing much quality?

### The Key Insight

Many token embeddings within a document are *similar*. Words like "the", "a", "is" often cluster together. Semantically related words ("pizza", "Italian", "restaurant") also cluster. Instead of storing all tokens, we can store cluster centroids.

### How It Works

Imagine a document with 8 token embeddings. We want to reduce to 5.

```
Step 1: Start with all tokens as singleton clusters
  [t1] [t2] [t3] [t4] [t5] [t6] [t7] [t8]

Step 2: Find most similar pair (say t1 and t2), merge them
  [t1,t2] [t3] [t4] [t5] [t6] [t7] [t8]  (7 clusters)

Step 3: Find next most similar pair (say t4 and t5), merge
  [t1,t2] [t3] [t4,t5] [t6] [t7] [t8]  (6 clusters)

Step 4: Merge again (say t1,t2 with t3)
  [t1,t2,t3] [t4,t5] [t6] [t7] [t8]  (5 clusters) ← target reached!

Step 5: Replace each cluster with its mean
  [mean(t1,t2,t3)] [mean(t4,t5)] [t6] [t7] [t8]
```

The result: 5 vectors instead of 8, a 37.5% reduction.

### Why Pooling Works

You might expect that merging tokens would lose information. Why doesn't it hurt quality more?

The answer lies in token redundancy. In a typical document:

1. **Function words** ("the", "a", "is", "and") cluster together—they share similar contextual patterns
2. **Stopwords** and punctuation are often uninformative for retrieval
3. **Related content words** ("machine", "learning", "algorithm") have similar embeddings

When we merge these similar tokens, we lose little discriminative information. The merged centroid still matches queries looking for that semantic region.

This is related to why BM25 with IDF weighting works: common words contribute less. Pooling is essentially a soft version of this—we keep all tokens but let redundant ones share storage.

### Pool Factor

| Factor | Tokens Kept | MRR@10 Loss (MS MARCO) | When to Use |
|--------|-------------|------------------------|-------------|
| 2 | 50% | 0.1–0.3% | Default choice |
| 3 | 33% | 0.5–1.0% | Good tradeoff |
| 4 | 25% | 1.5–3.0% | Use `hierarchical` feature |
| 8+ | 12% | 5–10% | Storage-critical only |

Numbers from Clavie et al. (2024) on MS MARCO dev set.

**When pooling hurts more:** Long documents with many distinct concepts. The merged centroid may not match specific query terms as precisely.

**When pooling is safe:** Short passages, documents with repetitive phrasing, or queries that match broad topics rather than specific terms.

### Greedy vs Ward's

**Greedy** (default): Merge most similar pair. $O(n^3)$.

**Ward's** (`hierarchical` feature): Minimize within-cluster variance.
Better for factor 4+. Uses [kodama](https://docs.rs/kodama).

**Note**: Tokens are generally evenly distributed across clusters, but this
is not guaranteed — some clusters may contain more tokens than others.

Reference: [Clavié et al., 2024](https://arxiv.org/abs/2409.14683), [Answer.AI blog](https://www.answer.ai/posts/colbert-pooling.html)

### Future: Token Importance Weighting

**Weighted Chamfer** (Archish et al., Nov 2025) extends MaxSim with learnable
per-token weights:

$$\text{WeightedMaxSim}(Q, D) = \sum_{i=1}^{m} w_i \cdot \max_{j \in [1,n]} (q_i \cdot d_j)$$

where weights $w_i$ are learned and constrained to sum to 1. Shows +2% improvement
on BEIR in few-shot settings. Not yet implemented here.

Reference: [arxiv.org/abs/2511.16106](https://arxiv.org/abs/2511.16106)

---

## Matryoshka Embeddings

### The Idea

Like Russian nesting dolls, a Matryoshka embedding contains smaller valid embeddings inside it. The first 64 dimensions form a complete embedding. So do the first 128, 256, and so on.

```
Full 768-dim embedding:
[████████████████████████████████████████████████]
 ↑─────────────────────────────────────────────↑
 dim 0                                    dim 767

Truncated views (all valid embeddings):
[████████]                                          dims 0-63   (coarse)
[████████████████]                                  dims 0-127  (medium)
[████████████████████████████████]                  dims 0-255  (fine)
[████████████████████████████████████████████████]  dims 0-767  (full)
```

### Why Does This Work?

During training, the model is penalized at multiple checkpoints:

$$\mathcal{L} = \sum_{k \in \{64, 128, 256, \ldots\}} \lambda_k \cdot \text{loss}(v_{0:k})$$

This forces the model to pack information hierarchically:
- **Early dimensions (0–64):** Broad topic (is this about sports? science? cooking?)
- **Middle dimensions (64–256):** Subtopic and key entities
- **Late dimensions (256+):** Fine-grained distinctions, rare concepts

Analogy: Like a progressive JPEG, where early bytes give you a blurry image and later bytes add detail.

### Two-Stage Retrieval

This enables a powerful optimization:

1. **Stage 1 (Fast, Coarse)**: Search using only the first 128 dims. Your vector DB does this efficiently.
2. **Stage 2 (Slow, Precise)**: Re-score the top 1000 candidates using the remaining dims.

```
Query embedding: [████|████████████████████████████████████]
                  head (128d)    tail (640d)
                  ↓
Stage 1: ANN search with head only → 1000 candidates
                  ↓
Stage 2: Re-score using tail similarity → 10 final results
```

### Blending Formula

Combine original (head-based) scores with tail similarity:

$$\text{final} = \alpha \cdot \text{original} + (1-\alpha) \cdot \text{tail similarity}$$

| Alpha | Behavior |
|-------|----------|
| 1.0 | Keep original ranking (no refinement) |
| 0.5 | Equal blend (default) |
| 0.0 | Rank purely by tail similarity |

### Worked Example

Two candidates with identical head similarity but different tails:

```
Query:  head=[0.5, 0.5]  tail=[0.9, 0.1]
Doc A:  head=[0.5, 0.5]  tail=[0.8, 0.2]  (tail similar to query)
Doc B:  head=[0.5, 0.5]  tail=[0.1, 0.9]  (tail opposite to query)

Original scores (from Stage 1, head only):
  Both have score 0.8 (tied)

Tail similarities:
  A: cosine([0.9,0.1], [0.8,0.2]) = 0.98
  B: cosine([0.9,0.1], [0.1,0.9]) = 0.18

With alpha=0.5:
  A: 0.5 * 0.8 + 0.5 * 0.98 = 0.89
  B: 0.5 * 0.8 + 0.5 * 0.18 = 0.49

Result: Doc A wins, the tail broke the tie.
```

Reference: [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)

### Future: 2D Matryoshka

2D Matryoshka (Li et al., 2024) extends MRL by training with loss at multiple
**layers** AND dimensions:

$$\mathcal{L}_{2D} = \sum_{l \in \text{layers}} \sum_{k \in \text{dims}} \lambda_{l,k} \cdot \text{loss}(v^{(l)}_{0:k})$$

This enables truncating intermediate transformer layers at inference time,
reducing latency ~50% with ~85% quality retention. Not yet implemented here.

Reference: [Li et al., 2024](https://arxiv.org/abs/2402.14776)

---

## Maximal Marginal Relevance (MMR)

### The Problem

You search "machine learning tutorials" and get:
1. "Introduction to ML" (score 0.95)
2. "ML Basics for Beginners" (score 0.93)
3. "Getting Started with ML" (score 0.91)
4. "Deep Learning Guide" (score 0.85)
5. "Reinforcement Learning Intro" (score 0.82)

The top 3 are essentially the same content. Users would prefer variety.

### The Solution

MMR balances relevance and diversity via iterative selection:

$$\text{MMR}(d) = \lambda \cdot \text{relevance}(d) - (1-\lambda) \cdot \max_{s \in \text{Selected}} \text{similarity}(d, s)$$

**In words:** Score each candidate by its relevance, then subtract a penalty for being too similar to already-selected items.

### Visual Intuition

Imagine candidates as points in embedding space:

```
        relevance →
    high ┌─────────────────────┐
         │     A               │   A, B, C are clustered (similar)
         │    B  C             │   D is isolated (different topic)
         │                     │
         │           D         │
     low └─────────────────────┘

Pure relevance (λ=1): Pick A, B, C → all from same cluster
MMR (λ=0.5):          Pick A, then D (diverse), then B
```

After selecting A, the penalty term `-max sim(d, A)` hurts B and C (similar to A) but barely affects D (different from A).

### Worked Example

After selecting "Intro to ML" (doc A), compute MMR for remaining candidates:

| Doc | Relevance | Sim to A | MMR (λ=0.5) |
|-----|-----------|----------|-------------|
| B "ML Basics" | 0.93 | 0.95 | 0.5(0.93) - 0.5(0.95) = **-0.01** |
| C "Getting Started" | 0.91 | 0.92 | 0.5(0.91) - 0.5(0.92) = **-0.005** |
| D "Deep Learning" | 0.85 | 0.40 | 0.5(0.85) - 0.5(0.40) = **0.225** |
| E "RL Intro" | 0.82 | 0.35 | 0.5(0.82) - 0.5(0.35) = **0.235** |

**E wins!** Despite lowest relevance, it's most diverse from A.

### Lambda Parameter

| λ Value | Behavior | Use Case |
|---------|----------|----------|
| 0.3 | Strong diversity penalty | Exploration, "show me different options" |
| 0.5 | Equal weight | General purpose, RAG context |
| 0.7 | Mild diversity penalty | "I want relevant results, but some variety" |
| 1.0 | No diversity (pure top-k) | Known-item search, single correct answer |

There's no universally optimal λ—it depends on how clustered your candidates are (Gao & Zhang, 2024).

### Complexity

O(k × n) for selecting k items from n candidates.

### Historical Note

MMR was introduced in 1998 for multi-document summarization—selecting sentences that cover different aspects of a topic. The algorithm predates neural retrieval by decades, yet remains remarkably effective.

Recent work (VRSD, Gao & Zhang 2024) proves that optimal λ depends on the geometric distribution of candidates in embedding space. There's no universal best λ—it's inherently data-dependent. The 0.5 default is a reasonable compromise.

For RAG specifically, the 0.4-0.6 range tends to work well: enough diversity to give language models varied context, enough relevance to stay on-topic.

Reference: [Carbonell & Goldstein, 1998](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)

---

## Determinantal Point Processes (DPP)

### Origins

DPPs originated in physics (Fermion statistics in quantum mechanics) and random matrix theory, where they model repulsive point processes. Kulesza & Taskar (2012) introduced them to machine learning as a principled way to model diversity.

### The Core Idea

A DPP defines a probability distribution over subsets where items "repel" each other. The probability of selecting a subset S is proportional to the determinant of a kernel submatrix:

$$P(S) \propto \det(L_S)$$

**Why determinants?** The determinant measures the volume spanned by vectors:

```
Parallel vectors:         Orthogonal vectors:
    →                         →
   /                         |
  /  (flat, zero volume)     └─→  (square, max volume)
```

Two similar items span a thin sliver. Two different items span a large area. DPP prefers sets that span large volumes.

### DPP vs MMR

| Aspect | MMR | DPP |
|--------|-----|-----|
| Diversity model | Max similarity to any selected item | Joint volume in embedding space |
| Computation | O(k × n) | O(k × n × d) |
| Pairwise interactions | Ignores (only max matters) | Considers all pairs |
| Theoretical grounding | Heuristic | Probabilistic model |

**When DPP wins:** Small k, when pairwise interactions matter. If items A and B are both similar to C but different from each other, MMR might reject both (each too similar to C). DPP recognizes that {A, B, C} spans good volume because A and B are different.

**When MMR wins:** Large k, when speed matters. MMR's O(k × n) is cheaper than DPP's O(k × n × d). For k > 20, the difference is noticeable.

### The Fast Greedy Approximation

Exact MAP inference for DPP is NP-hard. We use the fast greedy algorithm (Chen et al., 2018):

```
1. Initialize: c[i] = ||v_i||^2 for all items (squared norm = "self-volume")
2. For each selection:
   a. Pick item maximizing quality[i] × sqrt(c[i])
   b. Update c[j] -= (v_j · v_selected)^2 / c[selected] for remaining items
   c. (This subtracts the "volume already covered" by the selected item)
3. Repeat until k items selected
```

The update step is essentially Gram-Schmidt orthogonalization: after selecting an item, we project out its contribution from all remaining items. Items aligned with already-selected items see their effective volume shrink.

Reference: [Kulesza & Taskar, 2012](https://arxiv.org/abs/1207.6083), [Chen et al., 2018](https://papers.nips.cc/paper/7805-fast-greedy-map-inference-for-determinantal-point-process-to-improve-recommendation-diversity.pdf)

---

## SIMD Implementation

### Why SIMD?

A dot product of two 128-dimensional vectors requires 128 multiplications and 127 additions. Without SIMD, that's 255 scalar operations.

With AVX2, we process 8 floats per instruction:
- 128 / 8 = 16 multiply instructions
- 16 horizontal adds + final reduction
- **~10x fewer instructions**

### Dispatch Strategy

At runtime, we detect CPU features and pick the fastest available path:

```
Is dimension >= 16?
  ├─ No  → Portable scalar loop (SIMD overhead not worth it)
  └─ Yes → Check architecture
              ├─ x86_64 + AVX2+FMA → 256-bit vectors (8 floats/op)
              ├─ aarch64 (NEON)    → 128-bit vectors (4 floats/op)
              └─ other             → Portable scalar loop
```

### Why 16-Dimension Threshold?

SIMD has overhead: register setup, potential cache effects, horizontal reductions. For very short vectors, this overhead exceeds the benefit. Testing shows dimension 16 is the crossover point (matches qdrant's findings).

### Safety

All `unsafe` code has `// SAFETY:` comments documenting:
1. Feature detection performed
2. Alignment requirements met  
3. Bounds checking

### Performance Note: Subnormals

Denormalized (subnormal) floats can cause **140+ cycle penalties** per SIMD operation. Values below ~$10^{-38}$ for f32 trigger microcode paths.

In practice, this is rare for embeddings (typically normalized to unit length). If you encounter slow performance with small values, consider flush-to-zero mode at the application level.

---

## Mathematical Properties

### Dot Product

For $a, b \in \mathbb{R}^d$:

$$a \cdot b = \sum_{i=1}^{d} a_i b_i$$

Properties:
- **Commutative**: $a \cdot b = b \cdot a$
- **Bilinear**: $(\alpha a) \cdot b = \alpha(a \cdot b)$
- **Self-product**: $a \cdot a = \|a\|^2$
- **Cauchy-Schwarz**: $|a \cdot b| \leq \|a\| \cdot \|b\|$

### Cosine Similarity

$$\cos(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

Properties:
- **Bounded**: $\cos(a, b) \in [-1, 1]$
- **Scale-invariant**: $\cos(\alpha a, \beta b) = \cos(a, b)$ for $\alpha, \beta > 0$

### NaN Handling

Uses `f32::total_cmp` for deterministic ordering:

$$-\infty < \ldots < -0 < +0 < \ldots < +\infty < \text{NaN}$$

NaN values sort to the **end** (lowest priority in descending sort).

---

## When to Use What

### Scoring Methods

| Method | Latency | Quality | Storage | Best For |
|--------|---------|---------|---------|----------|
| Dense (cosine) | ~50ns | Baseline | 1 vec/doc | First-stage retrieval, simple use cases |
| Late interaction (MaxSim) | ~50μs | +15-20% | 128 vecs/doc | Second-stage reranking, precision-critical |
| Cross-encoder | ~10ms | +30-40% | N/A | Final reranking over <100 candidates |

**Typical pipeline:**
1. Dense retrieval → 1000 candidates (fast ANN)
2. MaxSim reranking → 100 candidates (if storage allows)
3. Cross-encoder → 10 results (if latency allows)

### Diversity Methods

| Method | Best For | Avoid When |
|--------|----------|------------|
| Top-k | Precision search, known-item finding | Users want variety |
| MMR (λ=0.5) | General purpose, RAG | Candidates very clustered |
| MMR (λ=0.3) | Exploration, discovery | Single correct answer exists |
| DPP | Small k, pairwise diversity matters | k > 20, latency-critical |

### Token Compression

| Pool Factor | Quality Loss | Use Case |
|-------------|--------------|----------|
| None | 0% | Latency-critical, storage-abundant |
| 2x | ~0% | Default for most deployments |
| 3x | ~1% | Good storage/quality tradeoff |
| 4x | ~3% | Storage-constrained (use `hierarchical`) |
| 8x+ | ~7% | Extreme storage constraints only |

**Key insight:** Pool aggressively at index time. You can always re-score with unpooled embeddings at query time if precision matters.

### Matryoshka Strategy

| Scenario | Head Dims | Strategy |
|----------|-----------|----------|
| Vector DB with fixed dims | Match DB | Use what you have |
| Latency-critical | 64-128 | Sacrifice precision for speed |
| Two-stage available | 128 + full | Coarse search, then refine |
| Storage-constrained | 256 | Good balance |

---

## Glossary

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor search |
| **Bi-encoder** | Encoder that processes query and document independently |
| **ColBERT** | Contextualized Late Interaction over BERT |
| **Cross-encoder** | Encoder that processes query+document jointly |
| **Dense retrieval** | One vector per document |
| **DPP** | Determinantal Point Process — diversity model based on matrix determinants |
| **Late interaction** | One vector per token, interaction deferred to query time |
| **MaxSim** | Sum of per-query-token max similarities |
| **MMR** | Maximal Marginal Relevance — greedy diversity selection |
| **MRL** | Matryoshka Representation Learning |
| **Pool factor** | Compression ratio (factor 2 = 50% reduction) |
| **RAG** | Retrieval-Augmented Generation |
| **Submodular** | Set function with diminishing returns property |
| **Ward's method** | Clustering minimizing within-cluster variance |

---

## References

### Scoring

- **ColBERT** (Khattab & Zaharia, 2020): [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) — Original late interaction paper
- **ColBERTv2** (Santhanam et al., 2021): [arxiv.org/abs/2112.01488](https://arxiv.org/abs/2112.01488) — Residual compression, denoised supervision
- **PLAID** (Santhanam et al., 2022): [arxiv.org/abs/2205.09707](https://arxiv.org/abs/2205.09707) — Centroid-based indexing
- **BERT Reranking** (Nogueira & Cho, 2019): [arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085) — Cross-encoder baseline

### Embeddings

- **Matryoshka** (Kusupati et al., 2022): [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) — Nested dimension training
- **2D Matryoshka** (Li et al., 2024): [arxiv.org/abs/2402.14776](https://arxiv.org/abs/2402.14776) — Layer + dimension truncation

### Token Compression

- **Token Pooling** (Clavie et al., 2024): [arxiv.org/abs/2409.14683](https://arxiv.org/abs/2409.14683) — Clustering-based compression
- **Token Pruning** (Lassance et al., 2021): [arxiv.org/abs/2112.06540](https://arxiv.org/abs/2112.06540) — Remove uninformative tokens
- **Token Importance** (Archish et al., 2025): [arxiv.org/abs/2511.16106](https://arxiv.org/abs/2511.16106) — Learnable weights

### Diversity

- **MMR** (Carbonell & Goldstein, 1998): [ACM DL](https://dl.acm.org/doi/10.1145/290941.291025) — Maximal Marginal Relevance
- **DPPs for ML** (Kulesza & Taskar, 2012): [arxiv.org/abs/1207.6083](https://arxiv.org/abs/1207.6083) — Determinantal Point Processes survey
- **Fast Greedy DPP** (Chen et al., 2018): [NeurIPS](https://papers.nips.cc/paper/7805-fast-greedy-map-inference-for-determinantal-point-process-to-improve-recommendation-diversity.pdf)
- **VRSD** (Gao & Zhang, 2024): [arxiv.org/abs/2407.04573](https://arxiv.org/abs/2407.04573) — Lambda-free diversity
- **SMART-RAG** (Li et al., 2024): [arxiv.org/abs/2409.13992](https://arxiv.org/abs/2409.13992) — DPP for RAG

### Theory

- **Submodular Maximization** (Nemhauser et al., 1978): [Springer](https://link.springer.com/article/10.1007/BF01588971) — Greedy approximation guarantees

### Libraries

- **kodama**: [docs.rs/kodama](https://docs.rs/kodama) — Hierarchical clustering
