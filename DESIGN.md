# Design

## Mathematical Foundation

This crate addresses two mathematically distinct problems:

| Problem | Field | Function Type | This Module |
|---------|-------|---------------|-------------|
| **Scoring** | Learning to Rank | f(query, doc) → ℝ | `simd`, `colbert`, `matryoshka`, `crossencoder` |
| **Selection** | Submodular Optimization | f(set) → ℝ | `diversity` |

Both require looking at document **content** (embeddings), unlike rank fusion which only looks at scores/ranks.

## Scoring (Learning to Rank)

**Problem**: Given a query and candidate documents, compute relevance scores.

Three paradigms in increasing complexity and quality:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SCORING PARADIGMS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Dense             Late Interaction       Cross-encoder            │
│   ──────────────    ─────────────────      ──────────────           │
│   q · d             Σ_i max_j(q_i · d_j)   Model(q || d)            │
│                                                                     │
│   O(d)              O(q × d × tokens)      O(n) inference           │
│                                                                     │
│   Single vector     Token embeddings       Full text pair           │
│   per doc           per doc                (expensive)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dense Scoring (simd module)

The simplest: one embedding per document.

```rust
let score = dot(&query_embedding, &doc_embedding);
let score = cosine(&query_embedding, &doc_embedding);
```

**When to use**: Bi-encoder embeddings (Sentence-BERT, E5, etc.). Fast, SIMD-accelerated.

### Late Interaction (colbert module)

Token-level embeddings preserve fine-grained semantics. MaxSim scoring:

```
score = Σ_{q ∈ Q} max_{d ∈ D} sim(q, d)
```

For each query token, find the most similar document token and sum.

```rust
let score = maxsim(&query_tokens, &doc_tokens);
let ranked = rank(&query_tokens, &documents);
```

**When to use**: ColBERT-style models. 5-15% better recall on complex queries.

### Matryoshka Refinement (matryoshka module)

For [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) embeddings.

Coarse search with prefix dimensions → refine with tail dimensions:

```rust
// First 256 dims for coarse ranking
let coarse_scores = candidates.iter().map(|d| dot(&q[..256], &d[..256]));

// Full 768 dims to refine top candidates
let refined = mrl_refine(query, candidates, config);
```

**When to use**: MRL-trained models (Nomic, Cohere v3, OpenAI text-embedding-3).

### Cross-encoder (crossencoder module)

The most accurate but most expensive. Requires running a transformer on each (query, document) pair.

```rust
impl CrossEncoderModel for MyModel {
    fn score(&self, query: &str, document: &str) -> f32 {
        // Your inference code here
    }
}
```

**When to use**: Final reranking stage. ~5x better than dense on MS MARCO.

## Selection (Submodular Optimization)

**Problem**: Given scored candidates, select a diverse subset.

This is fundamentally different from scoring:
- Scoring: f(query, doc) → ℝ  (independent per doc)
- Selection: f(S ⊆ candidates) → ℝ  (depends on what's already selected)

### The Submodular Structure

Adding an item has **diminishing returns**:

```
f(S ∪ {x}) - f(S) ≤ f(T ∪ {x}) - f(T)  when T ⊆ S
```

If you've already selected similar items, a new item adds less value.

### MMR (diversity module)

[Maximal Marginal Relevance](https://dl.acm.org/doi/10.1145/290941.291025) balances relevance and diversity:

```
MMR = argmax_{d ∈ R\S} [ λ · rel(d) - (1-λ) · max_{s ∈ S} sim(d, s) ]
```

- λ=1.0: Pure relevance (no diversity)
- λ=0.5: Balanced
- λ=0.0: Pure diversity (avoid similar items)

```rust
let diverse = mmr_cosine(&relevance_scores, &embeddings, MmrConfig::default());
```

**When to use**: Search results, recommendations, summarization.

## Why One Crate, Not Two?

We considered separating scoring and selection into `rank-refine` and `rank-select`.

Arguments for keeping them together:

1. **Shared SIMD**: Both use `dot`, `cosine`, `maxsim` from simd module
2. **Pipeline coupling**: Users typically score → filter → diversify in sequence
3. **Code size**: MMR is ~150 lines, not enough for a separate crate

The mathematical distinction is documented, but practical concerns favor a single crate.

## Architecture

```
                      ┌──────────────────┐
                      │   Your Model     │
                      │  (embeddings)    │
                      └────────┬─────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  matryoshka   │  │   colbert     │  │ crossencoder  │
    │  (MRL tail)   │  │  (MaxSim)     │  │    (BYOM)     │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │      simd        │
                      │  (dot, cosine,   │
                      │   maxsim, norm)  │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │    diversity     │
                      │  (MMR, selection)│
                      └──────────────────┘
```

## SIMD Implementation

Automatic dispatch based on CPU features:

| Platform | Features | Method |
|----------|----------|--------|
| x86_64 | AVX2 + FMA | 256-bit vectors, fused multiply-add |
| x86_64 | SSE4.1 | 128-bit vectors |
| aarch64 | NEON | 128-bit vectors |
| other | — | Portable scalar fallback |

Threshold: SIMD only used when dimension ≥ 16 (overhead not worth it for small vectors).

## Token Pooling

For ColBERT-style models, token count dominates storage. Pooling reduces tokens:

| Factor | Reduction | Quality Loss |
|--------|-----------|--------------|
| 2x | 50% | ~0% |
| 4x | 75% | <5% |
| 8x | 87.5% | <10% |

From [Clavié et al., 2024](https://arxiv.org/abs/2409.14683).

```rust
let pooled = pool_tokens(&doc_tokens, 4); // 75% reduction
```

Pooling happens at **indexing time only** — queries stay full resolution.

## API Design

### Generic over ID Type

```rust
rank::<&str>(&query, &[("doc1", &emb1), ("doc2", &emb2)])
rank::<u64>(&query, &[(1, &emb1), (2, &emb2)])
```

### Slice-Based (Zero-Copy)

```rust
// No allocation — works on borrowed data
let score = maxsim(&q_tokens, &d_tokens);
```

### BYOM (Bring Your Own Model)

No inference code bundled. Use fastembed, ort, candle, rust-bert, etc.

```rust
// You provide embeddings
let embeddings: Vec<Vec<f32>> = my_model.encode(&texts)?;

// We score them
let ranked = rank(&query_emb, &candidates);
```

## Relationship to rank-fusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Retrieval        Fusion           Scoring          Selection              │
│   ─────────────    ───────────      ───────────      ───────────            │
│   BM25 → list1  ─┐                                                          │
│   Dense → list2 ─┼──▶ rank-fusion ──▶ rank-refine ──▶ rank-refine           │
│   Sparse → list3 ┘    (combine)      (rescore)       (diversity.mmr)        │
│                                                                             │
│   Math field:      Social Choice    Learning to Rank  Submodular            │
│   Uses content:    No               Yes               Yes (inter-item)      │
│   Typical count:   1000s → 100      100 → 50          50 → 10               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## References

### Learning to Rank
- [Liu, 2011](https://link.springer.com/book/10.1007/978-3-642-14267-3) — Learning to Rank for IR (textbook)
- [Nogueira & Cho, 2019](https://arxiv.org/abs/1901.04085) — BERT for passage reranking

### Late Interaction
- [Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832) — ColBERT
- [Santhanam et al., 2022](https://arxiv.org/abs/2205.09707) — ColBERTv2

### Matryoshka
- [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147) — MRL embeddings
- [Wang et al., 2024](https://arxiv.org/abs/2411.17299) — 2D Matryoshka

### Submodular Optimization
- [Nemhauser et al., 1978](https://link.springer.com/article/10.1007/BF01588971) — Greedy approximation
- [Carbonell & Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025) — MMR
- [Krause & Golovin, 2014](https://las.inf.ethz.ch/files/krause12survey.pdf) — Submodular survey
