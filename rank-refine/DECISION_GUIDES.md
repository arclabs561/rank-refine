# Decision Guides

Quick reference guides for choosing algorithms, parameters, and configurations.

## Dense vs Late Interaction

```
┌─────────────────────────────────────────────────────────┐
│ Do you need token-level alignment?                     │
│ (e.g., "capital of France" vs "France's economic       │
│  capital")                                             │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use Late         │         │ Use Dense                 │
│ Interaction      │         │                           │
│ (MaxSim)         │         │ • Single vector per doc   │
│                  │         │ • Fast ANN search         │
│ • Token-level    │         │ • Broad matching          │
│   matching       │         │ • Lower storage           │
│ • Fine-grained   │         │                           │
│   semantics      │         │                           │
│ • Higher quality │         │                           │
│   reranking      │         │                           │
│ • More storage   │         │                           │
│   (one vector    │         │                           │
│   per token)     │         │                           │
└──────────────────┘         └──────────────────────────┘
```

**When to Use Each:**
- **Dense**: First-stage retrieval, latency-critical, storage-constrained
- **Late Interaction**: Reranking, precision-critical, token-level semantics matter

## Token Pooling Strategy

```
┌─────────────────────────────────────────────────────────┐
│ How much storage can you afford?                        │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ ABUNDANT                     │ CONSTRAINED
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ No pooling       │         │ Use pooling               │
│                  │         │                           │
│ • Best quality   │         │ • Factor 2-4x typical     │
│ • Full token     │         │ • ~0-3% quality loss      │
│   information    │         │ • Hierarchical best      │
│ • Highest        │         │   quality                 │
│   storage        │         │                           │
└──────────────────┘         └──────────────────────────┘
                                        │
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    │ QUALITY CRITICAL?                    │
                    │                                       │
                    ▼                                       ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │ Hierarchical     │                  │ Greedy           │
         │ (Ward's method)   │                  │                  │
         │                   │                  │ • Fastest        │
         │ • Best quality    │                  │ • Good for       │
         │ • Minimizes       │                  │   factor 2-3x   │
         │   within-cluster  │                  │ • ~1-3% loss     │
         │   variance        │                  │                  │
         │ • Slower          │                  │                  │
         └──────────────────┘                  └──────────────────┘
```

**Why Hierarchical (Ward's Method) is Better for Quality:**

1. **Minimizes Within-Cluster Variance**: Ward's method merges clusters to minimize the increase in within-cluster sum of squares. This means merged tokens are truly similar, preserving more semantic information.

2. **Greedy is Local**: Greedy pooling merges the most similar pair at each step, but this can lead to suboptimal global clustering. A token might be merged with its nearest neighbor, but that neighbor might not be the best representative for a larger cluster.

3. **Empirical Results**: Clavie et al. (2024) found hierarchical pooling maintains 0.5-1% better quality at factor 4x compared to greedy. The difference is small at factor 2x (~0.1%), but grows with more aggressive pooling.

4. **When Greedy is Fine**: For factor 2-3x, greedy is fast and quality loss is minimal (~1%). Use hierarchical when:
   - Pooling factor ≥ 4x
   - Quality is critical
   - You can afford the extra computation

**Why Ward's Method Specifically:**

Ward's linkage minimizes the increase in total within-cluster variance when merging clusters. This is optimal for token pooling because:
- We want merged tokens to be semantically similar (low variance)
- The centroid (mean) of a low-variance cluster is a good representative
- Other linkage methods (complete, average) don't optimize for this

**Pooling Factor Guidelines:**
- **Factor 2x**: ~0% quality loss, 50% storage reduction (default)
- **Factor 3x**: ~1% quality loss, 67% storage reduction
- **Factor 4x**: ~3% quality loss, 75% storage reduction (use hierarchical)
- **Factor 8x+**: ~7% quality loss, only for extreme constraints

**Strategy Selection:**
- **Greedy**: Fast, good for factor 2-3x, ~1-3% quality loss
- **Hierarchical**: Best quality, slower, use for factor 4x+
- **Adaptive**: Balances quality and speed, good default
- **Sequential**: Simple, fast, but lower quality

## MMR Lambda Parameter

```
┌─────────────────────────────────────────────────────────┐
│ What's your primary goal?                                │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ EXPLORATION/DISCOVERY         │ PRECISION/SPECIFIC
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ λ = 0.3-0.5      │         │ λ = 0.7-0.9                │
│                  │         │                            │
│ • More diversity  │         │ • More relevance           │
│ • Broader        │         │ • Less redundancy          │
│   coverage       │         │ • Specific intent          │
│ • Discovery      │         │                            │
│ • RAG systems    │         │                            │
│   (default 0.5)  │         │                            │
└──────────────────┘         └──────────────────────────┘
```

**Lambda Effect:**
- **λ = 0.0**: Pure diversity (ignores relevance)
- **λ = 0.3-0.5**: Balanced (exploration, RAG default)
- **λ = 0.7-0.9**: Relevance-focused (precision search)
- **λ = 1.0**: Pure relevance (no diversity)

**Tuning Guidelines:**
- Start with λ=0.5 (balanced default)
- Increase if results are too diverse (users want precision)
- Decrease if results are too redundant (users want variety)
- Consider query-dependent lambda (exploratory vs specific queries)

## MMR vs DPP

```
┌─────────────────────────────────────────────────────────┐
│ How many results do you need?                           │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ k ≤ 20                       │ k > 20
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Consider DPP     │         │ Use MMR                   │
│                  │         │                           │
│ • Better         │         │ • Faster                  │
│   theoretical    │         │ • O(k × n) complexity    │
│   guarantees     │         │ • Good enough for large k│
│ • Captures       │         │                           │
│   pairwise       │         │                           │
│   diversity      │         │                           │
│ • O(k × n × d)   │         │                           │
│   complexity     │         │                           │
└──────────────────┘         └──────────────────────────┘
```

**When to Use Each:**
- **MMR**: General purpose, k > 20, latency-critical, simple
- **DPP**: Small k (≤20), pairwise diversity matters, better theoretical guarantees

**Trade-offs:**
- **MMR**: Faster, simpler, greedy (may miss optimal diversity)
- **DPP**: Slower, more complex, better diversity modeling (determinant-based)

## Matryoshka Refinement

```
┌─────────────────────────────────────────────────────────┐
│ Do you have two-stage retrieval available?              │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use Matryoshka   │         │ Use single-stage          │
│                  │         │                           │
│ • Coarse search  │         │                           │
│   with low dims  │         │                           │
│ • Refine with    │         │                           │
│   full dims      │         │                           │
│ • Best           │         │                           │
│   latency/quality│         │                           │
│   trade-off      │         │                           │
│                  │         │                           │
│ Example:         │         │                           │
│ Stage 1: 128 dims│         │                           │
│ Stage 2: 768 dims│         │                           │
└──────────────────┘         └──────────────────────────┘
```

**Matryoshka Strategy:**
- **Head dims (64-128)**: Coarse search, fast, good recall
- **Full dims (768)**: Refinement, slower, high precision
- **Two-stage**: Best of both worlds (fast recall + precise reranking)

## Cross-Encoder vs MaxSim

```
┌─────────────────────────────────────────────────────────┐
│ How many candidates do you need to rerank?              │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ ≤ 100                        │ > 100
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Consider Cross-  │         │ Use MaxSim               │
│ Encoder          │         │ (Late Interaction)       │
│                  │         │                           │
│ • Highest        │         │ • Pre-computed tokens    │
│   quality        │         │ • Fast reranking         │
│ • Full attention │         │ • Scales to 1000+        │
│ • Slow (per      │         │   candidates             │
│   pair)          │         │ • Good quality           │
│ • Can't index    │         │                           │
└──────────────────┘         └──────────────────────────┘
```

**When to Use Each:**
- **Cross-Encoder**: Small candidate sets (≤100), highest quality needed, can afford latency
- **MaxSim**: Large candidate sets (100+), need speed, pre-computed tokens available

**Note**: Cross-encoder module is currently disabled (waiting for ort 2.0 stability). Use MaxSim for now.

## Quick Reference Table

| Scenario | Method | Config | Notes |
|----------|--------|--------|-------|
| First-stage retrieval | Dense cosine | - | Fast, broad matching |
| Reranking | MaxSim (Late Interaction) | - | Token-level alignment |
| Storage-constrained | Token pooling | Factor 2-4x | Hierarchical best quality |
| Need diversity | MMR | λ=0.5 | Default for RAG |
| Small k, pairwise diversity | DPP | - | Better theoretical guarantees |
| Two-stage retrieval | Matryoshka | 128 + 768 dims | Best latency/quality |
| Small candidate set | Cross-encoder | - | Highest quality (when enabled) |
| Large candidate set | MaxSim | - | Fast, scalable |

## Common Mistakes

1. **Using dense for reranking**
   - Problem: Loses token-level semantics
   - Fix: Use MaxSim (late interaction) for reranking

2. **Pooling too aggressively**
   - Problem: Factor 8x+ causes significant quality loss
   - Fix: Use factor 2-4x, prefer hierarchical pooling

3. **Setting MMR lambda too high**
   - Problem: λ=1.0 gives no diversity, redundant results
   - Fix: Use λ=0.5 default, decrease for more diversity

4. **Using DPP for large k**
   - Problem: DPP is O(k × n × d), slow for k > 20
   - Fix: Use MMR for large k, DPP only for small k

5. **Not normalizing embeddings**
   - Problem: Cosine similarity assumes normalized vectors
   - Fix: Normalize before computing similarity

