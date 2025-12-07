#set page(margin: (x: 2.5cm, y: 2cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)

#show heading: set text(weight: "bold")

= rank-refine: Late Interaction Scoring Documentation

#align(center)[
  #text(size: 14pt, weight: "bold")[Late Interaction Scoring for Reranking]
  
  #text(size: 10pt)[ColBERT-style Token-Level Similarity]
  
  #v(0.5cm)
  #text(size: 9pt, style: "italic")[Version 0.1.0]
]

== Introduction

`rank-refine` provides SIMD-accelerated late interaction scoring for reranking, implementing ColBERT-style token-level similarity (MaxSim). This enables precise reranking of 100-1000 candidates in ~61ms per query.

== Features

- *MaxSim*: SIMD-accelerated token-level similarity
- *ColBERT-style*: Late interaction scoring for precision
- *Performance Target*: 61ms per query for 100-1000 candidates
- *Multiple Variants*: MaxSim, MaxSimCosine, MaxSimWeighted

== Quick Start

```rust
use rank_refine::maxsim;

let query_embeddings = vec![
    vec![0.1, 0.2, 0.3, ...], // token 1
    vec![0.4, 0.5, 0.6, ...], // token 2
];

let doc_embeddings = vec![
    vec![vec![0.2, 0.3, 0.4, ...], ...], // doc 1 tokens
    vec![vec![0.5, 0.6, 0.7, ...], ...], // doc 2 tokens
];

let scores = maxsim(&query_embeddings, &doc_embeddings);
// Returns similarity scores for each document
```

== MaxSim Algorithm

MaxSim computes token-level similarity between query and document:

$ "MaxSim"(q, d) = sum_(t "in" T_q) max_(t' "in" T_d) "sim"(e_t^q, e_(t')^d) $

#v(0.5em)
Parameters: T_q is query tokens; T_d is document tokens; e_t^q is embedding of query token t; e_t'^d is embedding of document token t'; sim is similarity function (dot product, cosine, etc.).

=== Variants

#v(0.3em)
- *MaxSim*: Dot product similarity
- *MaxSimCosine*: Cosine similarity
- MaxSimWeighted: Weighted similarity

== Performance

- Target: 61ms per query for 100-1000 candidates
- SIMD Acceleration: Vectorized operations for speed
- Memory Efficient: Optimized embedding storage

== Statistical Analysis

Real data analysis shows:

- MaxSim provides better alignment than dense scoring
- Score distributions follow expected patterns
- Correlation with query/document length

See hack/viz/ for detailed visualizations.

== Installation

```bash
cargo add rank-refine
```

== Examples

See examples/complete_workflow.rs for full usage.

== References

=== Primary Sources

- Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT". In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 39-48).

=== Related Work

- Late interaction techniques for information retrieval
- Token-level similarity computation for dense retrieval

== License

MIT OR Apache-2.0

