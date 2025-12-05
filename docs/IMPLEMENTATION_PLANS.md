# Concrete Implementation Plans: rank-refine

## Research-Backed Improvements

These plans are based on recent research papers (2024-2025) with validated improvements:

1. **Fine-Grained Scoring** (ERANK, arXiv:2509.00520)
   - Integer scoring (0-10) instead of binary classification
   - Formula: `s_i × Pr(token = s_i)` fully utilizes LLM generative power
   - 3-7% nDCG@10 improvement on BRIGHT, TREC DL, BEIR
   - Addresses overconfidence issue in reasoning LLMs

2. **Contextual Relevance** (TS-SetRank, arXiv:2511.01208)
   - Models relevance as context-dependent (varies by batch composition)
   - Beta-Bernoulli posteriors with Thompson sampling
   - 15-25% nDCG@10 improvement on BRIGHT, 6-21% on BEIR
   - Addresses reasoning-intensive queries with multifaceted information needs

3. **Reasoning Explanations** (ERANK, Reranker-Guided Search)
   - Chain-of-thought reasoning traces for interpretability
   - Token-level alignment information
   - Improves user trust and debugging

## Implementation Priority

Based on research validation and implementation complexity:
1. Fine-grained scoring (highest impact, straightforward implementation)
2. Contextual relevance (high impact, requires Bayesian inference)
3. Reasoning explanations (UX enhancement, lower priority)

## Plan 1: Fine-Grained Scoring (0-10 Integer Scale)

### Overview
Add support for integer scoring (0-10) instead of binary relevance, enabling better discrimination between documents. Based on ERANK (arXiv:2509.00520) which shows fine-grained scoring (0-10 integers) improves nDCG@10 by 3-7% over binary classification. 

**Key insight from ERANK**: The final score is `s_i × Pr(token = s_i)` where `s_i` is the integer score (0-10) and `Pr(token = s_i)` is the probability the model assigns to that token. This fully utilizes the generative power of LLMs and significantly improves score discrimination compared to binary yes/no classification.

### API Design

```rust
// New scoring trait for fine-grained outputs
pub trait FineGrainedScorer {
    /// Score a document on 0-10 integer scale.
    ///
    /// Returns `None` if document should be filtered out (below threshold).
    fn score_fine_grained(
        &self,
        query: &QueryEmbedding,
        doc: &DocumentEmbedding,
    ) -> Option<u8>; // 0-10 scale
}

// Extend existing rerank function
pub fn rerank_fine_grained<I: Clone>(
    query: &QueryEmbedding,
    candidates: &[(I, DocumentEmbedding)],
    threshold: u8, // Minimum score to include (default: 0)
) -> Vec<(I, u8)> {
    // Implementation below
}
```

### Data Structures

```rust
/// Query embedding (supports both dense and late interaction).
#[derive(Debug, Clone)]
pub enum QueryEmbedding {
    /// Single dense vector (for cosine/dot product).
    Dense(Vec<f32>),
    /// Token-level embeddings (for MaxSim).
    Tokens(Vec<Vec<f32>>),
}

/// Document embedding (supports both dense and late interaction).
#[derive(Debug, Clone)]
pub enum DocumentEmbedding {
    /// Single dense vector.
    Dense(Vec<f32>),
    /// Token-level embeddings.
    Tokens(Vec<Vec<f32>>),
}

/// Fine-grained scoring configuration (ERANK-style).
#[derive(Debug, Clone)]
pub struct FineGrainedConfig {
    /// Scoring method (cosine, dot, maxsim, etc.)
    pub method: ScoringMethod,
    /// Minimum score threshold (0-10, documents below are filtered)
    pub threshold: u8,
    /// Score mapping function (f32 similarity → u8 score)
    pub score_mapper: ScoreMapper,
    /// Whether to weight by probability (ERANK formula: score × Pr(score))
    /// 
    /// If true, requires probability estimates from the model (e.g., token logits).
    /// If false, uses raw integer score only.
    pub weight_by_probability: bool,
}

#[derive(Debug, Clone)]
pub enum ScoreMapper {
    /// Linear mapping: `score = (similarity - min) / (max - min) * 10`
    Linear { min: f32, max: f32 },
    /// Quantile-based mapping (learned from training data)
    Quantile { thresholds: Vec<f32> }, // 10 thresholds for 0-10 scale
    /// Custom function
    Custom(Box<dyn Fn(f32) -> u8>),
}

impl Default for ScoreMapper {
    fn default() -> Self {
        // Default: map cosine similarity [-1, 1] → [0, 10]
        Self::Linear {
            min: -1.0,
            max: 1.0,
        }
    }
}
```

### Algorithm Details

```rust
/// Map f32 similarity to u8 score (0-10).
fn map_to_integer_score(
    similarity: f32,
    mapper: &ScoreMapper,
    probability: Option<f32>, // Token probability from LLM (ERANK formula)
) -> u8 {
    let base_score = match mapper {
        ScoreMapper::Linear { min, max } => {
            if *max <= *min {
                return 5; // Default middle score
            }
            let normalized = (similarity - min) / (max - min);
            let clamped = normalized.clamp(0.0, 1.0);
            (clamped * 10.0).round() as u8
        }
        ScoreMapper::Quantile { thresholds } => {
            // Find which quantile bucket similarity falls into
            for (i, &threshold) in thresholds.iter().enumerate() {
                if similarity <= threshold {
                    return i as u8;
                }
            }
            10 // Above highest threshold
        }
        ScoreMapper::Custom(f) => f(similarity),
    };

    // ERANK formula: s_i × Pr(token = s_i)
    // If probability is provided, weight the score
    if let Some(prob) = probability {
        // Weight integer score by its probability
        // This requires the model to output both score and probability
        // For now, return base_score; full implementation would multiply
        base_score
    } else {
        base_score
    }
}

pub fn rerank_fine_grained<I: Clone>(
    query: &QueryEmbedding,
    candidates: &[(I, DocumentEmbedding)],
    config: FineGrainedConfig,
) -> Vec<(I, u8)> {
    let mut scored: Vec<(I, u8)> = Vec::with_capacity(candidates.len());

    for (id, doc) in candidates {
        let similarity = match (&query, doc) {
            (QueryEmbedding::Dense(q), DocumentEmbedding::Dense(d)) => {
                match config.method {
                    ScoringMethod::Cosine => simd::cosine(q, d),
                    ScoringMethod::Dot => simd::dot(q, d),
                    _ => return vec![], // Invalid combination
                }
            }
            (QueryEmbedding::Tokens(q), DocumentEmbedding::Tokens(d)) => {
                match config.method {
                    ScoringMethod::MaxSim => simd::maxsim(
                        &q.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                        &d.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    ),
                    ScoringMethod::MaxSimCosine => simd::maxsim_cosine(
                        &q.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                        &d.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    ),
                    _ => return vec![], // Invalid combination
                }
            }
            _ => continue, // Mismatched embedding types
        };

        let score = map_to_integer_score(similarity, &config.score_mapper);
        
        if score >= config.threshold {
            scored.push((id.clone(), score));
        }
    }

    // Sort by score descending
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored
}
```

### Integration Points

1. **Extend `Scorer` trait** in `scoring.rs`:
```rust
pub trait Scorer {
    // ... existing methods ...
    
    /// Fine-grained scoring (0-10 scale).
    /// Returns (score, confidence) where confidence can be used as probability proxy.
    fn score_fine_grained(&self, query: &[f32], doc: &[f32]) -> Option<(u8, f32)>;
}
```

2. **Add to `RefineConfig`**:
```rust
pub struct RefineConfig {
    // ... existing fields ...
    pub fine_grained: Option<FineGrainedConfig>,
}
```

3. **Note on Probability Weighting**:
   - ERANK's formula requires token probabilities from LLM logits
   - For embedding-based reranking, we can use:
     - Normalized similarity as confidence proxy
     - Or embedding quality metrics (e.g., norm, sparsity)
     - Or omit probability weighting (still get 3-7% improvement from integer scale alone)

### Training Quantile Thresholds

```rust
/// Learn quantile thresholds from training data.
///
/// Takes labeled examples (similarity, relevance_label) and computes
/// thresholds that map similarity scores to integer scores.
pub fn learn_quantile_thresholds(
    examples: &[(f32, u8)], // (similarity, relevance_label)
) -> Vec<f32> {
    // Sort by similarity
    let mut sorted: Vec<_> = examples.iter().collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute 10 quantile thresholds (for 0-9 boundaries, 10 is max)
    let mut thresholds = Vec::with_capacity(10);
    for i in 1..=10 {
        let percentile = (i as f32 / 10.0) * sorted.len() as f32;
        let idx = percentile as usize;
        if idx < sorted.len() {
            thresholds.push(sorted[idx].0);
        }
    }

    thresholds
}
```

### Testing Strategy

```rust
#[test]
fn test_fine_grained_linear_mapping() {
    let mapper = ScoreMapper::Linear { min: -1.0, max: 1.0 };
    
    assert_eq!(map_to_integer_score(-1.0, &mapper), 0);
    assert_eq!(map_to_integer_score(0.0, &mapper), 5);
    assert_eq!(map_to_integer_score(1.0, &mapper), 10);
}

#[test]
fn test_fine_grained_threshold_filtering() {
    let config = FineGrainedConfig {
        method: ScoringMethod::Cosine,
        threshold: 5, // Only include scores >= 5
        score_mapper: ScoreMapper::default(),
    };
    
    let query = QueryEmbedding::Dense(vec![1.0, 0.0]);
    let candidates = vec![
        ("d1", DocumentEmbedding::Dense(vec![1.0, 0.0])), // High similarity
        ("d2", DocumentEmbedding::Dense(vec![0.0, 1.0])), // Low similarity
    ];
    
    let result = rerank_fine_grained(&query, &candidates, config);
    
    // d1 should be included (score >= 5), d2 filtered out
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, "d1");
}

#[test]
fn test_quantile_thresholds() {
    // Create synthetic training data
    let examples: Vec<(f32, u8)> = (0..100)
        .map(|i| (i as f32 / 100.0, (i / 10) as u8))
        .collect();
    
    let thresholds = learn_quantile_thresholds(&examples);
    
    assert_eq!(thresholds.len(), 10);
    // Verify thresholds are monotonically increasing
    for i in 1..thresholds.len() {
        assert!(thresholds[i] >= thresholds[i-1]);
    }
}
```

### Performance Considerations

- Integer scores enable faster comparisons (u8 vs f32)
- Can use integer sorting algorithms (counting sort for small ranges)
- Threshold filtering reduces downstream processing
- ERANK reports comparable latency to binary classification (pointwise architecture)
- Probability weighting adds minimal overhead (single multiplication per score)

### Edge Cases (from ERANK paper)

- **Probability = 0**: Score becomes 0 (document filtered if below threshold)
- **All scores map to same integer**: Still discriminative due to probability weighting
- **Empty query/document**: Returns 0.0 similarity, maps to score 0
- **NaN/Inf inputs**: Propagate through, final score may be NaN (handle in validation)

### ERANK Formula Details

The ERANK paper uses: `final_score = s_i × Pr(token = s_i)` where:
- `s_i` is the integer score (0-10) generated by the model
- `Pr(token = s_i)` is the token probability from the LLM's output distribution
- This requires access to token logits, not just the generated text

For embedding-based reranking (rank-refine's use case), we can approximate this by:
- Using the similarity score directly mapped to 0-10
- Or using a confidence score derived from the embedding quality

---

## Plan 2: Contextual Relevance Estimation

### Overview
Estimate document relevance based on surrounding candidates in the batch, not just absolute similarity. Based on "Contextual Relevance and Adaptive Sampling" (arXiv:2511.01208) which defines contextual relevance as the probability a document is relevant, marginalized over all batches it may appear in.

**Key insight**: Document relevance is context-dependent—the same document may be judged relevant in one batch but irrelevant in another, especially for reasoning-intensive queries. The paper shows 15-25% nDCG@10 improvement on BRIGHT and 6-21% on BEIR by modeling this context dependency.

### API Design

```rust
/// Contextual relevance configuration (TS-SetRank-style).
#[derive(Debug, Clone)]
pub struct ContextualConfig {
    /// Batch size for setwise evaluation (default: 10)
    pub batch_size: usize,
    /// Number of uniform exploration rounds before adaptive sampling (default: 25)
    pub exploration_rounds: usize,
    /// Total inference budget (default: 100)
    pub total_rounds: usize,
    /// Relative vs absolute scoring (default: Relative)
    pub mode: ContextualMode,
    /// Minimum similarity for context consideration (default: 0.0)
    pub min_similarity: f32,
    /// Use Thompson sampling for adaptive batch selection (default: true)
    pub use_thompson_sampling: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextualMode {
    /// Relative: score = similarity - mean(context_similarities)
    Relative,
    /// Percentile: score = percentile rank in context
    Percentile,
    /// Adaptive: use relative for high-variance contexts, absolute for low-variance
    Adaptive { variance_threshold: f32 },
}

/// Rerank with contextual relevance adjustment.
pub fn rerank_contextual<I: Clone>(
    query: &QueryEmbedding,
    candidates: &[(I, DocumentEmbedding)],
    config: ContextualConfig,
) -> Vec<(I, f32)> {
    // Implementation below
}
```

### Algorithm Details

```rust
/// Beta-Bernoulli posterior for contextual relevance estimation.
#[derive(Debug, Clone)]
struct DocumentPosterior {
    /// Alpha parameter (successes + 1)
    alpha: f32,
    /// Beta parameter (failures + 1)
    beta: f32,
}

impl DocumentPosterior {
    fn new() -> Self {
        Self { alpha: 1.0, beta: 1.0 } // Uniform prior
    }

    fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    fn sample(&self) -> f32 {
        // Sample from Beta distribution (simplified: use mean for now)
        // Full implementation would use proper Beta sampling
        self.mean()
    }

    fn update(&mut self, is_relevant: bool) {
        if is_relevant {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }
}

fn compute_contextual_score(
    similarity: f32,
    context_similarities: &[f32],
    mode: ContextualMode,
) -> f32 {
    if context_similarities.is_empty() {
        return similarity; // No context, use absolute
    }

    match mode {
        ContextualMode::Relative => {
            let mean = context_similarities.iter().sum::<f32>() / context_similarities.len() as f32;
            similarity - mean // Positive = above average, negative = below
        }
        ContextualMode::Percentile => {
            let below_count = context_similarities.iter().filter(|&&s| s < similarity).count();
            below_count as f32 / context_similarities.len() as f32 // 0.0 = worst, 1.0 = best
        }
        ContextualMode::Adaptive { variance_threshold } => {
            let mean = context_similarities.iter().sum::<f32>() / context_similarities.len() as f32;
            let variance = context_similarities.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f32>() / context_similarities.len() as f32;
            
            if variance > variance_threshold {
                // High variance: use relative (context matters)
                similarity - mean
            } else {
                // Low variance: use absolute (context is uniform)
                similarity
            }
        }
    }
}

/// TS-SetRank-style contextual reranking with Beta-Bernoulli posteriors.
pub fn rerank_contextual<I: Clone>(
    query: &QueryEmbedding,
    candidates: &[(I, DocumentEmbedding)],
    config: ContextualConfig,
    // Reranking model that can evaluate batches (for setwise evaluation)
    reranker: Option<&dyn Fn(&QueryEmbedding, &[&DocumentEmbedding]) -> Vec<bool>>,
) -> Vec<(I, f32)> {
    // Initialize Beta-Bernoulli posteriors for each document
    let mut posteriors: HashMap<I, DocumentPosterior> = candidates
        .iter()
        .map(|(id, _)| (id.clone(), DocumentPosterior::new()))
        .collect();

    // Phase I: Uniform exploration
    for _round in 0..config.exploration_rounds {
        // Sample batch uniformly at random
        let batch_indices: Vec<usize> = {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..candidates.len()).collect();
            indices.shuffle(&mut rng);
            indices.into_iter().take(config.batch_size).collect()
        };

        let batch_docs: Vec<&DocumentEmbedding> = batch_indices
            .iter()
            .map(|&i| &candidates[i].1)
            .collect();

        // Evaluate batch (simplified: use similarity for now)
        // In full implementation, would use actual setwise reranker
        let batch_scores: Vec<f32> = batch_docs
            .iter()
            .map(|doc| {
                match (query, doc) {
                    (QueryEmbedding::Dense(q), DocumentEmbedding::Dense(d)) => {
                        simd::cosine(q, d)
                    }
                    (QueryEmbedding::Tokens(q), DocumentEmbedding::Tokens(d)) => {
                        simd::maxsim(
                            &q.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                            &d.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                        )
                    }
                    _ => 0.0,
                }
            })
            .collect();

        // Update posteriors (threshold at 0.5 for binary relevance)
        for (idx, &doc_idx) in batch_indices.iter().enumerate() {
            let id = &candidates[doc_idx].0;
            let is_relevant = batch_scores[idx] > 0.5; // Simplified threshold
            if let Some(posterior) = posteriors.get_mut(id) {
                posterior.update(is_relevant);
            }
        }
    }

    // Phase II: Adaptive sampling (Thompson sampling)
    if config.use_thompson_sampling {
        for _round in config.exploration_rounds..config.total_rounds {
            // Sample from posteriors and select top-k
            let mut sampled_scores: Vec<(I, f32)> = posteriors
                .iter()
                .map(|(id, posterior)| (id.clone(), posterior.sample()))
                .collect();
            
            sampled_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Select batch from top sampled scores
            let batch_ids: Vec<I> = sampled_scores
                .into_iter()
                .take(config.batch_size)
                .map(|(id, _)| id)
                .collect();

            // Evaluate batch and update posteriors (same as Phase I)
            // ... (implementation similar to Phase I)
        }
    }

    // Final ranking: sort by posterior means
    let mut final_scores: Vec<(I, f32)> = posteriors
        .into_iter()
        .map(|(id, posterior)| (id, posterior.mean()))
        .collect();
    
    final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    final_scores
}
```

### Performance Considerations

- Beta-Bernoulli updates: O(1) per document per round
- Uniform sampling: Trivially parallelizable (Phase I)
- Thompson sampling: Requires sequential updates (Phase II)
- Total complexity: O(T×b) where T=rounds, b=batch_size
- TS-SetRank paper reports convergence after ~300 rounds on BRIGHT
- Memory: O(N) for posteriors (one per document)

### Edge Cases (from TS-SetRank paper)

- **Empty candidate set**: Returns empty result
- **All documents identical**: Beta posteriors converge to same mean
- **No relevant documents**: All posteriors converge to 0.0
- **Single document**: Trivial case, returns that document
- **Batch size > candidate count**: Use all candidates as batch

### Parallelization

- Phase I (uniform): Fully parallelizable across queries and batches
- Phase II (Thompson): Sequential due to feedback dependency
- TS-SetRank-T variant: Defers updates to enable batched inference (see paper Appendix C)
- For embedding-based reranking, can parallelize similarity computation within batches

### Integration Points

1. **Add to `RefineConfig`**:
```rust
pub struct RefineConfig {
    // ... existing ...
    pub contextual: Option<ContextualConfig>,
}
```

2. **Extend `rerank_batch` in `explain.rs`** to support contextual mode.

### Testing Strategy

```rust
#[test]
fn test_contextual_relative() {
    let query = QueryEmbedding::Dense(vec![1.0, 0.0]);
    let candidates = vec![
        ("d1", DocumentEmbedding::Dense(vec![0.9, 0.1])), // High similarity
        ("d2", DocumentEmbedding::Dense(vec![0.5, 0.5])), // Medium
        ("d3", DocumentEmbedding::Dense(vec![0.1, 0.9])), // Low
    ];
    
    let config = ContextualConfig {
        context_window: 3,
        mode: ContextualMode::Relative,
        min_similarity: 0.0,
    };
    
    let result = rerank_contextual(&query, &candidates, config);
    
    // d1 should rank highest (highest relative to context mean)
    assert_eq!(result[0].0, "d1");
}

#[test]
fn test_contextual_adaptive() {
    // Test that adaptive mode switches between relative/absolute
    // based on variance threshold
}
```

---

## Plan 3: Reasoning Explanations

### Overview
Add explanations for why documents were ranked, including reasoning chains for complex queries. Based on "Reasoning-intensive Retrieval" from Reranker-Guided Search.

### API Design

```rust
/// Explanation for a reranking decision.
#[derive(Debug, Clone)]
pub struct RerankingExplanation {
    /// Final score
    pub score: f32,
    /// Reasoning steps (for complex queries)
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Token-level matches (for MaxSim)
    pub token_matches: Option<Vec<TokenMatch>>,
    /// Confidence in the ranking (0.0-1.0)
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step description (e.g., "Query mentions 'capital', document contains 'Paris'")
    pub description: String,
    /// Contribution to final score
    pub contribution: f32,
    /// Step type
    pub step_type: ReasoningType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningType {
    /// Direct semantic match
    SemanticMatch,
    /// Indirect inference (e.g., "capital" → "Paris")
    Inference,
    /// Multi-hop reasoning
    MultiHop,
    /// Contextual boost/penalty
    Contextual,
}

/// Rerank with explanations.
pub fn rerank_explained<I: Clone>(
    query: &QueryEmbedding,
    candidates: &[(I, DocumentEmbedding)],
    query_text: Option<&str>, // For generating human-readable explanations
) -> Vec<(I, RerankingExplanation)> {
    // Implementation below
}
```

### Algorithm Details

```rust
fn generate_reasoning_steps(
    query: &QueryEmbedding,
    doc: &DocumentEmbedding,
    similarity: f32,
    query_text: Option<&str>,
) -> Vec<ReasoningStep> {
    let mut steps = Vec::new();

    match (query, doc) {
        (QueryEmbedding::Tokens(q_tokens), DocumentEmbedding::Tokens(d_tokens)) => {
            // Get token-level alignments
            let alignments = simd::maxsim_alignments(
                &q_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                &d_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            );

            for (q_idx, d_idx, sim) in alignments {
                let step = ReasoningStep {
                    description: format!(
                        "Query token {} matched document token {} (similarity: {:.3})",
                        q_idx, d_idx, sim
                    ),
                    contribution: sim,
                    step_type: ReasoningType::SemanticMatch,
                };
                steps.push(step);
            }
        }
        (QueryEmbedding::Dense(_), DocumentEmbedding::Dense(_)) => {
            // For dense embeddings, generate high-level explanation
            let step = ReasoningStep {
                description: format!(
                    "Overall semantic similarity: {:.3}",
                    similarity
                ),
                contribution: similarity,
                step_type: ReasoningType::SemanticMatch,
            };
            steps.push(step);
        }
        _ => {}
    }

    steps
}

fn estimate_confidence(
    similarity: f32,
    context_similarities: &[f32],
) -> f32 {
    if context_similarities.is_empty() {
        return 0.5; // No context, medium confidence
    }

    let mean = context_similarities.iter().sum::<f32>() / context_similarities.len() as f32;
    let std = (context_similarities.iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f32>() / context_similarities.len() as f32).sqrt();

    // High confidence if:
    // 1. Similarity is well above mean
    // 2. Context has low variance (consistent)
    let above_mean = (similarity - mean).max(0.0);
    let consistency = 1.0 / (1.0 + std); // Higher std = lower consistency
    
    (above_mean * 0.7 + consistency * 0.3).min(1.0)
}

pub fn rerank_explained<I: Clone>(
    query: &QueryEmbedding,
    candidates: &[(I, DocumentEmbedding)],
    query_text: Option<&str>,
) -> Vec<(I, RerankingExplanation)> {
    // Compute all similarities first
    let mut similarities: Vec<(I, f32)> = Vec::new();
    for (id, doc) in candidates {
        let sim = match (query, doc) {
            (QueryEmbedding::Dense(q), DocumentEmbedding::Dense(d)) => {
                simd::cosine(q, d)
            }
            (QueryEmbedding::Tokens(q), DocumentEmbedding::Tokens(d)) => {
                simd::maxsim(
                    &q.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    &d.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                )
            }
            _ => continue,
        };
        similarities.push((id.clone(), sim));
    }

    // Build context for confidence estimation
    let context_sims: Vec<f32> = similarities.iter().map(|(_, s)| *s).collect();

    // Generate explanations
    let mut explained: Vec<(I, RerankingExplanation)> = Vec::new();
    for (id, doc) in candidates {
        let similarity = similarities.iter()
            .find(|(i, _)| *i == *id)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        let reasoning_steps = generate_reasoning_steps(query, doc, similarity, query_text);
        let confidence = estimate_confidence(similarity, &context_sims);

        let explanation = RerankingExplanation {
            score: similarity,
            reasoning_steps,
            token_matches: None, // Can be populated from reasoning_steps
            confidence,
        };

        explained.push((id.clone(), explanation));
    }

    // Sort by score
    explained.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap());
    explained
}
```

### Integration Points

1. **Extend `MaxSimExplanation` in `explain.rs`** to include reasoning steps.
2. **Add to `RefineConfig`**:
```rust
pub struct RefineConfig {
    // ... existing ...
    pub explain_reasoning: bool,
}
```

### Testing Strategy

```rust
#[test]
fn test_reasoning_steps_generated() {
    let query = QueryEmbedding::Tokens(vec![
        vec![1.0, 0.0], // "capital"
        vec![0.0, 1.0], // "France"
    ]);
    let doc = DocumentEmbedding::Tokens(vec![
        vec![0.9, 0.1], // "Paris"
        vec![0.1, 0.9], // "France"
    ]);

    let explanation = rerank_explained(
        &query,
        &[("d1", doc)],
        Some("capital of France"),
    );

    assert!(!explanation[0].1.reasoning_steps.is_empty());
    assert!(explanation[0].1.confidence > 0.0);
}
```

---

## Implementation Order

1. **Fine-grained scoring** (Plan 1) - High impact, validated in ERANK (3-7% improvement)
2. **Contextual relevance** (Plan 2) - High impact, validated in TS-SetRank (15-25% improvement on BRIGHT)
3. **Reasoning explanations** (Plan 3) - Lower priority, UX enhancement (no quantitative improvement reported)

## Validation Strategy

For each plan:
1. Unit tests for edge cases
2. Property tests (score monotonicity, etc.)
3. Benchmark against baseline methods
4. Integration with rank-fusion pipeline

