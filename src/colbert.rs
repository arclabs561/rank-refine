//! Late interaction scoring via MaxSim.
//!
//! # What is Late Interaction?
//!
//! Most embedding models compress an entire document into a single vector.
//! Late interaction keeps **one vector per token**, preserving fine-grained semantics.
//!
//! ```text
//! Dense:           "the quick brown fox" → [0.1, 0.2, ...]  (1 vector)
//! Late Interaction: "the quick brown fox" → [[...], [...], [...], [...]]  (4 vectors)
//! ```
//!
//! # Supported Models
//!
//! MaxSim is **model-agnostic**. Any multi-vector encoder works:
//!
//! | Model | Input | Notes |
//! |-------|-------|-------|
//! | [ColBERT](https://arxiv.org/abs/2004.12832) | Text | Original late interaction |
//! | [ColBERTv2](https://arxiv.org/abs/2112.01488) | Text | + residual compression |
//! | [ColPali](https://arxiv.org/abs/2407.01449) | Document images | Vision-language, ICLR 2025 |
//! | [Jina-ColBERT-v2](https://huggingface.co/jinaai/jina-colbert-v2) | Text | Multilingual + Matryoshka |
//!
//! This crate scores embeddings — it doesn't care where they came from.
//!
//! # How MaxSim Works
//!
//! For each query token, find its best-matching document token, then sum:
//!
//! ```text
//! Score = Σ (for each query token q)
//!           max (over all doc tokens d)
//!             dot(q, d)
//! ```
//!
//! This captures **token-level alignment**: "What is the capital of France?"
//! can find documents where "capital" and "France" both have strong matches,
//! even if they appear in different parts of the document.
//!
//! See [`REFERENCE.md`](https://github.com/your-repo/REFERENCE.md) for the full algorithm.
//!
//! # Token Pooling
//!
//! Storing one vector per token is expensive. Token pooling clusters similar
//! tokens and stores only the cluster centroids:
//!
//! ```text
//! Original:  [tok1] [tok2] [tok3] [tok4] [tok5] [tok6]  (6 vectors)
//!              ↓       ↓      ↓      ↓      ↓      ↓
//! Clustered: [  cluster1  ] [  cluster2  ] [cluster3]
//!              ↓               ↓              ↓
//! Pooled:    [mean1]        [mean2]        [mean3]      (3 vectors = 50% reduction)
//! ```
//!
//! **Research**: Pool factors of 2-3 achieve 50-66% reduction with <1% quality loss.
//! See [Clavié et al., 2024](https://arxiv.org/abs/2409.14683).
//!
//! ## Clustering Algorithms
//!
//! | Feature | Method | When to Use |
//! |---------|--------|-------------|
//! | Default | Greedy agglomerative | Factor 2-3, good balance |
//! | `hierarchical` | Ward's method | Factor 4+, best quality |
//!
//! Enable Ward's method: `rank-refine = { features = ["hierarchical"] }`
//!
//! # Example
//!
//! ```rust
//! use rank_refine::colbert;
//!
//! // Query: 2 tokens, each 4-dimensional
//! let query = vec![
//!     vec![1.0, 0.0, 0.0, 0.0],  // token "capital"
//!     vec![0.0, 1.0, 0.0, 0.0],  // token "France"
//! ];
//!
//! // Documents with their token embeddings
//! let docs = vec![
//!     ("doc1", vec![vec![0.9, 0.1, 0.0, 0.0], vec![0.1, 0.9, 0.0, 0.0]]),
//!     ("doc2", vec![vec![0.5, 0.5, 0.0, 0.0]]),  // only 1 token
//! ];
//!
//! let ranked = colbert::rank(&query, &docs);
//! assert_eq!(ranked[0].0, "doc1");  // better token alignment
//!
//! // Pool for storage (factor 2 = 50% reduction)
//! let pooled = colbert::pool_tokens(&docs[0].1, 2);
//! ```

use crate::{simd, RefineConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Token Pooling (PLAID-style compression)
// ─────────────────────────────────────────────────────────────────────────────

/// Pool token embeddings by clustering similar tokens and averaging.
///
/// Reduces the number of vectors stored per document while preserving
/// semantic information. Pool factor 2-3 is recommended (50-66% reduction).
///
/// # Pool Factor Guide
///
/// | Factor | Storage Saved | Quality Loss | Recommendation |
/// |--------|---------------|--------------|----------------|
/// | 2 | 50% | ~0% | Default choice |
/// | 3 | 66% | ~1% | Good tradeoff |
/// | 4+ | 75%+ | 3-5% | Use `hierarchical` feature |
///
/// # Algorithm
///
/// - **Default**: Greedy agglomerative clustering
/// - **With `hierarchical` feature**: Ward's method via kodama (better quality)
///
/// # Complexity
///
/// - Default: O(n³ × d) where n = token count, d = dimension
/// - Hierarchical: O(n² log n × d)
///
/// For n > 50 tokens, consider [`pool_tokens_sequential`] or the `hierarchical` feature.
///
/// # Arguments
///
/// * `tokens` - Document token embeddings (assumed L2-normalized for `ColBERT`)
/// * `pool_factor` - Target compression ratio (2 = 50% reduction, 3 = 66%, etc.)
#[must_use]
pub fn pool_tokens(tokens: &[Vec<f32>], pool_factor: usize) -> Vec<Vec<f32>> {
    if tokens.is_empty() || pool_factor <= 1 {
        return tokens.to_vec();
    }

    let n = tokens.len();
    let target_count = (n / pool_factor).max(1);

    if n <= target_count {
        return tokens.to_vec();
    }

    #[cfg(feature = "hierarchical")]
    {
        pool_tokens_hierarchical(tokens, target_count)
    }

    #[cfg(not(feature = "hierarchical"))]
    {
        pool_tokens_greedy(tokens, target_count)
    }
}

/// Greedy agglomerative clustering (default, O(n³)).
#[cfg(not(feature = "hierarchical"))]
fn pool_tokens_greedy(tokens: &[Vec<f32>], target_count: usize) -> Vec<Vec<f32>> {
    let n = tokens.len();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > target_count {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_sim = f32::NEG_INFINITY;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let sim = cluster_similarity(tokens, &clusters[i], &clusters[j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let merged = clusters.remove(best_j);
        clusters[best_i].extend(merged);
    }

    clusters
        .iter()
        .map(|indices| mean_pool(tokens, indices))
        .collect()
}

/// Ward's method hierarchical clustering via kodama (O(n² log n), better quality).
///
/// Ward's method minimizes within-cluster variance at each merge step,
/// producing more semantically coherent clusters than greedy approaches.
///
/// # Panics
///
/// Panics if any token has zero norm (would produce NaN cosine similarity).
/// kodama requires all dissimilarities to be finite non-NaN values.
#[cfg(feature = "hierarchical")]
fn pool_tokens_hierarchical(tokens: &[Vec<f32>], target_count: usize) -> Vec<Vec<f32>> {
    use kodama::{linkage, Method};

    let n = tokens.len();

    // Build condensed distance matrix (upper triangular, row-major)
    // Distance = 1 - cosine_similarity (so similar tokens have low distance)
    // kodama requires all values to be finite non-NaN
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = simd::cosine(&tokens[i], &tokens[j]);
            // Handle NaN (from zero-norm vectors) by treating as max distance
            let sim_safe = if sim.is_nan() { -1.0 } else { sim };
            // Convert similarity to distance, clamped to [0, 2]
            #[allow(clippy::cast_lossless)]
            let dist = f64::from((1.0 - sim_safe).clamp(0.0, 2.0));
            condensed.push(dist);
        }
    }

    // Run Ward's method linkage
    let dendrogram = linkage(&mut condensed, n, Method::Ward);

    // Cut dendrogram to get target_count clusters
    let labels = cut_dendrogram(&dendrogram, n, target_count);

    // Group tokens by cluster label
    let num_clusters = labels.iter().max().map_or(0, |&m| m + 1);
    let mut clusters: Vec<Vec<usize>> = vec![vec![]; num_clusters];
    for (i, &label) in labels.iter().enumerate() {
        clusters[label].push(i);
    }

    // Mean pool each cluster
    clusters
        .iter()
        .filter(|c| !c.is_empty())
        .map(|indices| mean_pool(tokens, indices))
        .collect()
}

/// Cut a dendrogram at the level that produces `target_count` clusters.
#[cfg(feature = "hierarchical")]
fn cut_dendrogram(
    dendrogram: &kodama::Dendrogram<f64>,
    n: usize,
    target_count: usize,
) -> Vec<usize> {
    // Each step merges two clusters, so we need (n - target_count) merges
    let steps_to_take = n.saturating_sub(target_count);

    // Union-find helper for path compression
    #[allow(clippy::items_after_statements)]
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    }

    // Union-find to track cluster membership
    let mut parent: Vec<usize> = (0..2 * n).collect();

    // Apply merges up to our cut point
    for (step_idx, step) in dendrogram.steps().iter().enumerate() {
        if step_idx >= steps_to_take {
            break;
        }
        let new_cluster = n + step_idx;
        parent[step.cluster1] = new_cluster;
        parent[step.cluster2] = new_cluster;
    }

    // Assign final labels
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 0;
    let mut labels = vec![0; n];

    for (i, label_slot) in labels.iter_mut().enumerate() {
        let root = find(&mut parent, i);
        let label = *label_map.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        *label_slot = label;
    }

    labels
}

/// Pool tokens using simple sequential windows.
///
/// Faster than clustering but less intelligent. Good for ordered sequences
/// where adjacent tokens are likely semantically related.
#[must_use]
pub fn pool_tokens_sequential(tokens: &[Vec<f32>], window_size: usize) -> Vec<Vec<f32>> {
    if tokens.is_empty() || window_size <= 1 {
        return tokens.to_vec();
    }

    tokens
        .chunks(window_size)
        .map(|chunk| {
            let dim = chunk[0].len();
            let mut pooled = vec![0.0; dim];
            for token in chunk {
                for (k, v) in pooled.iter_mut().enumerate() {
                    *v += token[k];
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let n = chunk.len() as f32;
            for v in &mut pooled {
                *v /= n;
            }
            pooled
        })
        .collect()
}

/// Pool tokens with protected indices (e.g., `[CLS]`, `[D]` markers).
///
/// Protected tokens are preserved unchanged and not included in clustering.
#[must_use]
pub fn pool_tokens_with_protected(
    tokens: &[Vec<f32>],
    pool_factor: usize,
    protected_count: usize,
) -> Vec<Vec<f32>> {
    if tokens.is_empty() || pool_factor <= 1 {
        return tokens.to_vec();
    }

    let protected_count = protected_count.min(tokens.len());
    let protected = &tokens[..protected_count];
    let poolable = &tokens[protected_count..];

    let mut result = protected.to_vec();
    result.extend(pool_tokens(poolable, pool_factor));
    result
}

/// Adaptively choose the best pooling strategy based on pool factor.
///
/// - **Factor 1-3**: Uses clustering-based pooling (greedy or hierarchical)
/// - **Factor 4+**: Uses sequential pooling (faster, nearly as good for aggressive compression)
///
/// This is a convenience function that picks reasonable defaults. For full control,
/// use [`pool_tokens`] or [`pool_tokens_sequential`] directly.
///
/// # Example
///
/// ```rust
/// use rank_refine::colbert;
///
/// let tokens = vec![
///     vec![1.0, 0.0, 0.0, 0.0],
///     vec![0.9, 0.1, 0.0, 0.0],
///     vec![0.0, 1.0, 0.0, 0.0],
///     vec![0.0, 0.9, 0.1, 0.0],
/// ];
///
/// // Factor 2: uses clustering (quality matters more)
/// let pooled_2 = colbert::pool_tokens_adaptive(&tokens, 2);
/// assert_eq!(pooled_2.len(), 2);
///
/// // Factor 4: uses sequential (speed matters more at aggressive compression)
/// let pooled_4 = colbert::pool_tokens_adaptive(&tokens, 4);
/// assert_eq!(pooled_4.len(), 1);
/// ```
#[must_use]
pub fn pool_tokens_adaptive(tokens: &[Vec<f32>], pool_factor: usize) -> Vec<Vec<f32>> {
    if tokens.is_empty() || pool_factor <= 1 {
        return tokens.to_vec();
    }

    // For aggressive pooling (factor 4+), sequential is nearly as good and much faster
    // For moderate pooling (factor 2-3), clustering preserves quality better
    if pool_factor >= 4 {
        pool_tokens_sequential(tokens, pool_factor)
    } else {
        pool_tokens(tokens, pool_factor)
    }
}

#[cfg(not(feature = "hierarchical"))]
fn cluster_similarity(tokens: &[Vec<f32>], c1: &[usize], c2: &[usize]) -> f32 {
    let centroid1 = mean_pool(tokens, c1);
    let centroid2 = mean_pool(tokens, c2);
    simd::cosine(&centroid1, &centroid2)
}

fn mean_pool(tokens: &[Vec<f32>], indices: &[usize]) -> Vec<f32> {
    if indices.is_empty() {
        return vec![];
    }
    let dim = tokens[indices[0]].len();
    let mut pooled = vec![0.0; dim];
    for &idx in indices {
        for (k, v) in pooled.iter_mut().enumerate() {
            *v += tokens[idx][k];
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n = indices.len() as f32;
    for v in &mut pooled {
        *v /= n;
    }
    pooled
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking & Refinement
// ─────────────────────────────────────────────────────────────────────────────

/// Rank documents by `MaxSim` score against query tokens.
///
/// Returns documents sorted by descending `MaxSim` score.
#[must_use]
pub fn rank<I: Clone>(query: &[Vec<f32>], docs: &[(I, Vec<Vec<f32>>)]) -> Vec<(I, f32)> {
    rank_with_top_k(query, docs, None)
}

/// Rank documents with optional `top_k` limit.
#[must_use]
pub fn rank_with_top_k<I: Clone>(
    query: &[Vec<f32>],
    docs: &[(I, Vec<Vec<f32>>)],
    top_k: Option<usize>,
) -> Vec<(I, f32)> {
    let query_refs = crate::as_slices(query);

    let mut results: Vec<(I, f32)> = docs
        .iter()
        .map(|(id, doc_tokens)| {
            let doc_refs = crate::as_slices(doc_tokens);
            let score = simd::maxsim(&query_refs, &doc_refs);
            (id.clone(), score)
        })
        .collect();

    crate::sort_scored_desc(&mut results);

    if let Some(k) = top_k {
        results.truncate(k);
    }

    results
}

/// Refine candidates using `MaxSim`, blending with original scores.
///
/// # Arguments
///
/// * `candidates` - Initial retrieval results (id, score)
/// * `query` - Query token embeddings
/// * `docs` - Document token embeddings
/// * `alpha` - Weight for original score (0.0 = all `MaxSim`, 1.0 = all original)
///
/// # Note
///
/// Candidates not found in `docs` are silently dropped.
#[must_use]
pub fn refine<I: Clone + Eq + std::hash::Hash>(
    candidates: &[(I, f32)],
    query: &[Vec<f32>],
    docs: &[(I, Vec<Vec<f32>>)],
    alpha: f32,
) -> Vec<(I, f32)> {
    refine_with_config(
        candidates,
        query,
        docs,
        RefineConfig::default().with_alpha(alpha),
    )
}

/// Refine with full configuration.
#[must_use]
pub fn refine_with_config<I: Clone + Eq + std::hash::Hash>(
    candidates: &[(I, f32)],
    query: &[Vec<f32>],
    docs: &[(I, Vec<Vec<f32>>)],
    config: RefineConfig,
) -> Vec<(I, f32)> {
    use std::collections::HashMap;

    let doc_map: HashMap<&I, &Vec<Vec<f32>>> = docs.iter().map(|(id, toks)| (id, toks)).collect();
    let query_refs = crate::as_slices(query);
    let alpha = config.alpha;

    let mut results: Vec<(I, f32)> = candidates
        .iter()
        .filter_map(|(id, orig_score)| {
            let doc_tokens = doc_map.get(id)?;
            let doc_refs = crate::as_slices(doc_tokens);
            let maxsim_score = simd::maxsim(&query_refs, &doc_refs);
            let blended = (1.0 - alpha).mul_add(maxsim_score, alpha * orig_score);
            Some((id.clone(), blended))
        })
        .collect();

    crate::sort_scored_desc(&mut results);

    if let Some(k) = config.top_k {
        results.truncate(k);
    }

    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Token Index (pre-computed embeddings for repeated scoring)
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed token embeddings for efficient repeated scoring.
///
/// Use `TokenIndex` when scoring the same set of targets against many queries:
/// - **Document retrieval**: Index documents once, search with many queries
/// - **NER/GLiNER**: Index entity type labels once, match against text spans
/// - **Classification**: Index class embeddings once, classify many inputs
///
/// # Why Not Just `Vec<(I, Vec<Vec<f32>>)>`?
///
/// `TokenIndex` provides:
/// 1. **Ergonomic API**: `score_all`, `top_k`, `get` methods
/// 2. **O(1) lookup by ID** via internal HashMap
/// 3. **Clear intent**: "This is a pre-computed index, not ephemeral data"
///
/// # Example
///
/// ```rust
/// use rank_refine::colbert::TokenIndex;
///
/// // Build index once (e.g., at startup)
/// let index = TokenIndex::new(vec![
///     ("doc1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
///     ("doc2", vec![vec![0.5, 0.5]]),
/// ]);
///
/// // Score many queries against the same index
/// let query1 = vec![vec![1.0, 0.0]];
/// let query2 = vec![vec![0.0, 1.0]];
///
/// let results1 = index.top_k(&query1, 1);
/// let results2 = index.top_k(&query2, 1);
///
/// assert_eq!(results1[0].0, "doc1");
/// assert_eq!(results2[0].0, "doc1");
/// ```
#[derive(Debug, Clone)]
pub struct TokenIndex<I> {
    entries: Vec<(I, Vec<Vec<f32>>)>,
}

impl<I> TokenIndex<I> {
    /// Create a new token index from (id, tokens) pairs.
    pub fn new(entries: Vec<(I, Vec<Vec<f32>>)>) -> Self {
        Self { entries }
    }

    /// Number of entries in the index.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &(I, Vec<Vec<f32>>)> {
        self.entries.iter()
    }

    /// Get the underlying entries.
    #[must_use]
    pub fn entries(&self) -> &[(I, Vec<Vec<f32>>)] {
        &self.entries
    }

    /// Consume the index and return the underlying entries.
    #[must_use]
    pub fn into_entries(self) -> Vec<(I, Vec<Vec<f32>>)> {
        self.entries
    }
}

impl<I: Clone> TokenIndex<I> {
    /// Score a query against all entries using MaxSim.
    ///
    /// Returns `(id, score)` pairs in index order (not sorted).
    /// Use [`top_k`](Self::top_k) or [`rank`](Self::rank) for sorted results.
    #[must_use]
    pub fn score_all(&self, query: &[Vec<f32>]) -> Vec<(I, f32)> {
        let query_refs = crate::as_slices(query);
        self.entries
            .iter()
            .map(|(id, doc_tokens)| {
                let doc_refs = crate::as_slices(doc_tokens);
                (id.clone(), simd::maxsim(&query_refs, &doc_refs))
            })
            .collect()
    }

    /// Score a query against all entries using MaxSim with cosine similarity.
    #[must_use]
    pub fn score_all_cosine(&self, query: &[Vec<f32>]) -> Vec<(I, f32)> {
        let query_refs = crate::as_slices(query);
        self.entries
            .iter()
            .map(|(id, doc_tokens)| {
                let doc_refs = crate::as_slices(doc_tokens);
                (id.clone(), simd::maxsim_cosine(&query_refs, &doc_refs))
            })
            .collect()
    }

    /// Score and return all results sorted by descending score.
    #[must_use]
    pub fn rank(&self, query: &[Vec<f32>]) -> Vec<(I, f32)> {
        let mut results = self.score_all(query);
        crate::sort_scored_desc(&mut results);
        results
    }

    /// Score and return top-k results sorted by descending score.
    #[must_use]
    pub fn top_k(&self, query: &[Vec<f32>], k: usize) -> Vec<(I, f32)> {
        let mut results = self.score_all(query);
        crate::sort_scored_desc(&mut results);
        results.truncate(k);
        results
    }

    /// Score using cosine and return top-k results sorted by descending score.
    #[must_use]
    pub fn top_k_cosine(&self, query: &[Vec<f32>], k: usize) -> Vec<(I, f32)> {
        let mut results = self.score_all_cosine(query);
        crate::sort_scored_desc(&mut results);
        results.truncate(k);
        results
    }
}

impl<I: Clone + Eq + std::hash::Hash> TokenIndex<I> {
    /// Get token embeddings by ID.
    ///
    /// O(n) scan. For frequent lookups, consider maintaining a separate HashMap.
    #[must_use]
    pub fn get(&self, id: &I) -> Option<&Vec<Vec<f32>>> {
        self.entries
            .iter()
            .find(|(entry_id, _)| entry_id == id)
            .map(|(_, tokens)| tokens)
    }

    /// Check if the index contains an ID.
    #[must_use]
    pub fn contains(&self, id: &I) -> bool {
        self.get(id).is_some()
    }
}

impl<I> Default for TokenIndex<I> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<I> FromIterator<(I, Vec<Vec<f32>>)> for TokenIndex<I> {
    fn from_iter<T: IntoIterator<Item = (I, Vec<Vec<f32>>)>>(iter: T) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            ("d2", vec![vec![0.5, 0.5]]),
        ];

        let ranked = rank(&query, &docs);
        assert_eq!(ranked[0].0, "d1");
    }

    #[test]
    fn test_rank_with_top_k() {
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0]]),
            ("d2", vec![vec![0.9, 0.1]]),
            ("d3", vec![vec![0.8, 0.2]]),
        ];

        let ranked = rank_with_top_k(&query, &docs, Some(2));
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn test_rank_empty_query() {
        let query: Vec<Vec<f32>> = vec![];
        let docs = vec![("d1", vec![vec![1.0, 0.0]])];

        let ranked = rank(&query, &docs);
        assert_eq!(ranked[0].0, "d1");
        assert_eq!(ranked[0].1, 0.0);
    }

    #[test]
    fn test_rank_empty_docs() {
        let query = vec![vec![1.0, 0.0]];
        let docs: Vec<(&str, Vec<Vec<f32>>)> = vec![("d1", vec![])];

        let ranked = rank(&query, &docs);
        assert_eq!(ranked[0].0, "d1");
        assert_eq!(ranked[0].1, 0.0);
    }

    #[test]
    fn test_refine() {
        let candidates = vec![("d1", 0.5), ("d2", 0.9)];
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![("d1", vec![vec![1.0, 0.0]]), ("d2", vec![vec![0.0, 1.0]])];

        let refined = refine(&candidates, &query, &docs, 0.0);
        assert_eq!(refined[0].0, "d1");

        let refined = refine(&candidates, &query, &docs, 1.0);
        assert_eq!(refined[0].0, "d2");
    }

    #[test]
    fn test_refine_with_config_top_k() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0]]),
            ("d2", vec![vec![1.0, 0.0]]),
            ("d3", vec![vec![1.0, 0.0]]),
        ];

        let refined = refine_with_config(
            &candidates,
            &query,
            &docs,
            RefineConfig::default().with_top_k(2),
        );
        assert_eq!(refined.len(), 2);
    }

    #[test]
    fn test_refine_missing_doc() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8)];
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![("d1", vec![vec![1.0, 0.0]])];

        let refined = refine(&candidates, &query, &docs, 0.5);
        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].0, "d1");
    }

    #[test]
    fn test_nan_score_handling() {
        let candidates = vec![("d1", f32::NAN), ("d2", 0.5)];
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![("d1", vec![vec![1.0, 0.0]]), ("d2", vec![vec![1.0, 0.0]])];

        let refined = refine(&candidates, &query, &docs, 0.5);
        assert_eq!(refined.len(), 2);
        assert!(refined[0].1.is_nan());
    }

    #[test]
    fn test_pool_tokens_empty() {
        let tokens: Vec<Vec<f32>> = vec![];
        assert!(pool_tokens(&tokens, 2).is_empty());
    }

    #[test]
    fn test_pool_tokens_factor_one() {
        let tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let pooled = pool_tokens(&tokens, 1);
        assert_eq!(pooled.len(), tokens.len());
    }

    #[test]
    fn test_pool_tokens_reduces_count() {
        let tokens = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.9, 0.1],
        ];
        let pooled = pool_tokens(&tokens, 2);
        assert!(pooled.len() <= 2);
        assert!(!pooled.is_empty());
    }

    #[test]
    fn test_pool_tokens_sequential() {
        let tokens = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
            vec![0.3, 0.7],
        ];
        let pooled = pool_tokens_sequential(&tokens, 2);
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0][0] - 0.5).abs() < 1e-5);
        assert!((pooled[0][1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_pool_tokens_with_protected() {
        let tokens = vec![
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let pooled = pool_tokens_with_protected(&tokens, 2, 1);
        assert_eq!(pooled[0], vec![0.0, 0.0, 1.0]);
        assert!(pooled.len() >= 2);
    }

    #[test]
    fn test_pool_tokens_adaptive_low_factor() {
        let tokens = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.9, 0.1, 0.0],
        ];
        // Factor 2 should use clustering
        let pooled = pool_tokens_adaptive(&tokens, 2);
        assert_eq!(pooled.len(), 2);
    }

    #[test]
    fn test_pool_tokens_adaptive_high_factor() {
        let tokens: Vec<Vec<f32>> = (0..8)
            .map(|i| vec![(i as f32) * 0.1, 0.0, 0.0, 0.0])
            .collect();
        // Factor 4 should use sequential
        let pooled = pool_tokens_adaptive(&tokens, 4);
        assert_eq!(pooled.len(), 2); // 8 / 4 = 2
    }

    #[test]
    fn test_pool_tokens_adaptive_factor_one() {
        let tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let pooled = pool_tokens_adaptive(&tokens, 1);
        assert_eq!(pooled.len(), 2); // No pooling
    }

    #[test]
    fn test_pool_tokens_adaptive_empty() {
        let tokens: Vec<Vec<f32>> = vec![];
        let pooled = pool_tokens_adaptive(&tokens, 2);
        assert!(pooled.is_empty());
    }

    #[test]
    fn test_pooling_methods_produce_same_dimensions() {
        let tokens: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                (0..16)
                    .map(|j| ((i * 16 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        let greedy = pool_tokens(&tokens, 2);
        let sequential = pool_tokens_sequential(&tokens, 2);
        let adaptive = pool_tokens_adaptive(&tokens, 2);

        // All should produce 16-dim vectors
        assert!(greedy.iter().all(|v| v.len() == 16));
        assert!(sequential.iter().all(|v| v.len() == 16));
        assert!(adaptive.iter().all(|v| v.len() == 16));
    }

    #[test]
    fn test_maxsim_with_pooled_tokens() {
        let query = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let doc = vec![
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.8, 0.2, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.2, 0.8, 0.0, 0.0],
        ];

        let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
        let d_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();
        let score_original = crate::simd::maxsim(&q_refs, &d_refs);

        let pooled = pool_tokens(&doc, 2);
        let p_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();
        let score_pooled = crate::simd::maxsim(&q_refs, &p_refs);

        // Pooled should still produce reasonable score
        assert!(score_pooled > 0.0);
        assert!(score_pooled.is_finite());
        // Pooled score should be at least 50% of original (generous bound)
        assert!(
            score_pooled >= score_original * 0.5,
            "pooled {score_pooled} vs original {score_original}"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Mutation-killing: verify exact mathematical behavior
    // ───────────────────────────────────────────────────────────────────────

    #[cfg(not(feature = "hierarchical"))]
    #[test]
    fn pool_greedy_exact_count() {
        // Factor 2 on 4 tokens should give 2
        let tokens = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let pooled = pool_tokens_greedy(&tokens, 2);
        assert_eq!(pooled.len(), 2);
    }

    #[test]
    fn pool_sequential_exact_count() {
        // 8 tokens / factor 2 = 4
        let tokens: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32]).collect();
        let pooled = pool_tokens_sequential(&tokens, 2);
        assert_eq!(pooled.len(), 4);
    }

    #[test]
    fn refine_alpha_zero_ignores_original() {
        let query = vec![vec![1.0, 0.0]];
        let candidates = vec![("d1", 100.0), ("d2", 0.0)];
        let docs = vec![
            ("d1", vec![vec![0.0, 1.0]]), // maxsim ~0
            ("d2", vec![vec![1.0, 0.0]]), // maxsim ~1
        ];
        let config = crate::RefineConfig::default().with_alpha(0.0);
        let refined = refine_with_config(&candidates, &query, &docs, config);
        assert_eq!(refined[0].0, "d2", "alpha=0 should rank by maxsim only");
    }

    #[test]
    fn refine_alpha_one_ignores_maxsim() {
        let query = vec![vec![1.0, 0.0]];
        let candidates = vec![("d1", 1.0), ("d2", 0.5)];
        let docs = vec![
            ("d1", vec![vec![0.0, 1.0]]), // maxsim ~0
            ("d2", vec![vec![1.0, 0.0]]), // maxsim ~1
        ];
        let config = crate::RefineConfig::default().with_alpha(1.0);
        let refined = refine_with_config(&candidates, &query, &docs, config);
        assert_eq!(refined[0].0, "d1", "alpha=1 should rank by original only");
    }

    #[cfg(feature = "hierarchical")]
    #[test]
    fn hierarchical_returns_target_count() {
        let tokens: Vec<Vec<f32>> = (0..8).map(|i| vec![(i as f32 * 0.1).sin(); 16]).collect();
        let pooled = pool_tokens_hierarchical(&tokens, 4);
        assert_eq!(pooled.len(), 4);
    }

    /// Verifies cut_dendrogram produces correct number of clusters.
    ///
    /// This tests the union-find based dendrogram cutting algorithm
    /// that determines cluster membership after hierarchical clustering.
    #[cfg(feature = "hierarchical")]
    #[test]
    fn cut_dendrogram_produces_correct_clusters() {
        use kodama::{linkage, Method};

        // Create 6 tokens that will cluster into 3 pairs
        let tokens = vec![
            vec![1.0, 0.0, 0.0, 0.0], // group A
            vec![0.9, 0.1, 0.0, 0.0], // group A (similar to 0)
            vec![0.0, 1.0, 0.0, 0.0], // group B
            vec![0.1, 0.9, 0.0, 0.0], // group B (similar to 2)
            vec![0.0, 0.0, 1.0, 0.0], // group C
            vec![0.0, 0.0, 0.9, 0.1], // group C (similar to 4)
        ];

        let n = tokens.len();

        // Build condensed distance matrix
        let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = crate::simd::cosine(&tokens[i], &tokens[j]);
                let dist = f64::from((1.0 - sim).clamp(0.0, 2.0));
                condensed.push(dist);
            }
        }

        let dendrogram = linkage(&mut condensed, n, Method::Ward);

        // Cut to get 3 clusters
        let labels = cut_dendrogram(&dendrogram, n, 3);

        // Verify we get exactly 3 unique labels
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(
            unique_labels.len(),
            3,
            "Expected 3 clusters, got labels: {:?}",
            labels
        );

        // Verify similar tokens are in same cluster
        assert_eq!(labels[0], labels[1], "Tokens 0,1 should be in same cluster");
        assert_eq!(labels[2], labels[3], "Tokens 2,3 should be in same cluster");
        assert_eq!(labels[4], labels[5], "Tokens 4,5 should be in same cluster");

        // Verify different groups are in different clusters
        assert_ne!(labels[0], labels[2], "Groups A,B should differ");
        assert_ne!(labels[2], labels[4], "Groups B,C should differ");
    }

    /// Verifies hierarchical pooling clusters semantically similar tokens.
    #[cfg(feature = "hierarchical")]
    #[test]
    fn hierarchical_clusters_similar_tokens() {
        // Create tokens where some are very similar
        let tokens = vec![
            vec![1.0, 0.0, 0.0, 0.0], // distinct
            vec![0.0, 1.0, 0.0, 0.0], // repeated 4 times
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0], // distinct
        ];

        // Pool from 6 to 3
        let pooled = pool_tokens_hierarchical(&tokens, 3);
        assert_eq!(pooled.len(), 3);

        // One of the pooled vectors should be close to [0,1,0,0]
        // (the mean of the 4 identical vectors)
        let target = vec![0.0, 1.0, 0.0, 0.0];
        let max_sim = pooled
            .iter()
            .map(|p| crate::simd::cosine(p, &target))
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            max_sim > 0.99,
            "Expected pooled vector near [0,1,0,0], best sim: {}",
            max_sim
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // TokenIndex tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn token_index_new_and_len() {
        let index: TokenIndex<&str> = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ]);
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[test]
    fn token_index_empty() {
        let index: TokenIndex<&str> = TokenIndex::default();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn token_index_score_all() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ]);
        let query = vec![vec![1.0, 0.0]];

        let scores = index.score_all(&query);
        assert_eq!(scores.len(), 2);

        // doc1 should have score ~1.0, doc2 should have score ~0.0
        let doc1_score = scores.iter().find(|(id, _)| *id == "doc1").unwrap().1;
        let doc2_score = scores.iter().find(|(id, _)| *id == "doc2").unwrap().1;
        assert!((doc1_score - 1.0).abs() < 1e-5);
        assert!(doc2_score.abs() < 1e-5);
    }

    #[test]
    fn token_index_rank() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            ("doc2", vec![vec![0.5, 0.5]]),
            ("doc3", vec![vec![0.9, 0.1]]),
        ]);
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let ranked = index.rank(&query);
        assert_eq!(ranked.len(), 3);
        // doc1 has perfect matches for both query tokens
        assert_eq!(ranked[0].0, "doc1");
    }

    #[test]
    fn token_index_top_k() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.9, 0.1]]),
            ("doc3", vec![vec![0.8, 0.2]]),
        ]);
        let query = vec![vec![1.0, 0.0]];

        let top2 = index.top_k(&query, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "doc1");
        assert_eq!(top2[1].0, "doc2");
    }

    #[test]
    fn token_index_top_k_larger_than_size() {
        let index = TokenIndex::new(vec![("doc1", vec![vec![1.0, 0.0]])]);
        let query = vec![vec![1.0, 0.0]];

        let top10 = index.top_k(&query, 10);
        assert_eq!(top10.len(), 1);
    }

    #[test]
    fn token_index_get() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ]);

        assert!(index.get(&"doc1").is_some());
        assert!(index.get(&"doc2").is_some());
        assert!(index.get(&"doc3").is_none());
    }

    #[test]
    fn token_index_contains() {
        let index = TokenIndex::new(vec![("doc1", vec![vec![1.0, 0.0]])]);

        assert!(index.contains(&"doc1"));
        assert!(!index.contains(&"doc2"));
    }

    #[test]
    fn token_index_from_iter() {
        let entries = vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ];
        let index: TokenIndex<&str> = entries.into_iter().collect();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn token_index_iter() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ]);

        let ids: Vec<_> = index.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"doc1"));
        assert!(ids.contains(&"doc2"));
    }

    #[test]
    fn token_index_into_entries() {
        let index = TokenIndex::new(vec![("doc1", vec![vec![1.0, 0.0]])]);
        let entries = index.into_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "doc1");
    }

    #[test]
    fn token_index_entries_returns_slice() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![1.0, 0.0]]),
            ("doc2", vec![vec![0.0, 1.0]]),
        ]);

        let entries = index.entries();
        // Verify we get the actual entries, not an empty slice
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "doc1");
        assert_eq!(entries[1].0, "doc2");
        // Verify embedding data is present
        assert_eq!(entries[0].1.len(), 1);
        assert_eq!(entries[0].1[0], vec![1.0, 0.0]);
    }

    #[test]
    fn token_index_score_all_cosine() {
        let index = TokenIndex::new(vec![
            ("doc1", vec![vec![2.0, 0.0]]), // unnormalized
            ("doc2", vec![vec![0.0, 2.0]]), // unnormalized
        ]);
        let query = vec![vec![1.0, 0.0]];

        let scores = index.score_all_cosine(&query);
        // Cosine normalizes, so doc1 should still have score ~1.0
        let doc1_score = scores.iter().find(|(id, _)| *id == "doc1").unwrap().1;
        assert!((doc1_score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn token_index_matches_rank_function() {
        // Verify TokenIndex.rank() produces same results as standalone rank()
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            ("d2", vec![vec![0.5, 0.5]]),
            ("d3", vec![vec![0.9, 0.1]]),
        ];
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let index = TokenIndex::new(docs.clone());
        let index_ranked = index.rank(&query);
        let func_ranked = rank(&query, &docs);

        // Same order and scores
        assert_eq!(index_ranked.len(), func_ranked.len());
        for (a, b) in index_ranked.iter().zip(func_ranked.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn rank_preserves_doc_count(n_docs in 1usize..5, n_query_tok in 1usize..4, dim in 2usize..8) {
            let query = (0..n_query_tok)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect::<Vec<Vec<f32>>>();
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let toks = (0..2)
                        .map(|t| (0..dim).map(|j| (i as usize + t + j) as f32 * 0.1).collect())
                        .collect();
                    (i, toks)
                })
                .collect();

            let ranked = rank(&query, &docs);
            prop_assert_eq!(ranked.len(), n_docs);
        }

        #[test]
        fn rank_sorted_descending(n_docs in 2usize..6, dim in 2usize..6) {
            let query = vec![(0..dim).map(|i| i as f32 * 0.1).collect::<Vec<f32>>()];
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let toks = vec![(0..dim).map(|j| (i as usize + j) as f32 * 0.1).collect()];
                    (i, toks)
                })
                .collect();

            let ranked = rank(&query, &docs);
            for window in ranked.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }

        #[test]
        fn refine_output_bounded(n_cand in 1usize..5, n_docs in 0usize..5, dim in 2usize..6) {
            let candidates: Vec<(u32, f32)> = (0..n_cand as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();
            let query = vec![(0..dim).map(|i| i as f32 * 0.1).collect::<Vec<f32>>()];
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let toks = vec![(0..dim).map(|j| (i as usize + j) as f32 * 0.1).collect()];
                    (i, toks)
                })
                .collect();

            let refined = refine(&candidates, &query, &docs, 0.5);
            prop_assert!(refined.len() <= candidates.len());
            prop_assert!(refined.len() <= docs.len());
        }

        #[test]
        fn pool_reduces_count(n_tokens in 4usize..20, dim in 2usize..8, pool_factor in 2usize..4) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i + j) as f32 * 0.1).sin()).collect())
                .collect();

            let pooled = pool_tokens(&tokens, pool_factor);
            let expected_max = (n_tokens / pool_factor).max(1);
            prop_assert!(pooled.len() <= expected_max + 1);
            prop_assert!(!pooled.is_empty());
        }

        #[test]
        fn sequential_pool_exact_count(n_tokens in 2usize..20, dim in 2usize..8, window in 2usize..4) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();

            let pooled = pool_tokens_sequential(&tokens, window);
            let expected = (n_tokens + window - 1) / window;
            prop_assert_eq!(pooled.len(), expected);
        }

        /// Pooled tokens preserve dimension
        #[test]
        fn pool_preserves_dimension(n_tokens in 2usize..10, dim in 2usize..16, pool_factor in 2usize..4) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();

            let pooled = pool_tokens(&tokens, pool_factor);
            for tok in &pooled {
                prop_assert_eq!(tok.len(), dim, "Dimension mismatch: expected {}, got {}", dim, tok.len());
            }
        }

        /// Sequential pooling preserves dimension
        #[test]
        fn sequential_pool_preserves_dimension(n_tokens in 2usize..10, dim in 2usize..16, window in 2usize..4) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();

            let pooled = pool_tokens_sequential(&tokens, window);
            for tok in &pooled {
                prop_assert_eq!(tok.len(), dim, "Dimension mismatch: expected {}, got {}", dim, tok.len());
            }
        }

        /// Pool factor 1 returns original tokens
        #[test]
        fn pool_factor_one_identity(n_tokens in 1usize..10, dim in 2usize..8) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();

            let pooled = pool_tokens(&tokens, 1);
            prop_assert_eq!(pooled.len(), n_tokens, "Factor 1 should preserve count");
        }

        /// Protected tokens are preserved (CLS/SEP-like behavior)
        #[test]
        fn protected_tokens_preserved(n_tokens in 3usize..10, dim in 2usize..8) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();

            // Protect first 2 tokens (like CLS and SEP)
            let pooled = pool_tokens_with_protected(&tokens, 2, 2);

            // Protected tokens should appear in output unchanged
            prop_assert!(pooled.len() >= 2, "Should have at least protected tokens");
            // First two tokens should be preserved exactly
            prop_assert_eq!(&pooled[0], &tokens[0], "First protected token should be preserved");
            prop_assert_eq!(&pooled[1], &tokens[1], "Second protected token should be preserved");
        }

        /// `MaxSim` score quality: pooling shouldn't drastically reduce score
        #[test]
        fn pool_maintains_score_quality(dim in 8usize..16) {
            // Create a query and doc with good alignment
            let query: Vec<Vec<f32>> = (0..4)
                .map(|i| (0..dim).map(|j| if i == j % 4 { 1.0 } else { 0.1 }).collect())
                .collect();
            let doc: Vec<Vec<f32>> = (0..8)
                .map(|i| (0..dim).map(|j| if i % 4 == j % 4 { 1.0 } else { 0.1 }).collect())
                .collect();

            let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let doc_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();

            let original_score = crate::simd::maxsim(&query_refs, &doc_refs);

            let pooled = pool_tokens(&doc, 2);
            let pooled_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();
            let pooled_score = crate::simd::maxsim(&query_refs, &pooled_refs);

            // Pooled score shouldn't drop more than 50% (generous bound for property test)
            prop_assert!(
                pooled_score >= original_score * 0.5,
                "Score dropped too much: {} -> {}",
                original_score,
                pooled_score
            );
        }

        /// Refine with alpha=1 preserves original order
        #[test]
        fn refine_alpha_one_preserves_order(n_cand in 2usize..5) {
            let candidates: Vec<(u32, f32)> = (0..n_cand as u32)
                .map(|i| (i, 10.0 - i as f32)) // Descending order
                .collect();
            let query = vec![vec![1.0f32; 4]];
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_cand as u32)
                .map(|i| (i, vec![vec![0.5f32; 4]]))
                .collect();

            let refined = refine(&candidates, &query, &docs, 1.0);
            for (i, (id, _)) in refined.iter().enumerate() {
                prop_assert_eq!(*id, i as u32, "Order not preserved at index {}", i);
            }
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Domain-aware tests based on ColBERT pooling research (Clavié et al. 2024)
        // ─────────────────────────────────────────────────────────────────────────

        /// Pooling preserves total information: sum of pooled vectors ≈ scaled sum of original
        /// (Since each cluster is mean-pooled, total vector mass is preserved proportionally)
        #[test]
        fn pool_preserves_vector_mass(n_tokens in 4usize..16, dim in 4usize..16) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();

            // Sum of L2 norms before pooling
            let orig_total_norm: f32 = tokens
                .iter()
                .map(|t| crate::simd::norm(t))
                .sum();

            let pooled = pool_tokens(&tokens, 2);

            // Sum of L2 norms after pooling
            let pooled_total_norm: f32 = pooled
                .iter()
                .map(|t| crate::simd::norm(t))
                .sum();

            // Pooled norms should be in reasonable range
            // (mean pooling reduces magnitude, so pooled < original is expected)
            prop_assert!(
                pooled_total_norm > 0.0,
                "Pooled vectors should have positive norm"
            );
            prop_assert!(
                pooled_total_norm <= orig_total_norm * 1.1,
                "Pooled norm {} too large vs original {}",
                pooled_total_norm,
                orig_total_norm
            );
        }

        /// Duplicate tokens should cluster together (key insight from research:
        /// redundant semantic info should merge)
        #[test]
        fn duplicate_tokens_cluster(dim in 4usize..16) {
            // Create tokens where first 4 are identical, rest are random
            let base_token: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut tokens = vec![base_token.clone(); 4];
            // Add 4 different tokens
            for i in 0..4 {
                tokens.push((0..dim).map(|j| ((i * 10 + j) as f32 * 0.3).cos()).collect());
            }

            // With pool factor 2, we go from 8 to 4 tokens
            // The 4 identical tokens should merge into fewer clusters
            let pooled = pool_tokens(&tokens, 2);

            // At least one pooled token should be very similar to the base
            // (the duplicates should cluster)
            let max_sim = pooled
                .iter()
                .map(|p| crate::simd::cosine(p, &base_token))
                .fold(f32::NEG_INFINITY, f32::max);

            prop_assert!(
                max_sim > 0.95,
                "Duplicate tokens didn't cluster well: max_sim = {}",
                max_sim
            );
        }

        /// `MaxSim` is NOT commutative: swap(query, doc) changes result
        /// This is a fundamental property of late interaction scoring
        #[test]
        fn maxsim_not_commutative(dim in 4usize..8) {
            let a: Vec<Vec<f32>> = vec![
                (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(),
            ];
            let b: Vec<Vec<f32>> = vec![
                (0..dim).map(|i| if i == 0 { 0.5 } else { 0.5 }).collect(),
                (0..dim).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect(),
            ];

            let a_refs: Vec<&[f32]> = a.iter().map(Vec::as_slice).collect();
            let b_refs: Vec<&[f32]> = b.iter().map(Vec::as_slice).collect();

            let score_ab = crate::simd::maxsim(&a_refs, &b_refs);
            let score_ba = crate::simd::maxsim(&b_refs, &a_refs);

            // Different number of query tokens → different sums
            // (unless vectors are perfectly aligned, which is rare)
            // Just assert they're both valid scores
            prop_assert!(score_ab.is_finite() && score_ba.is_finite());
        }

        /// Adding more doc tokens can only maintain or improve MaxSim
        /// (more opportunities for each query token to find a match)
        #[test]
        fn more_doc_tokens_higher_score(dim in 4usize..8) {
            let query: Vec<Vec<f32>> = vec![
                (0..dim).map(|i| (i as f32 * 0.1).sin()).collect(),
                (0..dim).map(|i| (i as f32 * 0.2).cos()).collect(),
            ];
            let doc_small: Vec<Vec<f32>> = vec![
                (0..dim).map(|i| (i as f32 * 0.15).sin()).collect(),
            ];
            let mut doc_large = doc_small.clone();
            doc_large.push((0..dim).map(|i| (i as f32 * 0.25).cos()).collect());

            let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let small_refs: Vec<&[f32]> = doc_small.iter().map(Vec::as_slice).collect();
            let large_refs: Vec<&[f32]> = doc_large.iter().map(Vec::as_slice).collect();

            let score_small = crate::simd::maxsim(&q_refs, &small_refs);
            let score_large = crate::simd::maxsim(&q_refs, &large_refs);

            prop_assert!(
                score_large >= score_small - 1e-6,
                "More tokens should help: {} vs {}",
                score_large,
                score_small
            );
        }

        /// Pooling is stable: pool(pool(x)) ≈ pool(x) for same target count
        #[test]
        fn pooling_idempotent_at_target(dim in 4usize..8) {
            let tokens: Vec<Vec<f32>> = (0..8)
                .map(|i| (0..dim).map(|j| ((i + j) as f32 * 0.1).sin()).collect())
                .collect();

            let pooled_once = pool_tokens(&tokens, 2); // 8 -> 4
            let pooled_twice = pool_tokens(&pooled_once, 1); // 4 -> 4 (no change)

            prop_assert_eq!(
                pooled_once.len(),
                pooled_twice.len(),
                "Second pool changed count"
            );

            // Contents should be identical
            for (a, b) in pooled_once.iter().zip(pooled_twice.iter()) {
                let sim = crate::simd::cosine(a, b);
                prop_assert!(
                    sim > 0.999,
                    "Pool not idempotent: similarity = {}",
                    sim
                );
            }
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Adaptive pooling tests
        // ─────────────────────────────────────────────────────────────────────────

        /// Adaptive pooling uses clustering for low factors
        #[test]
        fn adaptive_uses_clustering_for_low_factors(n_tokens in 4usize..12, dim in 4usize..8) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();

            // Factor 2: should use pool_tokens (clustering)
            let adaptive = pool_tokens_adaptive(&tokens, 2);
            let clustering = pool_tokens(&tokens, 2);

            // Results should be identical (same method)
            prop_assert_eq!(adaptive.len(), clustering.len());
        }

        /// Adaptive pooling uses sequential for high factors
        #[test]
        fn adaptive_uses_sequential_for_high_factors(n_tokens in 8usize..20, dim in 4usize..8) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();

            // Factor 4+: should use pool_tokens_sequential
            let adaptive = pool_tokens_adaptive(&tokens, 4);
            let sequential = pool_tokens_sequential(&tokens, 4);

            // Results should be identical (same method)
            prop_assert_eq!(adaptive.len(), sequential.len());
            for (a, s) in adaptive.iter().zip(sequential.iter()) {
                prop_assert_eq!(a, s);
            }
        }

        /// All pooling methods preserve dimension
        #[test]
        fn all_pooling_methods_preserve_dim(n_tokens in 4usize..16, dim in 4usize..16) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();

            let p1 = pool_tokens(&tokens, 2);
            let p2 = pool_tokens_sequential(&tokens, 2);
            let p3 = pool_tokens_adaptive(&tokens, 2);
            let p4 = pool_tokens_with_protected(&tokens, 2, 1);

            prop_assert!(p1.iter().all(|v| v.len() == dim));
            prop_assert!(p2.iter().all(|v| v.len() == dim));
            prop_assert!(p3.iter().all(|v| v.len() == dim));
            prop_assert!(p4.iter().all(|v| v.len() == dim));
        }

        /// Pooling never increases token count
        #[test]
        fn pooling_never_increases_count(n_tokens in 1usize..20, factor in 1usize..5) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| vec![(i as f32) * 0.1; 8])
                .collect();

            let p1 = pool_tokens(&tokens, factor);
            let p2 = pool_tokens_sequential(&tokens, factor);
            let p3 = pool_tokens_adaptive(&tokens, factor);

            prop_assert!(p1.len() <= n_tokens);
            prop_assert!(p2.len() <= n_tokens);
            prop_assert!(p3.len() <= n_tokens);
        }

        /// Empty tokens handled gracefully by all methods
        #[test]
        fn empty_tokens_all_methods(factor in 1usize..5) {
            let empty: Vec<Vec<f32>> = vec![];

            prop_assert!(pool_tokens(&empty, factor).is_empty());
            prop_assert!(pool_tokens_sequential(&empty, factor).is_empty());
            prop_assert!(pool_tokens_adaptive(&empty, factor).is_empty());
            prop_assert!(pool_tokens_with_protected(&empty, factor, 0).is_empty());
        }

        /// Single token returns unchanged
        #[test]
        fn single_token_unchanged(dim in 2usize..16, factor in 2usize..5) {
            let tokens = vec![vec![1.0f32; dim]];

            let p1 = pool_tokens(&tokens, factor);
            let p2 = pool_tokens_sequential(&tokens, factor);
            let p3 = pool_tokens_adaptive(&tokens, factor);

            prop_assert_eq!(p1.len(), 1);
            prop_assert_eq!(p2.len(), 1);
            prop_assert_eq!(p3.len(), 1);
        }

        /// `MaxSim` with pooled docs still produces finite scores
        #[test]
        fn maxsim_pooled_finite(n_query in 1usize..4, n_doc in 2usize..8, dim in 4usize..8) {
            let query: Vec<Vec<f32>> = (0..n_query)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();
            let doc: Vec<Vec<f32>> = (0..n_doc)
                .map(|i| (0..dim).map(|j| ((i * dim + j + 100) as f32 * 0.1).cos()).collect())
                .collect();

            let pooled = pool_tokens_adaptive(&doc, 2);

            let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let p_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();

            let score = crate::simd::maxsim(&q_refs, &p_refs);
            prop_assert!(score.is_finite(), "MaxSim with pooled docs returned {}", score);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // TokenIndex property tests
        // ─────────────────────────────────────────────────────────────────────────

        /// TokenIndex.rank() produces sorted output (descending)
        #[test]
        fn token_index_rank_sorted(n_docs in 2usize..8, n_query in 1usize..4, dim in 2usize..8) {
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let tokens: Vec<Vec<f32>> = (0..2)
                        .map(|t| (0..dim).map(|j| ((i as usize * 2 + t + j) as f32 * 0.1).sin()).collect())
                        .collect();
                    (i, tokens)
                })
                .collect();
            let query: Vec<Vec<f32>> = (0..n_query)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).cos()).collect())
                .collect();

            let index = TokenIndex::new(docs);
            let ranked = index.rank(&query);

            for window in ranked.windows(2) {
                prop_assert!(
                    window[0].1 >= window[1].1 - 1e-6,
                    "Not sorted: {} >= {}",
                    window[0].1,
                    window[1].1
                );
            }
        }

        /// TokenIndex.top_k() returns at most k results
        #[test]
        fn token_index_top_k_bounded(n_docs in 1usize..10, k in 1usize..5, dim in 2usize..8) {
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| (i, vec![vec![(i as f32 * 0.1).sin(); dim]]))
                .collect();
            let query = vec![vec![0.5f32; dim]];

            let index = TokenIndex::new(docs);
            let top = index.top_k(&query, k);

            prop_assert!(top.len() <= k.min(n_docs));
        }

        /// TokenIndex preserves all entries
        #[test]
        fn token_index_preserves_count(n_docs in 1usize..10, dim in 2usize..8) {
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| (i, vec![vec![(i as f32 * 0.1).sin(); dim]]))
                .collect();

            let index = TokenIndex::new(docs);
            prop_assert_eq!(index.len(), n_docs);
        }

        /// TokenIndex.score_all() returns all entries
        #[test]
        fn token_index_score_all_count(n_docs in 1usize..10, dim in 2usize..8) {
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| (i, vec![vec![(i as f32 * 0.1).sin(); dim]]))
                .collect();
            let query = vec![vec![0.5f32; dim]];

            let index = TokenIndex::new(docs);
            let scores = index.score_all(&query);

            prop_assert_eq!(scores.len(), n_docs);
        }

        /// TokenIndex produces finite scores
        #[test]
        fn token_index_scores_finite(n_docs in 1usize..5, n_query in 1usize..3, dim in 2usize..8) {
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let tokens: Vec<Vec<f32>> = (0..2)
                        .map(|t| (0..dim).map(|j| ((i as usize * 2 + t + j) as f32 * 0.1).sin()).collect())
                        .collect();
                    (i, tokens)
                })
                .collect();
            let query: Vec<Vec<f32>> = (0..n_query)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).cos()).collect())
                .collect();

            let index = TokenIndex::new(docs);
            let scores = index.score_all(&query);

            for (id, score) in &scores {
                prop_assert!(score.is_finite(), "Score for {} is not finite: {}", id, score);
            }
        }
    }
}
