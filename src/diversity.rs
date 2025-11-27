//! Diversity-aware reranking algorithms.
//!
//! Select results that balance relevance with diversity — maximizing coverage
//! while minimizing redundancy.
//!
//! # Algorithms
//!
//! | Function | Best For | Complexity |
//! |----------|----------|------------|
//! | [`mmr`] | General use, simple | O(k × n) |
//! | [`mmr_cosine`] | Raw embeddings | O(k × n × d) |
//! | [`dpp`] | Better theoretical guarantees | O(k × n × d) |
//!
//! ## MMR vs DPP
//!
//! - **MMR**: Penalizes max similarity to already-selected items. Simple, fast.
//! - **DPP**: Models joint diversity via determinants. Better for small k,
//!   captures pairwise diversity more holistically.
//!
//! # Lambda Parameter Guide
//!
//! | Value | Use Case |
//! |-------|----------|
//! | 0.3–0.5 | Exploratory search, discovery |
//! | 0.5 | Balanced default (RAG systems) |
//! | 0.7–0.9 | Precision search, specific intent |
//!
//! Research (VRSD, 2024) shows λ=0.5 is a reasonable default, but
//! optimal value depends on candidate distribution in embedding space.
//!
//! # The Diversity Problem
//!
//! Given top-100 from retrieval, many results may be near-duplicates.
//! Users want variety: different perspectives, aspects, or subtopics.
//!
//! ```text
//! Before MMR (λ=1.0):       After MMR (λ=0.5):
//! 1. Python async/await     1. Python async/await
//! 2. Python asyncio guide   2. Rust async/await
//! 3. Python async tutorial  3. JavaScript promises
//! 4. Python coroutines      4. Go goroutines
//! 5. Understanding asyncio  5. Python asyncio guide
//! ```
//!
//! # Example
//!
//! ```rust
//! use rank_refine::diversity::{mmr, MmrConfig};
//!
//! // Candidates with precomputed relevance scores
//! let candidates: Vec<(&str, f32)> = vec![
//!     ("doc1", 0.95),
//!     ("doc2", 0.90),
//!     ("doc3", 0.85),
//! ];
//!
//! // Pairwise similarity matrix (flattened, row-major)
//! // similarity[i * n + j] = sim(candidates[i], candidates[j])
//! let similarity = vec![
//!     1.0, 0.9, 0.2,  // doc1 vs [doc1, doc2, doc3]
//!     0.9, 1.0, 0.3,  // doc2 vs [doc1, doc2, doc3]
//!     0.2, 0.3, 1.0,  // doc3 vs [doc1, doc2, doc3]
//! ];
//!
//! let config = MmrConfig::default().with_lambda(0.5).with_k(2);
//! let selected = mmr(&candidates, &similarity, config);
//!
//! // doc1 selected first (highest relevance)
//! // doc3 selected second (diverse from doc1, even though doc2 is more relevant)
//! assert_eq!(selected[0].0, "doc1");
//! assert_eq!(selected[1].0, "doc3");
//! ```

use crate::simd;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Maximal Marginal Relevance (MMR).
#[derive(Debug, Clone, Copy)]
pub struct MmrConfig {
    /// Trade-off between relevance and diversity.
    /// - `λ=1.0`: pure relevance (no diversity)
    /// - `λ=0.0`: pure diversity (maximize distance from selected)
    /// - `λ=0.5`: balanced (common default)
    pub lambda: f32,
    /// Number of results to select.
    pub k: usize,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self { lambda: 0.5, k: 10 }
    }
}

impl MmrConfig {
    /// Create config with custom lambda and k. Lambda clamped to \[0, 1\].
    #[must_use]
    pub fn new(lambda: f32, k: usize) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
            k,
        }
    }

    /// Set lambda (relevance-diversity tradeoff). Clamped to \[0, 1\].
    ///
    /// - `0.0` = maximum diversity (ignore relevance)
    /// - `0.5` = balanced (default)
    /// - `1.0` = pure relevance (ignore diversity)
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Set k (number of results to select).
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MMR with precomputed similarity matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Maximal Marginal Relevance with precomputed similarity.
///
/// Iteratively selects documents that maximize:
/// `λ * relevance(d) - (1-λ) * max_{s ∈ selected} similarity(d, s)`
///
/// # Arguments
///
/// * `candidates` - `(id, relevance_score)` pairs
/// * `similarity` - Flattened row-major similarity matrix (n×n)
/// * `config` - MMR configuration (lambda, k)
///
/// # Returns
///
/// Selected documents in MMR order (most relevant diverse first).
///
/// # Complexity
///
/// O(k × n) where k = `config.k` and n = `candidates.len()`.
///
/// # Panics
///
/// Panics if `similarity.len() != candidates.len()²`.
#[must_use]
pub fn mmr<I: Clone>(
    candidates: &[(I, f32)],
    similarity: &[f32],
    config: MmrConfig,
) -> Vec<(I, f32)> {
    try_mmr(candidates, similarity, config).expect("similarity matrix must be n×n")
}

/// Fallible version of [`mmr`]. Returns `Err` if similarity matrix is wrong size.
///
/// # Errors
///
/// Returns [`RefineError::DimensionMismatch`](crate::RefineError::DimensionMismatch)
/// if `similarity.len() != candidates.len()²`.
pub fn try_mmr<I: Clone>(
    candidates: &[(I, f32)],
    similarity: &[f32],
    config: MmrConfig,
) -> Result<Vec<(I, f32)>, crate::RefineError> {
    let n = candidates.len();
    if similarity.len() != n * n {
        return Err(crate::RefineError::DimensionMismatch {
            expected: n * n,
            got: similarity.len(),
        });
    }

    if n == 0 || config.k == 0 {
        return Ok(Vec::new());
    }

    // Normalize relevance scores to [0, 1] for fair comparison with similarity
    let (rel_min, rel_max) = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        });
    let rel_range = rel_max - rel_min;
    let rel_norm: Vec<f32> = if rel_range > 1e-9 {
        candidates
            .iter()
            .map(|(_, s)| (s - rel_min) / rel_range)
            .collect()
    } else {
        vec![1.0; n] // All equal relevance
    };

    let mut selected_indices: Vec<usize> = Vec::with_capacity(config.k.min(n));
    let mut remaining: Vec<usize> = (0..n).collect();

    for _ in 0..config.k.min(n) {
        if remaining.is_empty() {
            break;
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (remaining_pos, &cand_idx) in remaining.iter().enumerate() {
            let relevance = rel_norm[cand_idx];

            // Max similarity to any already-selected document
            let max_sim = if selected_indices.is_empty() {
                0.0
            } else {
                selected_indices
                    .iter()
                    .map(|&sel_idx| similarity[cand_idx * n + sel_idx])
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            // MMR score: λ * relevance - (1-λ) * max_similarity
            let mmr_score = config.lambda * relevance - (1.0 - config.lambda) * max_sim;

            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = remaining_pos;
            }
        }

        let chosen = remaining.swap_remove(best_idx);
        selected_indices.push(chosen);
    }

    // Return in selection order with original scores
    Ok(selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// MMR with embeddings (computes similarity on-the-fly)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximal Marginal Relevance with cosine similarity computed from embeddings.
///
/// More convenient than [`mmr`] when you have embeddings but haven't
/// precomputed the similarity matrix. Less efficient for repeated calls
/// on the same candidate set.
///
/// # Arguments
///
/// * `candidates` - `(id, relevance_score)` pairs
/// * `embeddings` - Embeddings for each candidate (same order)
/// * `config` - MMR configuration
///
/// # Panics
///
/// Panics if `embeddings.len() != candidates.len()`.
///
/// # Example
///
/// ```rust
/// use rank_refine::diversity::{mmr_cosine, MmrConfig};
///
/// let candidates = vec![("doc1", 0.9), ("doc2", 0.85), ("doc3", 0.8)];
/// let embeddings: Vec<Vec<f32>> = vec![
///     vec![1.0, 0.0],
///     vec![0.9, 0.1],  // Similar to doc1
///     vec![0.0, 1.0],  // Different from doc1
/// ];
///
/// let config = MmrConfig::default().with_lambda(0.5).with_k(2);
/// let selected = mmr_cosine(&candidates, &embeddings, config);
///
/// // doc1 first (highest relevance), doc3 second (diverse)
/// assert_eq!(selected[0].0, "doc1");
/// assert_eq!(selected[1].0, "doc3");
/// ```
#[must_use]
pub fn mmr_cosine<I: Clone, V: AsRef<[f32]>>(
    candidates: &[(I, f32)],
    embeddings: &[V],
    config: MmrConfig,
) -> Vec<(I, f32)> {
    let n = candidates.len();
    assert_eq!(
        embeddings.len(),
        n,
        "embeddings must have same length as candidates"
    );

    if n == 0 || config.k == 0 {
        return Vec::new();
    }

    // Normalize relevance scores to [0, 1]
    let (rel_min, rel_max) = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        });
    let rel_range = rel_max - rel_min;
    let rel_norm: Vec<f32> = if rel_range > 1e-9 {
        candidates
            .iter()
            .map(|(_, s)| (s - rel_min) / rel_range)
            .collect()
    } else {
        vec![1.0; n]
    };

    let mut selected_indices: Vec<usize> = Vec::with_capacity(config.k.min(n));
    let mut remaining: Vec<usize> = (0..n).collect();

    for _ in 0..config.k.min(n) {
        if remaining.is_empty() {
            break;
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (remaining_pos, &cand_idx) in remaining.iter().enumerate() {
            let relevance = rel_norm[cand_idx];
            let cand_emb = embeddings[cand_idx].as_ref();

            // Compute max similarity to selected documents on-the-fly
            let max_sim = if selected_indices.is_empty() {
                0.0
            } else {
                selected_indices
                    .iter()
                    .map(|&sel_idx| simd::cosine(cand_emb, embeddings[sel_idx].as_ref()))
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            let mmr_score = config.lambda * relevance - (1.0 - config.lambda) * max_sim;

            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = remaining_pos;
            }
        }

        let chosen = remaining.swap_remove(best_idx);
        selected_indices.push(chosen);
    }

    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast Greedy DPP
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Determinantal Point Process (DPP) diversity selection.
///
/// # Alpha Parameter
///
/// `alpha` controls the relevance-diversity trade-off by scaling relevance scores
/// before they're combined with diversity:
///
/// | Alpha | Effect |
/// |-------|--------|
/// | 0.0 | Pure diversity (ignore relevance) |
/// | 1.0 | Balanced (default) |
/// | 2.0+ | Strong relevance preference |
///
/// Internally, relevance is transformed as `exp(relevance × alpha)`, so:
/// - Higher alpha amplifies score differences
/// - At `alpha=0`, all items have equal quality weight
#[derive(Debug, Clone, Copy)]
pub struct DppConfig {
    /// Number of results to select.
    pub k: usize,
    /// Relevance weight (higher = more weight on relevance vs diversity).
    /// Default: 1.0. Range: \[0.0, ∞).
    pub alpha: f32,
}

impl Default for DppConfig {
    fn default() -> Self {
        Self { k: 10, alpha: 1.0 }
    }
}

impl DppConfig {
    /// Create config with custom k and alpha.
    #[must_use]
    pub fn new(k: usize, alpha: f32) -> Self {
        Self {
            k,
            alpha: alpha.max(0.0),
        }
    }

    /// Set k (number of results to select).
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set alpha (relevance weight). Higher = more relevance, less diversity.
    ///
    /// # Note
    ///
    /// Quality scores are computed as `exp(relevance * alpha)`. Large values
    /// of `relevance * alpha` (> ~88) will overflow to infinity. For relevance
    /// scores in [0, 1], alpha up to ~80 is safe.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }
}

/// Fast Greedy DPP diversity selection.
///
/// Uses the Fast Greedy MAP algorithm to select diverse items.
/// DPP models diversity through the determinant of a kernel matrix,
/// favoring sets where items are dissimilar.
///
/// # Algorithm
///
/// Unlike MMR which uses a simple max-similarity penalty, DPP considers
/// the **joint** diversity of the selected set via orthogonality in
/// embedding space.
///
/// # Arguments
///
/// * `candidates` - `(id, relevance_score)` pairs
/// * `embeddings` - Embedding vectors (one per candidate)
/// * `config` - DPP configuration
///
/// # Returns
///
/// Selected documents in DPP order.
///
/// # Panics
///
/// Panics if `embeddings.len() != candidates.len()`.
///
/// # Complexity
///
/// O(k × n × d) where k = selected, n = candidates, d = embedding dim.
///
/// # Example
///
/// ```rust
/// use rank_refine::diversity::{dpp, DppConfig};
///
/// let candidates: Vec<(&str, f32)> = vec![
///     ("doc1", 0.95),
///     ("doc2", 0.90),
///     ("doc3", 0.85),
/// ];
/// let embeddings: Vec<Vec<f32>> = vec![
///     vec![1.0, 0.0, 0.0],
///     vec![0.99, 0.1, 0.0], // Very similar to doc1
///     vec![0.0, 0.0, 1.0],  // Orthogonal to doc1
/// ];
///
/// let config = DppConfig::default().with_k(2);
/// let selected = dpp(&candidates, &embeddings, config);
///
/// // DPP prefers orthogonal items
/// assert_eq!(selected.len(), 2);
/// ```
///
/// # References
///
/// - [Fast Greedy MAP Inference for DPP](https://papers.nips.cc/paper/7805-fast-greedy-map-inference-for-determinantal-point-process-to-improve-recommendation-diversity.pdf)
/// - [DPP for ML](https://arxiv.org/abs/1207.6083)
#[must_use]
pub fn dpp<I: Clone, V: AsRef<[f32]>>(
    candidates: &[(I, f32)],
    embeddings: &[V],
    config: DppConfig,
) -> Vec<(I, f32)> {
    let n = candidates.len();
    assert_eq!(
        embeddings.len(),
        n,
        "embeddings must have same length as candidates"
    );

    if n == 0 || config.k == 0 {
        return Vec::new();
    }

    // Build quality scores (relevance scaled by alpha)
    let qualities: Vec<f32> = candidates
        .iter()
        .map(|(_, r)| (r * config.alpha).exp())
        .collect();

    // Fast Greedy DPP: iteratively select item that maximizes log-det gain
    let mut selected_indices: Vec<usize> = Vec::with_capacity(config.k.min(n));
    let mut remaining: Vec<usize> = (0..n).collect();

    // Track orthogonal components for efficient updates
    // c[i] = ||v_i - proj_{selected}(v_i)||^2 (starts as ||v_i||^2)
    let mut c: Vec<f32> = embeddings
        .iter()
        .map(|e| {
            let v = e.as_ref();
            simd::dot(v, v)
        })
        .collect();

    // d[i][j] stores dot products needed for incremental updates
    // We compute on-the-fly to save memory

    for _ in 0..config.k.min(n) {
        if remaining.is_empty() {
            break;
        }

        // Find item with maximum quality * residual_norm
        // This is the greedy approximation to max log-det
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (pos, &cand_idx) in remaining.iter().enumerate() {
            // DPP score: quality * sqrt(c[i]) where c[i] is residual norm squared
            // Using sqrt for numerical stability
            let score = qualities[cand_idx] * c[cand_idx].max(0.0).sqrt();

            if score > best_score {
                best_score = score;
                best_idx = pos;
            }
        }

        let chosen = remaining.swap_remove(best_idx);
        selected_indices.push(chosen);

        // Update residual norms for remaining items
        // c[i] -= (v_i · v_chosen)^2 / c[chosen]
        let chosen_emb = embeddings[chosen].as_ref();
        let c_chosen = c[chosen].max(1e-9); // Avoid division by zero

        for &idx in &remaining {
            let v_i = embeddings[idx].as_ref();
            let dot_product = simd::dot(v_i, chosen_emb);
            c[idx] -= (dot_product * dot_product) / c_chosen;
            c[idx] = c[idx].max(0.0); // Clamp to avoid negative from numerical error
        }
    }

    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmr_pure_relevance() {
        // λ=1.0 should return in relevance order
        let candidates = vec![("a", 0.9), ("b", 0.8), ("c", 0.7)];
        let sim = vec![
            1.0, 0.9, 0.9, // High similarity everywhere
            0.9, 1.0, 0.9, 0.9, 0.9, 1.0,
        ];

        let result = mmr(&candidates, &sim, MmrConfig::new(1.0, 3));
        assert_eq!(result[0].0, "a");
        assert_eq!(result[1].0, "b");
        assert_eq!(result[2].0, "c");
    }

    #[test]
    fn mmr_config_clamps_lambda() {
        assert_eq!(MmrConfig::new(-0.5, 10).lambda, 0.0);
        assert_eq!(MmrConfig::new(1.5, 10).lambda, 1.0);
        assert_eq!(MmrConfig::default().with_lambda(-0.5).lambda, 0.0);
        assert_eq!(MmrConfig::default().with_lambda(1.5).lambda, 1.0);
        assert_eq!(MmrConfig::default().with_lambda(0.7).lambda, 0.7);
    }

    #[test]
    fn mmr_prefers_diverse() {
        // λ=0.5 with diverse third option should prefer it over similar second
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];
        let sim = vec![
            1.0, 0.95, 0.1, // a: very similar to b, very different from c
            0.95, 1.0, 0.1, // b: very similar to a, very different from c
            0.1, 0.1, 1.0, // c: different from both a and b
        ];

        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 2));
        assert_eq!(result[0].0, "a"); // Most relevant
        assert_eq!(result[1].0, "c"); // Most diverse from a
    }

    #[test]
    fn mmr_pure_diversity() {
        // λ=0.0 should maximize diversity
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.1)];
        let sim = vec![
            1.0, 0.99, 0.01, // a and b nearly identical, c is different
            0.99, 1.0, 0.01, 0.01, 0.01, 1.0,
        ];

        let result = mmr(&candidates, &sim, MmrConfig::new(0.0, 2));
        // First pick is arbitrary (all have same "diversity" from empty set)
        // Second pick must be c (most different from first)
        assert!(result.iter().any(|(id, _)| *id == "c"));
    }

    #[test]
    fn mmr_cosine_basic() {
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0], // Very similar to a
            vec![0.0, 0.0, 1.0],  // Orthogonal to a
        ];

        let result = mmr_cosine(&candidates, &embeddings, MmrConfig::new(0.5, 2));
        assert_eq!(result[0].0, "a");
        assert_eq!(result[1].0, "c"); // Diverse from a
    }

    #[test]
    fn mmr_empty_candidates() {
        let candidates: Vec<(&str, f32)> = vec![];
        let sim: Vec<f32> = vec![];
        let result = mmr(&candidates, &sim, MmrConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn mmr_k_larger_than_n() {
        let candidates = vec![("a", 0.9), ("b", 0.8)];
        let sim = vec![1.0, 0.5, 0.5, 1.0];
        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 10));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn mmr_single_candidate() {
        let candidates = vec![("a", 0.9)];
        let sim = vec![1.0];
        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 1));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "a");
    }

    #[test]
    fn try_mmr_invalid_matrix() {
        let candidates = vec![("a", 0.9), ("b", 0.8)];
        let sim = vec![1.0]; // Wrong size: should be 4
        let result = try_mmr(&candidates, &sim, MmrConfig::default());
        assert!(result.is_err());
    }

    /// Verify MMR formula: score = λ × Rel(d) - (1-λ) × max_{d' ∈ S} Sim(d, d')
    /// For first selection (S = ∅), score = λ × Rel(d) (no diversity penalty)
    #[test]
    fn mmr_exact_formula_first_selection() {
        // With λ=0.7 and normalized relevances, first pick should be highest relevance
        // since there's no diversity penalty yet
        let candidates = vec![("a", 0.5), ("b", 1.0), ("c", 0.8)];
        let sim = vec![
            1.0, 0.5, 0.5, // doesn't matter for first selection
            0.5, 1.0, 0.5, 0.5, 0.5, 1.0,
        ];

        let result = mmr(&candidates, &sim, MmrConfig::new(0.7, 1));

        // b has highest relevance (1.0), should be selected first
        assert_eq!(
            result[0].0, "b",
            "First selection should be highest relevance"
        );
    }

    /// Verify MMR formula for second selection
    /// After selecting 'a', score(d) = λ × norm_rel(d) - (1-λ) × sim(d, a)
    #[test]
    fn mmr_exact_formula_second_selection() {
        // Candidates with relevances: a=0.9, b=0.6, c=0.3
        // Normalized to [0,1]: a=1.0, b=0.5, c=0.0
        let candidates = vec![("a", 0.9), ("b", 0.6), ("c", 0.3)];

        // Similarity matrix:
        // a-a=1.0, a-b=0.9, a-c=0.1
        // b-a=0.9, b-b=1.0, b-c=0.2
        // c-a=0.1, c-b=0.2, c-c=1.0
        let sim = vec![
            1.0, 0.9, 0.1, // row a
            0.9, 1.0, 0.2, // row b
            0.1, 0.2, 1.0, // row c
        ];

        // With λ=0.5:
        // First selection: 'a' (highest normalized relevance = 1.0)
        // Second selection scores:
        //   b: 0.5 × 0.5 - 0.5 × 0.9 = 0.25 - 0.45 = -0.20
        //   c: 0.5 × 0.0 - 0.5 × 0.1 = 0.00 - 0.05 = -0.05
        // c has higher score, so should be selected second
        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 2));

        assert_eq!(result[0].0, "a", "First should be 'a' (highest relevance)");
        assert_eq!(
            result[1].0, "c",
            "Second should be 'c' (more diverse from 'a')"
        );
    }

    /// Verify that equal relevance with λ=0 selects based on diversity only
    #[test]
    fn mmr_pure_diversity_equal_relevance() {
        // All same relevance - selection should be purely based on diversity
        let candidates = vec![("a", 0.5), ("b", 0.5), ("c", 0.5)];

        // a-b very similar (0.99), a-c and b-c orthogonal (0.01)
        let sim = vec![
            1.0, 0.99, 0.01, // a
            0.99, 1.0, 0.01, // b
            0.01, 0.01, 1.0, // c
        ];

        // With λ=0 (pure diversity), after first pick, should select most diverse
        let result = mmr(&candidates, &sim, MmrConfig::new(0.0, 3));

        // After any first pick, 'c' should be second (most diverse from a or b)
        // If a selected first, c is most diverse from a
        // If b selected first, c is most diverse from b
        // If c selected first, then a or b next (both equally diverse from c)
        // Third pick completes the set

        // Just verify all three are selected (exact order depends on tie-breaking)
        assert_eq!(result.len(), 3);
        let ids: Vec<_> = result.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// MMR output length bounded by min(k, n)
        #[test]
        fn mmr_output_length_bounded(
            n in 1usize..10,
            k in 1usize..20,
            lambda in 0.0f32..1.0,
        ) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();
            let sim: Vec<f32> = (0..n * n).map(|_| 0.5).collect();

            let result = mmr(&candidates, &sim, MmrConfig::new(lambda, k));
            prop_assert!(result.len() <= k.min(n));
        }

        /// MMR returns unique IDs (no duplicates)
        #[test]
        fn mmr_unique_ids(n in 1usize..10) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();
            let sim: Vec<f32> = (0..n * n).map(|_| 0.5).collect();

            let result = mmr(&candidates, &sim, MmrConfig::default().with_k(n));
            let mut seen = std::collections::HashSet::new();
            for (id, _) in &result {
                prop_assert!(seen.insert(*id), "Duplicate ID: {}", id);
            }
        }

        /// λ=1.0 should return in relevance order
        #[test]
        fn mmr_lambda_1_is_relevance_order(n in 2usize..8) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();
            // Identity similarity matrix (all items equally similar)
            let sim: Vec<f32> = (0..n)
                .flat_map(|i| (0..n).map(move |j| if i == j { 1.0 } else { 0.0 }))
                .collect();

            let result = mmr(&candidates, &sim, MmrConfig::new(1.0, n));

            // Should be in relevance order (id 0 has highest score)
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1,
                    "Not sorted: {:?} >= {:?}", window[0], window[1]);
            }
        }

        /// MMR with empty input returns empty output
        #[test]
        fn mmr_empty_returns_empty(k in 0usize..10, lambda in 0.0f32..1.0) {
            let candidates: Vec<(u32, f32)> = vec![];
            let sim: Vec<f32> = vec![];

            let result = mmr(&candidates, &sim, MmrConfig::new(lambda, k));
            prop_assert!(result.is_empty());
        }

        /// MMR with k=0 returns empty output
        #[test]
        fn mmr_k_zero_returns_empty(n in 1usize..10) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 0.5))
                .collect();
            let sim: Vec<f32> = vec![0.5; n * n];

            let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 0));
            prop_assert!(result.is_empty());
        }

        /// try_mmr returns Err for wrong matrix size
        #[test]
        fn try_mmr_wrong_size_errors(n in 2usize..10, wrong_size in 0usize..5) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 0.5))
                .collect();
            let correct_size = n * n;
            let actual_size = if wrong_size == 0 { 0 } else { correct_size.saturating_sub(wrong_size) };

            // Skip if accidentally correct size
            prop_assume!(actual_size != correct_size);

            let sim: Vec<f32> = vec![0.5; actual_size];
            let result = try_mmr(&candidates, &sim, MmrConfig::default());
            prop_assert!(result.is_err());
        }

        /// MMR with all equal relevance should still work
        #[test]
        fn mmr_equal_relevance(n in 1usize..8) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 0.5)) // All same relevance
                .collect();
            let sim: Vec<f32> = vec![0.5; n * n];

            let result = mmr(&candidates, &sim, MmrConfig::default().with_k(n));
            prop_assert_eq!(result.len(), n);
        }

        /// mmr_cosine produces same IDs as mmr with equivalent matrix
        #[test]
        fn mmr_cosine_consistent_with_mmr(n in 2usize..6) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();

            // Create orthogonal embeddings for simple testing
            let embeddings: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let mut v = vec![0.0; n];
                    v[i] = 1.0;
                    v
                })
                .collect();

            // Build similarity matrix from embeddings
            let mut sim = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    sim.push(simd::cosine(&embeddings[i], &embeddings[j]));
                }
            }

            let mmr_result = mmr(&candidates, &sim, MmrConfig::new(0.5, n));
            let cosine_result = mmr_cosine(&candidates, &embeddings, MmrConfig::new(0.5, n));

            // Same IDs should be selected
            let mmr_ids: std::collections::HashSet<_> = mmr_result.iter().map(|(id, _)| *id).collect();
            let cosine_ids: std::collections::HashSet<_> = cosine_result.iter().map(|(id, _)| *id).collect();
            prop_assert_eq!(mmr_ids, cosine_ids);
        }

        /// DPP with empty input returns empty output
        #[test]
        fn dpp_empty_returns_empty(k in 0usize..10) {
            let candidates: Vec<(u32, f32)> = vec![];
            let embeddings: Vec<Vec<f32>> = vec![];

            let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(k));
            prop_assert!(result.is_empty());
        }

        /// DPP with k=0 returns empty output
        #[test]
        fn dpp_k_zero_returns_empty(n in 1usize..10) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 0.5))
                .collect();
            let embeddings: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let mut v = vec![0.0; 3];
                    v[i % 3] = 1.0;
                    v
                })
                .collect();

            let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(0));
            prop_assert!(result.is_empty());
        }

        /// DPP selects at most k items
        #[test]
        fn dpp_selects_at_most_k(n in 2usize..10, k in 1usize..8) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 0.5))
                .collect();
            let embeddings: Vec<Vec<f32>> = (0..n)
                .map(|_| vec![1.0, 0.0, 0.0])
                .collect();

            let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(k));
            prop_assert!(result.len() <= k.min(n));
        }

        /// DPP prefers orthogonal items (high diversity)
        #[test]
        fn dpp_prefers_orthogonal(n in 3usize..6) {
            // Create embeddings where first and second are similar, third is orthogonal
            let mut embeddings: Vec<Vec<f32>> = vec![
                vec![1.0, 0.0, 0.0],
                vec![0.99, 0.1, 0.0], // Similar to first
            ];
            // Add orthogonal items
            for i in 2..n {
                let mut v = vec![0.0; 3];
                v[i % 3] = 1.0;
                embeddings.push(v);
            }

            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 0.9 - i as f32 * 0.05)) // Decreasing relevance
                .collect();

            let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(2));

            // First item (highest quality) should be selected
            prop_assert_eq!(result[0].0, 0);
            // Second item should NOT be the similar one if there's an orthogonal alternative
            if n > 2 {
                prop_assert_ne!(result[1].0, 1, "DPP should prefer orthogonal over similar");
            }
        }

        /// DPP with all equal embeddings doesn't crash
        #[test]
        fn dpp_equal_embeddings(n in 1usize..8) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();
            let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0]; n]; // All identical

            // Should not panic
            let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(n));
            // Should still select something
            prop_assert!(!result.is_empty() || n == 0);
        }
    }
}

#[cfg(test)]
mod dpp_tests {
    use super::*;

    #[test]
    fn dpp_orthogonal_prefers_diverse() {
        // Three orthogonal vectors with high relevance
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(3));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn dpp_similar_items_penalized() {
        // a and b are nearly identical, c is orthogonal
        let candidates = vec![("a", 0.95), ("b", 0.90), ("c", 0.85)];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0], // Very similar to a
            vec![0.0, 0.0, 1.0],  // Orthogonal to both
        ];

        let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(2));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "a"); // Highest quality
                                      // Second should be c (orthogonal) not b (similar)
        assert_eq!(result[1].0, "c");
    }

    #[test]
    fn dpp_config_alpha() {
        // High alpha = more weight on relevance
        let config = DppConfig::default().with_alpha(10.0);
        assert_eq!(config.alpha, 10.0);

        // Negative alpha clamped to 0
        let config = DppConfig::default().with_alpha(-1.0);
        assert_eq!(config.alpha, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Failure Mode Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod failure_mode_tests {
    use super::*;

    /// MMR with negative similarity values (anti-correlated embeddings).
    /// Negative similarity should boost diversity score, not cause issues.
    #[test]
    fn mmr_negative_similarity_handled() {
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];
        // Negative similarity between a-b means they're anti-correlated
        let sim = vec![
            1.0, -0.9, 0.5, // a: anti-correlated with b
            -0.9, 1.0, 0.5, // b: anti-correlated with a
            0.5, 0.5, 1.0, // c: somewhat similar to both
        ];

        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 2));
        assert_eq!(result.len(), 2);
        // After selecting 'a', 'b' should be preferred since sim(a,b)=-0.9
        // gives MMR(b) = 0.5*rel(b) - 0.5*(-0.9) = 0.5*rel(b) + 0.45 (bonus!)
        assert_eq!(result[0].0, "a");
        assert_eq!(result[1].0, "b"); // Negative similarity = diversity boost
    }

    /// MMR with similarity outside [0, 1] range.
    /// The algorithm should still work correctly.
    #[test]
    fn mmr_similarity_outside_unit_range() {
        let candidates = vec![("a", 0.9), ("b", 0.85)];
        // Similarity > 1 (invalid but shouldn't crash)
        let sim = vec![
            1.0, 2.0, // a-b has sim=2.0 (impossible for cosine, but tests robustness)
            2.0, 1.0,
        ];

        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 2));
        assert_eq!(result.len(), 2);
        // Should still select both, just with weird scores
    }

    /// DPP with zero-norm embeddings.
    /// Should handle gracefully (c[i] = 0).
    #[test]
    fn dpp_zero_norm_embeddings() {
        let candidates = vec![("a", 0.9), ("b", 0.85)];
        let embeddings = vec![
            vec![0.0, 0.0, 0.0], // Zero vector
            vec![1.0, 0.0, 0.0], // Normal vector
        ];

        // Should not panic despite zero-norm
        let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(2));
        // At least one item should be selected (the non-zero one)
        assert!(!result.is_empty());
    }

    /// DPP with all identical embeddings.
    /// After first selection, c[i] → 0 for all remaining.
    #[test]
    fn dpp_all_identical_embeddings() {
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0], // Identical to a
            vec![1.0, 0.0, 0.0], // Identical to a
        ];

        // Should not panic
        let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(3));
        // First item selected, rest have c → 0, so quality dominates
        assert!(!result.is_empty());
    }

    /// MMR cosine with zero-norm embeddings.
    /// Cosine returns 0 for zero vectors.
    #[test]
    fn mmr_cosine_zero_norm_embeddings() {
        let candidates = vec![("a", 0.9), ("b", 0.85)];
        let embeddings = vec![
            vec![0.0, 0.0], // Zero vector
            vec![1.0, 0.0], // Normal vector
        ];

        let result = mmr_cosine(&candidates, &embeddings, MmrConfig::new(0.5, 2));
        assert_eq!(result.len(), 2);
    }

    /// DPP with anti-correlated embeddings (negative dot products).
    #[test]
    fn dpp_anticorrelated_embeddings() {
        let candidates = vec![("a", 0.9), ("b", 0.85)];
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0], // Opposite direction
        ];

        // Orthogonal in diversity terms (should both be selected)
        let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(2));
        assert_eq!(result.len(), 2);
    }

    /// MMR with NaN in similarity matrix.
    /// NaN propagates to scores, but shouldn't crash.
    #[test]
    fn mmr_nan_in_similarity() {
        let candidates = vec![("a", 0.9), ("b", 0.85)];
        let sim = vec![1.0, f32::NAN, f32::NAN, 1.0];

        // Should not panic (NaN comparison returns false)
        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 2));
        // At least some selection happens
        assert!(!result.is_empty());
    }

    /// DPP with NaN in embeddings.
    #[test]
    fn dpp_nan_in_embeddings() {
        let candidates = vec![("a", 0.9), ("b", 0.85)];
        let embeddings = vec![vec![1.0, 0.0], vec![f32::NAN, 0.0]];

        // Should not panic
        let result = dpp(&candidates, &embeddings, DppConfig::default().with_k(2));
        // First item (non-NaN) should be selected
        assert!(!result.is_empty());
    }

    /// MMR preserves original scores in output.
    #[test]
    fn mmr_preserves_original_scores() {
        let candidates = vec![("a", 0.95), ("b", 0.85), ("c", 0.75)];
        let sim = vec![1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0];

        let result = mmr(&candidates, &sim, MmrConfig::new(0.5, 3));

        // Output should have original scores, not normalized MMR scores
        for (id, score) in &result {
            let original = candidates.iter().find(|(i, _)| i == id).unwrap().1;
            assert!(
                (score - original).abs() < 1e-6,
                "Score for {} was modified: {} vs {}",
                id,
                score,
                original
            );
        }
    }
}
