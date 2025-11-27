//! Diversity-aware reranking algorithms.
//!
//! Select results that balance relevance with diversity — maximizing coverage
//! while minimizing redundancy.
//!
//! # Algorithms
//!
//! | Function | Best For |
//! |----------|----------|
//! | [`mmr`] | General diversity-relevance tradeoff |
//! | [`mmr_cosine`] | When you have raw embeddings |
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
    /// Create config with custom lambda and k.
    #[must_use]
    pub const fn new(lambda: f32, k: usize) -> Self {
        Self { lambda, k }
    }

    /// Set lambda (relevance-diversity tradeoff).
    #[must_use]
    pub const fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
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
/// * `candidates` - `(id, relevance_score)` pairs, sorted by relevance descending
/// * `similarity` - Flattened row-major similarity matrix (n×n)
/// * `config` - MMR configuration (lambda, k)
///
/// # Returns
///
/// Selected documents in MMR order (most relevant diverse first).
///
/// # Panics
///
/// Panics if `similarity.len() != candidates.len() * candidates.len()`.
#[must_use]
pub fn mmr<I: Clone>(
    candidates: &[(I, f32)],
    similarity: &[f32],
    config: MmrConfig,
) -> Vec<(I, f32)> {
    let n = candidates.len();
    assert_eq!(
        similarity.len(),
        n * n,
        "similarity matrix must be n×n where n = candidates.len()"
    );

    if n == 0 || config.k == 0 {
        return Vec::new();
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
    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect()
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
}
