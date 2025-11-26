//! Unified scoring traits for retrieval pipelines.
//!
//! This module provides a common abstraction over different scoring strategies:
//!
//! - **Dense scoring**: Single-vector embeddings (cosine, dot product)
//! - **Late interaction**: Multi-vector embeddings (`MaxSim`, `ColBERT`)
//! - **Cross-encoder**: Transformer-based scoring (external models)
//!
//! # Why Late Interaction?
//!
//! Dense retrieval (single-vector) is fast but loses fine-grained information.
//! Late interaction (multi-vector) preserves token-level semantics:
//!
//! | Approach | Representation | Query Time | Quality |
//! |----------|----------------|------------|---------|
//! | Dense | `Vec<f32>` | O(1) dot | Good |
//! | Late Interaction | `Vec<Vec<f32>>` | O(m*n) `MaxSim` | Better |
//! | Cross-encoder | Query + Doc text | O(n) inference | Best |
//!
//! Use dense for fast first-stage retrieval, then refine with late interaction
//! or cross-encoder for the top candidates.
//!
//! # Example
//!
//! ```rust
//! use rank_refine::scoring::{DenseScorer, Scorer};
//!
//! let scorer = DenseScorer::Cosine;
//! let query = &[1.0, 0.0, 0.0];
//! let doc = &[0.9, 0.1, 0.0];
//!
//! let score = scorer.score(query, doc);
//! assert!(score > 0.9);
//! ```

use crate::simd;

// ─────────────────────────────────────────────────────────────────────────────
// Dense Scoring (single-vector)
// ─────────────────────────────────────────────────────────────────────────────

/// Scoring strategy for dense (single-vector) embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseScorer {
    /// Dot product (assumes pre-normalized vectors for similarity).
    Dot,
    /// Cosine similarity (normalizes vectors).
    Cosine,
}

impl DenseScorer {
    /// Score similarity between query and document embeddings.
    #[must_use]
    pub fn score(&self, query: &[f32], doc: &[f32]) -> f32 {
        match self {
            Self::Dot => simd::dot(query, doc),
            Self::Cosine => simd::cosine(query, doc),
        }
    }
}

/// Trait for dense (single-vector) scoring.
///
/// Implement this for custom similarity functions.
pub trait Scorer {
    /// Score similarity between query and document.
    fn score(&self, query: &[f32], doc: &[f32]) -> f32;

    /// Rank documents by score (descending).
    fn rank<I: Clone>(&self, query: &[f32], docs: &[(I, &[f32])]) -> Vec<(I, f32)> {
        let mut results: Vec<(I, f32)> = docs
            .iter()
            .map(|(id, doc)| (id.clone(), self.score(query, doc)))
            .collect();
        crate::sort_scored_desc(&mut results);
        results
    }
}

impl Scorer for DenseScorer {
    fn score(&self, query: &[f32], doc: &[f32]) -> f32 {
        DenseScorer::score(self, query, doc)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Late Interaction Scoring (multi-vector)
// ─────────────────────────────────────────────────────────────────────────────

/// Scoring strategy for late interaction (multi-vector) embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateInteractionScorer {
    /// `MaxSim` with dot product (`ColBERT`-style).
    MaxSimDot,
    /// `MaxSim` with cosine similarity.
    MaxSimCosine,
}

impl LateInteractionScorer {
    /// Score similarity between query and document token embeddings.
    ///
    /// Returns the sum over query tokens of max similarity to any doc token.
    #[must_use]
    pub fn score(&self, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
        match self {
            Self::MaxSimDot => simd::maxsim(query_tokens, doc_tokens),
            Self::MaxSimCosine => simd::maxsim_cosine(query_tokens, doc_tokens),
        }
    }
}

/// Trait for late interaction (multi-vector) scoring.
///
/// Multi-vector representations preserve token-level semantics, enabling
/// fine-grained matching that outperforms dense scoring on complex queries.
///
/// # Convenience Methods
///
/// The trait provides both borrowed (`&[&[f32]]`) and owned (`&[Vec<f32>]`)
/// variants. Use whichever matches your data:
///
/// ```rust
/// use rank_refine::scoring::{TokenScorer, LateInteractionScorer};
///
/// let scorer = LateInteractionScorer::MaxSimDot;
///
/// // With owned vectors (common when loading from storage)
/// let q_owned = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let d_owned = vec![vec![0.9, 0.1]];
/// let score = scorer.score_vecs(&q_owned, &d_owned);
/// ```
pub trait TokenScorer {
    /// Score similarity between query tokens and document tokens.
    fn score_tokens(&self, query: &[&[f32]], doc: &[&[f32]]) -> f32;

    /// Score with owned vectors (convenience wrapper).
    fn score_vecs(&self, query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
        let q = crate::as_slices(query);
        let d = crate::as_slices(doc);
        self.score_tokens(&q, &d)
    }

    /// Rank documents by token-level score (descending).
    fn rank_tokens<I: Clone>(&self, query: &[&[f32]], docs: &[(I, Vec<&[f32]>)]) -> Vec<(I, f32)> {
        let mut results: Vec<(I, f32)> = docs
            .iter()
            .map(|(id, doc_tokens)| (id.clone(), self.score_tokens(query, doc_tokens)))
            .collect();
        crate::sort_scored_desc(&mut results);
        results
    }

    /// Rank with owned document vectors (convenience wrapper).
    fn rank_vecs<I: Clone>(&self, query: &[Vec<f32>], docs: &[(I, Vec<Vec<f32>>)]) -> Vec<(I, f32)> {
        let q = crate::as_slices(query);
        let mut results: Vec<(I, f32)> = docs
            .iter()
            .map(|(id, doc_tokens)| {
                let d = crate::as_slices(doc_tokens);
                (id.clone(), self.score_tokens(&q, &d))
            })
            .collect();
        crate::sort_scored_desc(&mut results);
        results
    }
}

impl TokenScorer for LateInteractionScorer {
    fn score_tokens(&self, query: &[&[f32]], doc: &[&[f32]]) -> f32 {
        self.score(query, doc)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Score Blending
// ─────────────────────────────────────────────────────────────────────────────

/// Blend two scores with a weight parameter.
///
/// `blended = alpha * score_a + (1 - alpha) * score_b`
///
/// Uses `mul_add` for better floating-point precision.
#[inline]
#[must_use]
pub fn blend(score_a: f32, score_b: f32, alpha: f32) -> f32 {
    (1.0 - alpha).mul_add(score_b, alpha * score_a)
}

/// Normalize scores to [0, 1] range.
///
/// Returns original scores if all values are equal (avoids division by zero).
#[must_use]
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let (min, max) = scores
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &s| {
            (lo.min(s), hi.max(s))
        });

    let range = max - min;
    if range < 1e-9 {
        // All scores equal, return 0.5 for all
        return vec![0.5; scores.len()];
    }

    scores.iter().map(|&s| (s - min) / range).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pooler Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for token pooling strategies.
///
/// Pooling reduces the number of token embeddings while preserving semantic
/// information. Different strategies trade off quality vs speed.
pub trait Pooler {
    /// Pool tokens to approximately `target_count` vectors.
    ///
    /// # Invariants
    ///
    /// - `pool(tokens).len() <= tokens.len()`
    /// - `pool(tokens).iter().all(|t| t.len() == dim)` (dimension preserved)
    /// - `pool(&[]).is_empty()`
    fn pool(&self, tokens: &[Vec<f32>], target_count: usize) -> Vec<Vec<f32>>;

    /// Pool with a compression factor (e.g., 2 = 50% reduction).
    fn pool_by_factor(&self, tokens: &[Vec<f32>], factor: usize) -> Vec<Vec<f32>> {
        if tokens.is_empty() || factor <= 1 {
            return tokens.to_vec();
        }
        let target = (tokens.len() / factor).max(1);
        self.pool(tokens, target)
    }
}

/// Sequential window pooling (fastest, position-aware).
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialPooler;

impl Pooler for SequentialPooler {
    fn pool(&self, tokens: &[Vec<f32>], target_count: usize) -> Vec<Vec<f32>> {
        if tokens.is_empty() || target_count >= tokens.len() {
            return tokens.to_vec();
        }
        let window = tokens.len().div_ceil(target_count);
        crate::colbert::pool_tokens_sequential(tokens, window)
    }
}

/// Greedy clustering pooler (default, quality-focused).
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusteringPooler;

impl Pooler for ClusteringPooler {
    fn pool(&self, tokens: &[Vec<f32>], target_count: usize) -> Vec<Vec<f32>> {
        if tokens.is_empty() || target_count >= tokens.len() {
            return tokens.to_vec();
        }
        let factor = tokens.len().div_ceil(target_count);
        crate::colbert::pool_tokens(tokens, factor)
    }
}

/// Adaptive pooler that chooses the best strategy based on compression factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaptivePooler;

impl Pooler for AdaptivePooler {
    fn pool(&self, tokens: &[Vec<f32>], target_count: usize) -> Vec<Vec<f32>> {
        if tokens.is_empty() || target_count >= tokens.len() {
            return tokens.to_vec();
        }
        let factor = tokens.len().div_ceil(target_count);
        crate::colbert::pool_tokens_adaptive(tokens, factor)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_dot() {
        let scorer = DenseScorer::Dot;
        assert!((scorer.score(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!((scorer.score(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-5);
    }

    #[test]
    fn test_dense_cosine() {
        let scorer = DenseScorer::Cosine;
        assert!((scorer.score(&[2.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!((scorer.score(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-5);
    }

    #[test]
    fn test_dense_rank() {
        let scorer = DenseScorer::Cosine;
        let query = &[1.0f32, 0.0][..];
        let docs: Vec<(&str, &[f32])> = vec![("d1", &[0.0, 1.0][..]), ("d2", &[1.0, 0.0][..])];

        let ranked = scorer.rank(query, &docs);
        assert_eq!(ranked[0].0, "d2");
    }

    #[test]
    fn test_late_interaction_maxsim() {
        let scorer = LateInteractionScorer::MaxSimDot;
        let q1: &[f32] = &[1.0, 0.0];
        let d1: &[f32] = &[1.0, 0.0];
        let d2: &[f32] = &[0.0, 1.0];

        let query = vec![q1];
        let doc = vec![d1, d2];

        // q1's max match is d1 (dot=1.0)
        assert!((scorer.score_tokens(&query, &doc) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend() {
        assert!((blend(1.0, 0.0, 1.0) - 1.0).abs() < 1e-5); // all score_a
        assert!((blend(1.0, 0.0, 0.0) - 0.0).abs() < 1e-5); // all score_b
        assert!((blend(1.0, 0.0, 0.5) - 0.5).abs() < 1e-5); // half and half
    }

    #[test]
    fn test_normalize_scores() {
        let scores = vec![0.0, 0.5, 1.0];
        let normalized = normalize_scores(&scores);
        assert!((normalized[0] - 0.0).abs() < 1e-5);
        assert!((normalized[1] - 0.5).abs() < 1e-5);
        assert!((normalized[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_scores_equal() {
        let scores = vec![0.5, 0.5, 0.5];
        let normalized = normalize_scores(&scores);
        assert!(normalized.iter().all(|&s| (s - 0.5).abs() < 1e-5));
    }

    #[test]
    fn test_normalize_scores_empty() {
        let scores: Vec<f32> = vec![];
        let normalized = normalize_scores(&scores);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_token_scorer_score_vecs() {
        let scorer = LateInteractionScorer::MaxSimDot;
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];

        let score = scorer.score_vecs(&query, &doc);
        assert!(score > 1.5); // both query tokens find good matches
    }

    #[test]
    fn test_token_scorer_rank_vecs() {
        let scorer = LateInteractionScorer::MaxSimDot;
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![
            ("d1", vec![vec![0.0, 1.0]]),  // orthogonal
            ("d2", vec![vec![1.0, 0.0]]),  // aligned
        ];

        let ranked = scorer.rank_vecs(&query, &docs);
        assert_eq!(ranked[0].0, "d2"); // aligned doc should rank first
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-10.0f32..10.0, len)
    }

    proptest! {
        /// Cosine similarity is commutative via Scorer trait
        #[test]
        fn scorer_cosine_commutative(a in arb_vec(32), b in arb_vec(32)) {
            let scorer = DenseScorer::Cosine;
            let ab = scorer.score(&a, &b);
            let ba = scorer.score(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5);
        }

        /// Dot product is commutative via Scorer trait
        #[test]
        fn scorer_dot_commutative(a in arb_vec(32), b in arb_vec(32)) {
            let scorer = DenseScorer::Dot;
            let ab = scorer.score(&a, &b);
            let ba = scorer.score(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5);
        }

        /// Rank preserves document count
        #[test]
        fn scorer_rank_preserves_count(n in 1usize..10, dim in 2usize..8) {
            let scorer = DenseScorer::Cosine;
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let docs: Vec<(u32, Vec<f32>)> = (0..n as u32)
                .map(|i| (i, (0..dim).map(|j| (i as usize + j) as f32 * 0.1).collect()))
                .collect();
            let doc_refs: Vec<(u32, &[f32])> = docs.iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();

            let ranked = scorer.rank(&query, &doc_refs);
            prop_assert_eq!(ranked.len(), n);
        }

        /// Blend with alpha=1 returns first score
        #[test]
        fn blend_alpha_one(a in -100.0f32..100.0, b in -100.0f32..100.0) {
            let blended = blend(a, b, 1.0);
            prop_assert!((blended - a).abs() < 1e-5);
        }

        /// Blend with alpha=0 returns second score
        #[test]
        fn blend_alpha_zero(a in -100.0f32..100.0, b in -100.0f32..100.0) {
            let blended = blend(a, b, 0.0);
            prop_assert!((blended - b).abs() < 1e-5);
        }

        /// Normalized scores are in [0, 1]
        #[test]
        fn normalize_bounded(scores in proptest::collection::vec(-100.0f32..100.0, 2..20)) {
            let normalized = normalize_scores(&scores);
            for &s in &normalized {
                prop_assert!(s >= -0.01 && s <= 1.01, "Score {} out of bounds", s);
            }
        }

        /// Normalized scores preserve relative ordering
        #[test]
        fn normalize_preserves_order(scores in proptest::collection::vec(-100.0f32..100.0, 2..10)) {
            let normalized = normalize_scores(&scores);
            for i in 0..scores.len() {
                for j in 0..scores.len() {
                    let orig_cmp = scores[i].total_cmp(&scores[j]);
                    let norm_cmp = normalized[i].total_cmp(&normalized[j]);
                    prop_assert_eq!(orig_cmp, norm_cmp, "Order changed at indices ({}, {})", i, j);
                }
            }
        }

        /// Blend is linear in alpha
        #[test]
        fn blend_is_linear(a in -10.0f32..10.0, b in -10.0f32..10.0, alpha in 0.0f32..1.0) {
            let blended = blend(a, b, alpha);
            let expected = alpha * a + (1.0 - alpha) * b;
            prop_assert!((blended - expected).abs() < 1e-5, "blend({}, {}, {}) = {}, expected {}", a, b, alpha, blended, expected);
        }

        /// Rank produces sorted output (descending)
        #[test]
        fn scorer_rank_is_sorted(n in 2usize..10, dim in 2usize..8) {
            let scorer = DenseScorer::Cosine;
            let query: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
            let docs: Vec<(u32, Vec<f32>)> = (0..n as u32)
                .map(|i| (i, (0..dim).map(|j| ((i as usize * dim + j) % 10) as f32).collect()))
                .collect();
            let doc_refs: Vec<(u32, &[f32])> = docs.iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();

            let ranked = scorer.rank(&query, &doc_refs);
            for w in ranked.windows(2) {
                prop_assert!(w[0].1 >= w[1].1, "Not sorted: {} < {}", w[0].1, w[1].1);
            }
        }

        /// Late interaction: `MaxSim` score is non-negative for non-negative inputs
        #[test]
        fn late_interaction_nonnegative(
            q_tokens in 1usize..4,
            d_tokens in 1usize..4,
            dim in 2usize..8
        ) {
            // Generate non-negative vectors
            let query: Vec<Vec<f32>> = (0..q_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) % 5) as f32 * 0.1 + 0.1).collect())
                .collect();
            let doc: Vec<Vec<f32>> = (0..d_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j + 3) % 5) as f32 * 0.1 + 0.1).collect())
                .collect();

            let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let doc_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();

            let scorer = LateInteractionScorer::MaxSimDot;
            let score = scorer.score(&query_refs, &doc_refs);
            prop_assert!(score >= 0.0, "`MaxSim` score {} should be non-negative", score);
        }

        /// Late interaction: empty doc returns 0
        #[test]
        fn late_interaction_empty_doc(dim in 2usize..8) {
            let query: Vec<Vec<f32>> = vec![vec![1.0; dim], vec![0.5; dim]];
            let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let doc_refs: Vec<&[f32]> = vec![];

            let scorer = LateInteractionScorer::MaxSimDot;
            let score = scorer.score(&query_refs, &doc_refs);
            prop_assert!((score - 0.0).abs() < 1e-9, "Empty doc should return 0, got {}", score);
        }

        /// Cosine scorer bounded [-1, 1] for normalized vectors
        #[test]
        fn scorer_cosine_bounded_normalized(dim in 2usize..16) {
            // Create unit vectors
            let a: Vec<f32> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
            let b: Vec<f32> = (0..dim).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();

            let scorer = DenseScorer::Cosine;
            let score = scorer.score(&a, &b);
            prop_assert!(score >= -1.01 && score <= 1.01, "Cosine {} out of bounds", score);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Pooler trait invariants
        // ─────────────────────────────────────────────────────────────────────────

        /// Pooler invariant: output count <= input count
        #[test]
        fn pooler_never_increases_count(n_tokens in 2usize..16, dim in 2usize..8, target in 1usize..8) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();

            let seq = SequentialPooler.pool(&tokens, target);
            let cluster = ClusteringPooler.pool(&tokens, target);
            let adaptive = AdaptivePooler.pool(&tokens, target);

            prop_assert!(seq.len() <= n_tokens, "Sequential increased count: {} -> {}", n_tokens, seq.len());
            prop_assert!(cluster.len() <= n_tokens, "Clustering increased count: {} -> {}", n_tokens, cluster.len());
            prop_assert!(adaptive.len() <= n_tokens, "Adaptive increased count: {} -> {}", n_tokens, adaptive.len());
        }

        /// Pooler invariant: dimension preserved
        #[test]
        fn pooler_preserves_dimension(n_tokens in 2usize..16, dim in 2usize..16, factor in 2usize..4) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();

            let seq = SequentialPooler.pool_by_factor(&tokens, factor);
            let cluster = ClusteringPooler.pool_by_factor(&tokens, factor);
            let adaptive = AdaptivePooler.pool_by_factor(&tokens, factor);

            prop_assert!(seq.iter().all(|t| t.len() == dim), "Sequential changed dim");
            prop_assert!(cluster.iter().all(|t| t.len() == dim), "Clustering changed dim");
            prop_assert!(adaptive.iter().all(|t| t.len() == dim), "Adaptive changed dim");
        }

        /// Pooler invariant: empty input returns empty
        #[test]
        fn pooler_empty_input(target in 1usize..10) {
            let empty: Vec<Vec<f32>> = vec![];

            prop_assert!(SequentialPooler.pool(&empty, target).is_empty());
            prop_assert!(ClusteringPooler.pool(&empty, target).is_empty());
            prop_assert!(AdaptivePooler.pool(&empty, target).is_empty());
        }

        /// Pooler invariant: factor 1 returns original
        #[test]
        fn pooler_factor_one_identity(n_tokens in 1usize..8, dim in 2usize..8) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();

            let seq = SequentialPooler.pool_by_factor(&tokens, 1);
            let cluster = ClusteringPooler.pool_by_factor(&tokens, 1);
            let adaptive = AdaptivePooler.pool_by_factor(&tokens, 1);

            prop_assert_eq!(seq.len(), n_tokens);
            prop_assert_eq!(cluster.len(), n_tokens);
            prop_assert_eq!(adaptive.len(), n_tokens);
        }

        /// TokenScorer rank produces sorted output
        #[test]
        fn token_scorer_rank_is_sorted(n_docs in 2usize..6, n_q in 1usize..3, dim in 2usize..8) {
            let query: Vec<Vec<f32>> = (0..n_q)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let toks: Vec<Vec<f32>> = (0..3)
                        .map(|t| (0..dim).map(|j| ((i as usize * 3 + t + j) as f32 * 0.1).cos()).collect())
                        .collect();
                    (i, toks)
                })
                .collect();

            let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let doc_refs: Vec<(u32, Vec<&[f32]>)> = docs.iter()
                .map(|(id, toks)| (*id, toks.iter().map(Vec::as_slice).collect()))
                .collect();

            let scorer = LateInteractionScorer::MaxSimDot;
            let ranked = scorer.rank_tokens(&query_refs, &doc_refs);

            for window in ranked.windows(2) {
                prop_assert!(
                    window[0].1 >= window[1].1 - 1e-6,
                    "Not sorted: {} >= {}",
                    window[0].1,
                    window[1].1
                );
            }
        }

        /// TokenScorer rank preserves count
        #[test]
        fn token_scorer_rank_preserves_count(n_docs in 1usize..6, n_q in 1usize..3, dim in 2usize..8) {
            let query: Vec<Vec<f32>> = (0..n_q)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();
            let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..n_docs as u32)
                .map(|i| {
                    let toks: Vec<Vec<f32>> = (0..2)
                        .map(|t| (0..dim).map(|j| ((i as usize * 2 + t + j) as f32 * 0.1).cos()).collect())
                        .collect();
                    (i, toks)
                })
                .collect();

            let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let doc_refs: Vec<(u32, Vec<&[f32]>)> = docs.iter()
                .map(|(id, toks)| (*id, toks.iter().map(Vec::as_slice).collect()))
                .collect();

            let scorer = LateInteractionScorer::MaxSimDot;
            let ranked = scorer.rank_tokens(&query_refs, &doc_refs);

            prop_assert_eq!(ranked.len(), n_docs);
        }
    }
}
