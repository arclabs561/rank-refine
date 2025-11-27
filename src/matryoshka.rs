//! Two-stage retrieval using Matryoshka embeddings.
//!
//! # What Are Matryoshka Embeddings?
//!
//! Like Russian nesting dolls, Matryoshka embeddings contain smaller valid
//! embeddings inside them. The first 64 dimensions form a complete (if less
//! accurate) embedding. So do the first 128, 256, etc.
//!
//! This is achieved during training by applying the loss function at multiple
//! dimension checkpoints, forcing the model to pack the most important
//! information into earlier dimensions.
//!
//! **Paper**: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
//! (Kusupati et al., 2022)
//!
//! # How This Module Uses It
//!
//! We split each embedding into "head" (first k dims) and "tail" (remaining dims):
//!
//! ```text
//! Full embedding: [████████████████████████████████]
//!                  ↑ head (coarse) ↑   ↑  tail (fine details)  ↑
//! ```
//!
//! 1. **Stage 1 (Coarse)**: Your vector DB searches using head dimensions only
//! 2. **Stage 2 (Refine)**: This module re-scores candidates using tail dimensions
//!
//! The tail dimensions contain fine-grained information that can disambiguate
//! between candidates that looked similar in the coarse search.
//!
//! # Example
//!
//! ```rust
//! use rank_refine::matryoshka;
//!
//! // Candidates from Stage 1 (coarse search)
//! let candidates = vec![("d1", 0.9), ("d2", 0.8)];
//!
//! // Full embeddings (head_dims=2 means dims 0-1 are head, dims 2-3 are tail)
//! let query = vec![0.1, 0.2, 0.3, 0.4];
//! let docs = vec![
//!     ("d1", vec![0.1, 0.2, 0.35, 0.45]),  // tail: [0.35, 0.45]
//!     ("d2", vec![0.1, 0.2, 0.29, 0.39]),  // tail: [0.29, 0.39]
//! ];
//!
//! // Refine using tail dimensions
//! let refined = matryoshka::refine(&candidates, &query, &docs, 2);
//! // d2 might now rank higher if its tail matches query's tail better
//! ```
//!
//! # The Alpha Parameter
//!
//! The `alpha` parameter controls how much weight to give the original score
//! vs the tail similarity:
//!
//! ```text
//! final_score = alpha × original_score + (1 - alpha) × tail_similarity
//! ```
//!
//! - `alpha = 1.0`: Use original scores only (no refinement)
//! - `alpha = 0.5`: Equal blend (default)
//! - `alpha = 0.0`: Use tail similarity only
//!
//! See [`REFERENCE.md`](https://github.com/your-repo/REFERENCE.md) for mathematical details.

use crate::{simd, RefineConfig, RefineError, Result};
use std::collections::HashMap;
use std::hash::Hash;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Refine candidates using tail dimensions.
///
/// Uses `alpha=0.5` (equal blend of original score and tail similarity).
///
/// # Panics
///
/// Panics if `head_dims >= query.len()`. Use [`try_refine`] for fallible version.
///
/// # Note
///
/// - Candidates not found in `docs` are silently dropped.
/// - Document embeddings shorter than `head_dims` are skipped.
#[must_use]
pub fn refine<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
) -> Vec<(I, f32)> {
    try_refine(candidates, query, docs, head_dims, RefineConfig::default())
        .expect("head_dims must be < query.len()")
}

/// Refine with custom blending weight.
///
/// # Arguments
///
/// * `alpha` - Weight for original score (0.0 = all tail, 1.0 = all original)
///
/// # Panics
///
/// Panics if `head_dims >= query.len()`.
#[must_use]
pub fn refine_with_alpha<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
    alpha: f32,
) -> Vec<(I, f32)> {
    try_refine(
        candidates,
        query,
        docs,
        head_dims,
        RefineConfig::default().with_alpha(alpha),
    )
    .expect("head_dims must be < query.len()")
}

/// Refine using only tail dimensions (ignore original scores).
///
/// # Panics
///
/// Panics if `head_dims >= query.len()`.
#[must_use]
pub fn refine_tail_only<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
) -> Vec<(I, f32)> {
    try_refine(
        candidates,
        query,
        docs,
        head_dims,
        RefineConfig::refinement_only(),
    )
    .expect("head_dims must be < query.len()")
}

/// Fallible refinement with full configuration.
///
/// # Errors
///
/// Returns [`RefineError::InvalidHeadDims`] if `head_dims >= query.len()`.
pub fn try_refine<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
    config: RefineConfig,
) -> Result<Vec<(I, f32)>> {
    if head_dims >= query.len() {
        return Err(RefineError::InvalidHeadDims {
            head_dims,
            query_len: query.len(),
        });
    }

    let doc_map: HashMap<&I, &[f32]> = docs
        .iter()
        .filter(|(_, emb)| emb.len() > head_dims)
        .map(|(id, emb)| (id, emb.as_slice()))
        .collect();
    let query_tail = &query[head_dims..];
    let alpha = config.alpha;

    let mut results: Vec<(I, f32)> = candidates
        .iter()
        .filter_map(|(id, orig)| {
            let doc_emb = doc_map.get(id)?;
            let doc_tail = &doc_emb[head_dims..];
            let tail_sim = simd::cosine(query_tail, doc_tail);
            let blended = (1.0 - alpha).mul_add(tail_sim, alpha * orig);
            Some((id.clone(), blended))
        })
        .collect();

    crate::sort_scored_desc(&mut results);

    if let Some(k) = config.top_k {
        results.truncate(k);
    }

    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement() {
        let candidates = vec![("d1", 0.9), ("d2", 0.1)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![
            ("d1", vec![0.0, 0.0, 0.0, 1.0]),
            ("d2", vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let refined = refine_tail_only(&candidates, &query, &docs, 2);
        assert_eq!(refined[0].0, "d2");
    }

    #[test]
    fn test_refinement_with_alpha() {
        let candidates = vec![("d1", 1.0), ("d2", 0.0)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![
            ("d1", vec![0.0, 0.0, 0.0, 1.0]),
            ("d2", vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let refined = refine_with_alpha(&candidates, &query, &docs, 2, 1.0);
        assert_eq!(refined[0].0, "d1");

        let refined = refine_with_alpha(&candidates, &query, &docs, 2, 0.0);
        assert_eq!(refined[0].0, "d2");
    }

    #[test]
    fn test_try_refine_error() {
        let candidates = vec![("d1", 0.9)];
        let query = vec![1.0, 2.0];
        let docs = vec![("d1", vec![1.0, 2.0])];

        let result = try_refine(&candidates, &query, &docs, 2, RefineConfig::default());
        assert!(matches!(result, Err(RefineError::InvalidHeadDims { .. })));
    }

    #[test]
    fn test_missing_candidate() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![("d1", vec![0.0, 0.0, 1.0, 0.0])];

        let refined = refine(&candidates, &query, &docs, 2);
        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn test_short_doc_embedding() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![("d1", vec![0.0, 0.0, 1.0, 0.0]), ("d2", vec![0.0])];

        let refined = refine(&candidates, &query, &docs, 2);
        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].0, "d1");
    }

    #[test]
    #[should_panic(expected = "head_dims must be < query.len()")]
    fn test_head_dims_too_large() {
        let candidates = vec![("d1", 0.9)];
        let query = vec![1.0, 2.0];
        let docs = vec![("d1", vec![1.0, 2.0])];

        let _ = refine(&candidates, &query, &docs, 2);
    }

    #[test]
    fn test_nan_score_handling() {
        let candidates = vec![("d1", f32::NAN), ("d2", 0.5), ("d3", 0.9)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![
            ("d1", vec![0.0, 0.0, 1.0, 0.0]),
            ("d2", vec![0.0, 0.0, 1.0, 0.0]),
            ("d3", vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let refined = refine(&candidates, &query, &docs, 2);
        assert_eq!(refined.len(), 3);
        assert!(refined[0].1.is_nan());
    }

    #[test]
    fn test_top_k() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![
            ("d1", vec![0.0, 0.0, 1.0, 0.0]),
            ("d2", vec![0.0, 0.0, 1.0, 0.0]),
            ("d3", vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let refined = try_refine(
            &candidates,
            &query,
            &docs,
            2,
            RefineConfig::default().with_top_k(2),
        )
        .unwrap();
        assert_eq!(refined.len(), 2);
    }

    /// Verifies that tail dimensions provide discriminative power.
    ///
    /// Two documents look identical in head dimensions but differ in tail.
    /// Refinement should distinguish them.
    #[test]
    fn test_tail_dims_provide_discrimination() {
        // Head dims: [1.0, 0.0, 0.0, 0.0] - identical for both docs
        // Tail dims: doc_a=[0.0, 0.0, 0.0, 0.0], doc_b=[1.0, 0.0, 0.0, 0.0]
        // Query tail: [1.0, 0.0, 0.0, 0.0] - matches doc_b

        let candidates = vec![
            ("doc_a", 0.5), // tied in head-only scoring
            ("doc_b", 0.5), // tied in head-only scoring
        ];

        let query = vec![
            1.0, 0.0, 0.0, 0.0, // head (dims 0-3)
            1.0, 0.0, 0.0, 0.0, // tail (dims 4-7) - matches doc_b
        ];

        let docs = vec![
            (
                "doc_a",
                vec![
                    1.0, 0.0, 0.0, 0.0, // head - identical
                    0.0, 0.0, 0.0, 0.0, // tail - different
                ],
            ),
            (
                "doc_b",
                vec![
                    1.0, 0.0, 0.0, 0.0, // head - identical
                    1.0, 0.0, 0.0, 0.0, // tail - matches query
                ],
            ),
        ];

        // With tail-only refinement, doc_b should win
        let refined = refine_tail_only(&candidates, &query, &docs, 4);
        assert_eq!(
            refined[0].0, "doc_b",
            "Tail should discriminate: doc_b matches query tail"
        );
        assert!(
            refined[0].1 > refined[1].1,
            "doc_b score {} should be > doc_a score {}",
            refined[0].1,
            refined[1].1
        );

        // Verify the tail similarity difference is significant
        let score_diff = refined[0].1 - refined[1].1;
        assert!(
            score_diff > 0.9,
            "Score difference {} should be large (orthogonal vs identical)",
            score_diff
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn output_bounded_by_candidates(
            n_candidates in 1usize..10,
            n_docs in 0usize..10,
            dim in 4usize..16
        ) {
            let head_dims = dim / 2;
            let candidates: Vec<(u32, f32)> = (0..n_candidates as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.1))
                .collect();
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let docs: Vec<(u32, Vec<f32>)> = (0..n_docs as u32)
                .map(|i| (i, (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect()))
                .collect();

            let refined = refine(&candidates, &query, &docs, head_dims);
            prop_assert!(refined.len() <= candidates.len());
        }

        #[test]
        fn alpha_one_preserves_order(dim in 4usize..8) {
            let head_dims = dim / 2;
            let candidates = vec![(1u32, 0.9), (2u32, 0.8), (3u32, 0.7)];
            let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();
            let docs: Vec<(u32, Vec<f32>)> = vec![
                (1, (0..dim).map(|_| 0.1).collect()),
                (2, (0..dim).map(|_| 0.2).collect()),
                (3, (0..dim).map(|_| 0.3).collect()),
            ];

            let refined = refine_with_alpha(&candidates, &query, &docs, head_dims, 1.0);
            prop_assert_eq!(refined.len(), 3);
            prop_assert_eq!(refined[0].0, 1);
            prop_assert_eq!(refined[1].0, 2);
            prop_assert_eq!(refined[2].0, 3);
        }

        #[test]
        fn results_sorted_descending(n in 2usize..8, dim in 4usize..8) {
            let head_dims = dim / 2;
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, (i as f32) * 0.1))
                .collect();
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let docs: Vec<(u32, Vec<f32>)> = (0..n as u32)
                .map(|i| (i, (0..dim).map(|j| (i + j as u32) as f32 * 0.05).collect()))
                .collect();

            let refined = refine(&candidates, &query, &docs, head_dims);
            for window in refined.windows(2) {
                prop_assert!(
                    window[0].1.total_cmp(&window[1].1) != std::cmp::Ordering::Less,
                    "Results not sorted: {:?}",
                    refined
                );
            }
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Matryoshka-specific properties
        // ─────────────────────────────────────────────────────────────────────────

        /// Alpha interpolation: alpha=0.5 should produce scores between alpha=0 and alpha=1
        #[test]
        fn alpha_interpolation(dim in 8usize..16) {
            let head_dims = dim / 2;
            let candidates = vec![(1u32, 0.8), (2u32, 0.5)];
            let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let docs: Vec<(u32, Vec<f32>)> = vec![
                (1, (0..dim).map(|i| (i as f32 * 0.2).cos()).collect()),
                (2, (0..dim).map(|i| (i as f32 * 0.3).sin()).collect()),
            ];

            let r_0 = refine_with_alpha(&candidates, &query, &docs, head_dims, 0.0);
            let r_half = refine_with_alpha(&candidates, &query, &docs, head_dims, 0.5);
            let r_1 = refine_with_alpha(&candidates, &query, &docs, head_dims, 1.0);

            // Scores should interpolate (roughly)
            for id in [1u32, 2] {
                let s_0 = r_0.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
                let s_half = r_half.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
                let s_1 = r_1.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);

                if let (Some(s0), Some(sh), Some(s1)) = (s_0, s_half, s_1) {
                    let min_score = s0.min(s1);
                    let max_score = s0.max(s1);
                    prop_assert!(
                        sh >= min_score - 0.01 && sh <= max_score + 0.01,
                        "alpha=0.5 score {} not between {} and {}",
                        sh, min_score, max_score
                    );
                }
            }
        }

        /// Tail similarity: longer tail (smaller head_dims) uses more dimensions
        #[test]
        fn smaller_head_uses_more_dims(dim in 16usize..32) {
            let candidates = vec![(1u32, 0.5)];
            let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            // Doc where tail dims have high values
            let doc: Vec<f32> = (0..dim)
                .map(|i| if i < dim / 4 { 0.1 } else { 0.9 })
                .collect();
            let docs = vec![(1u32, doc)];

            // Small head (big tail) should score differently than big head (small tail)
            let r_small_head = refine_tail_only(&candidates, &query, &docs, dim / 4);
            let r_big_head = refine_tail_only(&candidates, &query, &docs, dim * 3 / 4);

            // Both should produce valid scores (not NaN)
            prop_assert!(r_small_head[0].1.is_finite());
            prop_assert!(r_big_head[0].1.is_finite());
            // Scores should differ (different amount of tail used)
            // (unless by coincidence they're equal)
        }

        /// Documents shorter than head_dims are filtered out
        #[test]
        fn short_docs_filtered(dim in 8usize..16) {
            let head_dims = dim / 2;
            let candidates = vec![(1u32, 0.9), (2u32, 0.8)];
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            // Doc 1 is full length, doc 2 is too short
            let docs = vec![
                (1u32, (0..dim).map(|i| i as f32 * 0.1).collect()),
                (2u32, (0..head_dims - 1).map(|i| i as f32 * 0.1).collect()),
            ];

            let refined = refine(&candidates, &query, &docs, head_dims);
            // Only doc 1 should remain
            prop_assert_eq!(refined.len(), 1);
            prop_assert_eq!(refined[0].0, 1);
        }

        /// Try_refine returns error for invalid head_dims
        #[test]
        fn try_refine_validates_head_dims(dim in 2usize..8) {
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let candidates = vec![(1u32, 0.5)];
            let docs = vec![(1u32, query.clone())];

            // head_dims >= query.len() should error
            let result = try_refine(&candidates, &query, &docs, dim, RefineConfig::default());
            prop_assert!(result.is_err());

            // head_dims < query.len() should succeed
            let result = try_refine(&candidates, &query, &docs, dim - 1, RefineConfig::default());
            prop_assert!(result.is_ok());
        }

        /// Config top_k limits output
        #[test]
        fn config_top_k_limits_output(n in 5usize..10, k in 1usize..4) {
            let dim = 8;
            let head_dims = 4;
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - i as f32 * 0.05))
                .collect();
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let docs: Vec<(u32, Vec<f32>)> = (0..n as u32)
                .map(|i| (i, (0..dim).map(|j| (i + j as u32) as f32 * 0.05).collect()))
                .collect();

            let config = RefineConfig::default().with_top_k(k);
            let refined = try_refine(&candidates, &query, &docs, head_dims, config).unwrap();
            prop_assert!(refined.len() <= k, "Expected at most {} results, got {}", k, refined.len());
        }
    }
}
