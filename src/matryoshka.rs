//! Matryoshka refinement using tail dimensions.
//!
//! MRL (Matryoshka Representation Learning) embeddings have a useful property:
//! the first k dimensions form a valid coarse embedding. This enables two-stage
//! retrieval:
//!
//! 1. **Coarse**: Search using first k dims (fast, cache-friendly)
//! 2. **Refine**: Re-score top candidates using remaining dims (accurate)
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::matryoshka;
//!
//! let candidates = vec![("d1", 0.9), ("d2", 0.8)];
//! let query = vec![0.1, 0.2, 0.3, 0.4];
//! let docs = vec![
//!     ("d1", vec![0.1, 0.2, 0.35, 0.45]),
//!     ("d2", vec![0.1, 0.2, 0.29, 0.39]),
//! ];
//!
//! let refined = matryoshka::refine(&candidates, &query, &docs, 2);
//! ```

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

    results.sort_by(|a, b| b.1.total_cmp(&a.1));

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
        let docs = vec![
            ("d1", vec![0.0, 0.0, 1.0, 0.0]),
            ("d2", vec![0.0]),
        ];

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
    }
}
