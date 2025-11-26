//! Matryoshka refinement using tail dimensions.
//!
//! MRL embeddings have a useful property: the first k dimensions form a valid
//! coarse embedding. This enables two-stage retrieval:
//!
//! 1. **Coarse**: Search using first k dims (fast, cache-friendly)
//! 2. **Refine**: Re-score top candidates using tail dims (accurate)
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

use crate::simd;
use std::collections::HashMap;
use std::hash::Hash;

/// Refine candidates using tail dimensions.
///
/// Uses `alpha=0.5` (equal blend of original score and tail similarity).
pub fn refine<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
) -> Vec<(I, f32)> {
    refine_with_alpha(candidates, query, docs, head_dims, 0.5)
}

/// Refine with custom blending weight.
///
/// * `alpha` - Weight for original score (0.0 = all tail, 1.0 = all original)
pub fn refine_with_alpha<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
    alpha: f32,
) -> Vec<(I, f32)> {
    assert!(head_dims < query.len(), "head_dims must be < embedding length");

    let doc_map: HashMap<&I, &[f32]> = docs.iter().map(|(id, emb)| (id, emb.as_slice())).collect();
    let query_tail = &query[head_dims..];

    let mut results: Vec<(I, f32)> = candidates
        .iter()
        .filter_map(|(id, orig)| {
            let doc_tail = &doc_map.get(id)?[head_dims..];
            let tail_sim = simd::cosine(query_tail, doc_tail);
            Some((id.clone(), alpha * orig + (1.0 - alpha) * tail_sim))
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Refine using only tail dimensions (ignore original scores).
pub fn refine_tail_only<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
) -> Vec<(I, f32)> {
    refine_with_alpha(candidates, query, docs, head_dims, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement() {
        let candidates = vec![("d1", 0.9), ("d2", 0.1)];
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let docs = vec![
            ("d1", vec![0.0, 0.0, 0.0, 1.0]), // orthogonal tail
            ("d2", vec![0.0, 0.0, 1.0, 0.0]), // identical tail
        ];

        let refined = refine_tail_only(&candidates, &query, &docs, 2);
        assert_eq!(refined[0].0, "d2");
    }
}
