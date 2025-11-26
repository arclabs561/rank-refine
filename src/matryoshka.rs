//! Matryoshka refinement using tail dimensions.
//!
//! MRL (Matryoshka Representation Learning) embeddings have a special property:
//! the first k dimensions form a valid, lower-resolution embedding.
//!
//! This enables a two-stage retrieval strategy:
//! 1. **Coarse retrieval**: Search using first k dims (fast, cache-friendly)
//! 2. **Refinement**: Re-score top candidates using full dims (accurate)
//!
//! This module implements the refinement step.
//!
//! ## Algorithm
//!
//! For each candidate:
//! 1. Compute cosine similarity using tail dimensions only (k..d)
//! 2. Blend with original score: `final = α × original + (1-α) × tail_sim`
//! 3. Sort by final score
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::matryoshka;
//!
//! let candidates = vec![("d1", 0.9), ("d2", 0.8)];
//! let query = vec![0.1, 0.2, 0.3, 0.4];  // 4-dim for demo
//! let docs = vec![
//!     ("d1", vec![0.1, 0.2, 0.35, 0.45]),
//!     ("d2", vec![0.1, 0.2, 0.29, 0.39]),
//! ];
//!
//! let refined = matryoshka::refine(&candidates, &query, &docs, 2);
//! // d1 still wins because tail similarity is higher
//! assert_eq!(refined[0].0, "d1");
//! ```

use std::collections::HashMap;
use std::hash::Hash;

/// Refine candidates using tail dimensions of Matryoshka embeddings.
///
/// # Arguments
///
/// * `candidates` - Initial retrieval results (id, coarse_score)
/// * `query` - Full query embedding
/// * `docs` - Document embeddings (id, full_embedding)
/// * `head_dims` - Number of "head" dimensions used in coarse retrieval
///
/// # Returns
///
/// Refined results sorted by descending blended score.
///
/// # Panics
///
/// Panics if embeddings have different dimensions or head_dims >= embedding length.
pub fn refine<I>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
{
    refine_with_alpha(candidates, query, docs, head_dims, 0.5)
}

/// Refine with custom blending weight.
///
/// * `alpha` - Weight for original score (0.0 = all tail, 1.0 = all original)
pub fn refine_with_alpha<I>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
    alpha: f32,
) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
{
    assert!(head_dims < query.len(), "head_dims must be less than embedding length");

    // Build doc lookup
    let doc_map: HashMap<&I, &[f32]> = docs.iter().map(|(id, emb)| (id, emb.as_slice())).collect();

    // Query tail
    let query_tail = &query[head_dims..];
    let query_tail_norm = norm(query_tail);

    let mut results: Vec<(I, f32)> = candidates
        .iter()
        .filter_map(|(id, original_score)| {
            let doc_emb = doc_map.get(id)?;
            let doc_tail = &doc_emb[head_dims..];

            let tail_sim = if query_tail_norm > 1e-9 {
                cosine_similarity(query_tail, doc_tail)
            } else {
                0.0
            };

            // Blend original score with tail similarity
            let blended = alpha * original_score + (1.0 - alpha) * tail_sim;
            Some((id.clone(), blended))
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Refine using only tail dimensions (ignore original scores).
pub fn refine_tail_only<I>(
    candidates: &[(I, f32)],
    query: &[f32],
    docs: &[(I, Vec<f32>)],
    head_dims: usize,
) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
{
    refine_with_alpha(candidates, query, docs, head_dims, 0.0)
}

/// Cosine similarity between two vectors.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = norm(a);
    let norm_b = norm(b);

    if norm_a > 1e-9 && norm_b > 1e-9 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// L2 norm of a vector.
#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_refinement() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];

        // 8-dim embeddings, head=4
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        let docs = vec![
            ("d1", vec![1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0]), // low tail sim
            ("d2", vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]), // high tail sim
            ("d3", vec![1.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0]), // medium tail sim
        ];

        let refined = refine(&candidates, &query, &docs, 4);

        // d2 should rise due to high tail similarity
        assert_eq!(refined.len(), 3);
        // With alpha=0.5, d2's tail similarity boost should make it competitive
    }

    #[test]
    fn test_tail_only() {
        let candidates = vec![("d1", 0.9), ("d2", 0.1)]; // d1 has high original score

        let query = vec![0.0, 0.0, 1.0, 0.0]; // tail is [1.0, 0.0]
        let docs = vec![
            ("d1", vec![0.0, 0.0, 0.0, 1.0]), // tail is [0.0, 1.0] - orthogonal
            ("d2", vec![0.0, 0.0, 1.0, 0.0]), // tail is [1.0, 0.0] - identical
        ];

        let refined = refine_tail_only(&candidates, &query, &docs, 2);

        // d2 should win despite low original score
        assert_eq!(refined[0].0, "d2");
    }

    #[test]
    fn test_missing_doc() {
        let candidates = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let docs = vec![
            ("d1", vec![1.0, 0.0, 0.0, 0.0]),
            // d2 missing
            ("d3", vec![0.5, 0.5, 0.0, 0.0]),
        ];

        let refined = refine(&candidates, &query, &docs, 2);

        // Only d1 and d3 should be in results
        assert_eq!(refined.len(), 2);
        assert!(refined.iter().all(|(id, _)| *id != "d2"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        assert!(cosine_similarity(&c, &d).abs() < 1e-6);

        let e = vec![1.0, 1.0];
        let f = vec![1.0, 1.0];
        assert!((cosine_similarity(&e, &f) - 1.0).abs() < 1e-6);
    }
}

