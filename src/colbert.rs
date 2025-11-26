//! ColBERT/PLAID late interaction scoring.
//!
//! ColBERT represents queries and documents as bags of token embeddings,
//! then scores using MaxSim: sum over query tokens of max similarity to any doc token.
//!
//! This module provides the MaxSim operator for reranking candidates
//! when you have token-level embeddings.
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::colbert;
//!
//! // Query: 2 tokens × 4 dims
//! let query = vec![
//!     vec![1.0, 0.0, 0.0, 0.0],
//!     vec![0.0, 1.0, 0.0, 0.0],
//! ];
//!
//! // Candidates with their token embeddings
//! let docs = vec![
//!     ("doc1", vec![vec![0.9, 0.1, 0.0, 0.0], vec![0.1, 0.9, 0.0, 0.0]]),
//!     ("doc2", vec![vec![0.5, 0.5, 0.0, 0.0]]),
//! ];
//!
//! let ranked = colbert::rank(&query, &docs);
//! assert_eq!(ranked[0].0, "doc1"); // better token alignment
//! ```

use crate::simd;
use std::hash::Hash;

/// Rank documents by MaxSim score against query tokens.
///
/// # Arguments
///
/// * `query` - Query token embeddings
/// * `docs` - Document id and token embeddings
///
/// # Returns
///
/// Documents sorted by descending MaxSim score.
pub fn rank<I: Clone + Eq + Hash>(
    query: &[Vec<f32>],
    docs: &[(I, Vec<Vec<f32>>)],
) -> Vec<(I, f32)> {
    let query_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();

    let mut results: Vec<(I, f32)> = docs
        .iter()
        .map(|(id, doc_tokens)| {
            let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(|v| v.as_slice()).collect();
            let score = simd::maxsim(&query_refs, &doc_refs);
            (id.clone(), score)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Refine candidates using MaxSim, blending with original scores.
///
/// # Arguments
///
/// * `candidates` - Initial retrieval results (id, score)
/// * `query` - Query token embeddings
/// * `docs` - Document token embeddings
/// * `alpha` - Weight for original score (0.0 = all MaxSim, 1.0 = all original)
pub fn refine<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    query: &[Vec<f32>],
    docs: &[(I, Vec<Vec<f32>>)],
    alpha: f32,
) -> Vec<(I, f32)> {
    use std::collections::HashMap;

    let doc_map: HashMap<&I, &Vec<Vec<f32>>> = docs.iter().map(|(id, toks)| (id, toks)).collect();
    let query_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();

    let mut results: Vec<(I, f32)> = candidates
        .iter()
        .filter_map(|(id, orig_score)| {
            let doc_tokens = doc_map.get(id)?;
            let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(|v| v.as_slice()).collect();
            let maxsim_score = simd::maxsim(&query_refs, &doc_refs);
            let blended = alpha * orig_score + (1.0 - alpha) * maxsim_score;
            Some((id.clone(), blended))
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]), // perfect match
            ("d2", vec![vec![0.5, 0.5]]),                  // mediocre
        ];

        let ranked = rank(&query, &docs);
        assert_eq!(ranked[0].0, "d1");
    }

    #[test]
    fn test_refine() {
        let candidates = vec![("d1", 0.5), ("d2", 0.9)]; // d2 has higher original
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0]]), // perfect MaxSim
            ("d2", vec![vec![0.0, 1.0]]), // zero MaxSim
        ];

        // With alpha=0 (all MaxSim), d1 wins
        let refined = refine(&candidates, &query, &docs, 0.0);
        assert_eq!(refined[0].0, "d1");

        // With alpha=1 (all original), d2 wins
        let refined = refine(&candidates, &query, &docs, 1.0);
        assert_eq!(refined[0].0, "d2");
    }
}

