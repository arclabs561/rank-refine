//! `ColBERT`/PLAID late interaction scoring.
//!
//! `ColBERT` represents queries and documents as bags of token embeddings,
//! then scores using `MaxSim`: sum over query tokens of max similarity to any doc token.
//!
//! ## Token Pooling
//!
//! Token pooling reduces storage by clustering similar token embeddings and averaging them.
//! Research shows pool factors of 2-3 achieve 50-66% vector reduction with <1% quality loss.
//! See: <https://www.answer.ai/posts/colbert-pooling.html>
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::colbert;
//!
//! // Query: 2 tokens x 4 dims
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
//!
//! // Pool document tokens for storage efficiency
//! let pooled = colbert::pool_tokens(&docs[0].1, 2);
//! assert!(pooled.len() <= docs[0].1.len());
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
/// # Algorithm
///
/// Uses greedy agglomerative clustering: builds a similarity matrix, then
/// iteratively merges the most similar pair of clusters until reaching
/// the target count.
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

/// Pool tokens with protected indices (e.g., [CLS], [D] markers).
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

/// Rank documents with optional top_k limit.
#[must_use]
pub fn rank_with_top_k<I: Clone>(
    query: &[Vec<f32>],
    docs: &[(I, Vec<Vec<f32>>)],
    top_k: Option<usize>,
) -> Vec<(I, f32)> {
    let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();

    let mut results: Vec<(I, f32)> = docs
        .iter()
        .map(|(id, doc_tokens)| {
            let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(Vec::as_slice).collect();
            let score = simd::maxsim(&query_refs, &doc_refs);
            (id.clone(), score)
        })
        .collect();

    results.sort_by(|a, b| b.1.total_cmp(&a.1));

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
    refine_with_config(candidates, query, docs, RefineConfig::default().with_alpha(alpha))
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
    let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
    let alpha = config.alpha;

    let mut results: Vec<(I, f32)> = candidates
        .iter()
        .filter_map(|(id, orig_score)| {
            let doc_tokens = doc_map.get(id)?;
            let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(Vec::as_slice).collect();
            let maxsim_score = simd::maxsim(&query_refs, &doc_refs);
            let blended = (1.0 - alpha).mul_add(maxsim_score, alpha * orig_score);
            Some((id.clone(), blended))
        })
        .collect();

    results.sort_by(|a, b| b.1.total_cmp(&a.1));

    if let Some(k) = config.top_k {
        results.truncate(k);
    }

    results
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
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0]]),
            ("d2", vec![vec![0.0, 1.0]]),
        ];

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
        let docs = vec![
            ("d1", vec![vec![1.0, 0.0]]),
            ("d2", vec![vec![1.0, 0.0]]),
        ];

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
    }
}
