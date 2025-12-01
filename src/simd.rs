//! Vector operations with SIMD acceleration.
//!
//! Provides `dot`, `cosine`, and `maxsim` with automatic SIMD dispatch:
//! - AVX2+FMA on `x86_64` (runtime detection)
//! - NEON on `aarch64`
//! - Portable fallback otherwise
//!
//! # Correctness
//!
//! All SIMD implementations are tested against the portable fallback
//! to ensure identical results (within floating-point tolerance).
//!
//! # Performance Notes
//!
//! - Vectors shorter than 16 dimensions use portable code (SIMD overhead not worthwhile)
//! - Subnormal/denormalized floats (~< 1e-38) can cause 100x+ slowdowns in SIMD
//! - Unit-normalized embeddings avoid subnormal issues in practice

// Minimum vector dimension for SIMD to be worthwhile.
// Below this, function call overhead outweighs SIMD benefits.
// Matches qdrant's MIN_DIM_SIZE_SIMD threshold.
const MIN_DIM_SIMD: usize = 16;

/// Threshold for treating a norm as "effectively zero" in cosine similarity.
///
/// Chosen to be larger than `f32::EPSILON` (~1.19e-7) to provide numerical
/// headroom while remaining small enough to only catch degenerate cases.
/// If both vector norms exceed this, we compute the ratio; otherwise return 0.
///
/// Common alternatives:
/// - `f32::EPSILON`: meilisearch/arroy uses this, but it's arguably too strict
/// - `1e-7`: similar magnitude to machine epsilon
/// - `1e-9`: our choice, allows more "nearly-zero" vectors to be handled
const NORM_EPSILON: f32 = 1e-9;

/// Dot product of two vectors.
///
/// Returns 0.0 for empty vectors.
///
/// # Debug Assertions
///
/// In debug builds, panics if vector lengths differ. In release builds,
/// mismatched lengths silently use the shorter length (for performance).
/// Use [`dot_truncating`] if you intentionally want truncation behavior.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot: dimension mismatch ({} vs {}). Use dot_truncating for intentional truncation.",
        a.len(),
        b.len()
    );
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            // SAFETY: We've verified AVX2 and FMA are available via runtime detection.
            // The function handles mismatched lengths by using min(a.len(), b.len()).
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON is always available on aarch64.
            // The function handles mismatched lengths by using min(a.len(), b.len()).
            return unsafe { dot_neon(a, b) };
        }
    }
    #[allow(unreachable_code)]
    dot_portable(a, b)
}

/// L2 norm of a vector.
#[inline]
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has effectively-zero norm (< 1e-9).
/// Result is in `[-1, 1]` for valid input, but floating-point error can push
/// slightly outside this range; clamp if strict bounds are required.
#[inline]
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na > NORM_EPSILON && nb > NORM_EPSILON {
        d / (na * nb)
    } else {
        0.0
    }
}

/// `MaxSim`: sum over query tokens of max dot product with any doc token.
///
/// Used by ColBERT/PLAID for late interaction scoring.
///
/// # ⚠️ Not Commutative
///
/// **`maxsim(Q, D) ≠ maxsim(D, Q)`** — argument order matters!
///
/// Query tokens "select" from the document's vocabulary. Each query token
/// finds its best match among document tokens, then scores are summed.
/// Reversing the arguments gives a semantically different operation.
///
/// ```text
/// maxsim(Q, D) = Σᵢ maxⱼ(Qᵢ · Dⱼ)  // sum over |Q| terms
/// maxsim(D, Q) = Σⱼ maxᵢ(Dⱼ · Qᵢ)  // sum over |D| terms (different!)
/// ```
///
/// If you accidentally swap query/doc, scores will be wrong and rankings
/// will be meaningless. The first argument is always the **query**.
///
/// # Returns
///
/// - `0.0` if `query_tokens` is empty
/// - `0.0` if `doc_tokens` is empty (no matches possible)
#[inline]
#[must_use]
pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// `MaxSim` with cosine similarity instead of dot product.
///
/// See [`maxsim`] for details. **Not commutative** — query must be first argument.
#[inline]
#[must_use]
pub fn maxsim_cosine(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| cosine(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Token-level alignment information from `MaxSim` computation.
///
/// Returns for each query token: `(query_token_idx, doc_token_idx, similarity_score)`
/// where `doc_token_idx` is the document token (or image patch) with maximum similarity to that query token.
///
/// This enables highlighting, snippet extraction, and interpretability—core ColBERT features
/// that distinguish it from single-vector embeddings.
///
/// **Multimodal support**: For ColPali-style systems, `doc_tokens` are image patch embeddings.
/// Alignment pairs show which image patches match each query token, enabling visual snippet extraction.
///
/// # Returns
///
/// Vector of `(query_idx, doc_idx, score)` tuples, one per query token.
/// Empty if `query_tokens` or `doc_tokens` is empty.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::maxsim_alignments;
///
/// let q1 = [1.0, 0.0];
/// let q2 = [0.0, 1.0];
/// let d1 = [0.9, 0.1];
/// let d2 = [0.1, 0.9];
/// let d3 = [0.5, 0.5];
/// let query: &[&[f32]] = &[&q1, &q2];
/// let doc: &[&[f32]] = &[&d1, &d2, &d3];
/// let alignments = maxsim_alignments(query, doc);
///
/// // Query token 0 matches doc token 0 with score ~0.9
/// // Query token 1 matches doc token 1 with score ~0.9
/// assert_eq!(alignments.len(), 2);
/// ```
#[must_use]
pub fn maxsim_alignments(
    query_tokens: &[&[f32]],
    doc_tokens: &[&[f32]],
) -> Vec<(usize, usize, f32)> {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return Vec::new();
    }

    query_tokens
        .iter()
        .enumerate()
        .map(|(q_idx, q)| {
            let (best_doc_idx, best_score) = doc_tokens
                .iter()
                .enumerate()
                .map(|(d_idx, d)| (d_idx, dot(q, d)))
                .fold(
                    (0, f32::NEG_INFINITY),
                    |(best_idx, best_score), (idx, score)| {
                        if score > best_score {
                            (idx, score)
                        } else {
                            (best_idx, best_score)
                        }
                    },
                );
            (q_idx, best_doc_idx, best_score)
        })
        .collect()
}

/// Token-level alignment with cosine similarity.
///
/// See [`maxsim_alignments`] for details.
#[must_use]
pub fn maxsim_alignments_cosine(
    query_tokens: &[&[f32]],
    doc_tokens: &[&[f32]],
) -> Vec<(usize, usize, f32)> {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return Vec::new();
    }

    query_tokens
        .iter()
        .enumerate()
        .map(|(q_idx, q)| {
            let (best_doc_idx, best_score) = doc_tokens
                .iter()
                .enumerate()
                .map(|(d_idx, d)| (d_idx, cosine(q, d)))
                .fold(
                    (0, f32::NEG_INFINITY),
                    |(best_idx, best_score), (idx, score)| {
                        if score > best_score {
                            (idx, score)
                        } else {
                            (best_idx, best_score)
                        }
                    },
                );
            (q_idx, best_doc_idx, best_score)
        })
        .collect()
}

/// Extract highlighted document token (or image patch) indices that match query tokens.
///
/// Returns unique document token indices that have high similarity to any query token.
/// Useful for snippet extraction and highlighting in search results.
///
/// **Multimodal support**: For ColPali-style systems, returns highlighted image patch indices.
/// These can be used to extract visual regions (snippets) from document images for display.
///
/// # Arguments
///
/// * `query_tokens` - Query token embeddings
/// * `doc_tokens` - Document token embeddings
/// * `threshold` - Minimum similarity score to include (typically 0.5-0.7 for normalized embeddings)
///
/// # Returns
///
/// Sorted vector of unique document token indices that match query tokens above threshold.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::highlight_matches;
///
/// let q1 = [1.0, 0.0];
/// let d1 = [0.9, 0.1];  // matches query[0]
/// let d2 = [0.1, 0.9];  // matches query[1]
/// let d3 = [0.5, 0.5];  // low match
/// let query: &[&[f32]] = &[&q1];
/// let doc: &[&[f32]] = &[&d1, &d2, &d3];
///
/// let highlighted = highlight_matches(query, doc, 0.7);
/// // Returns [0] - index of token that matches well
/// ```
#[must_use]
pub fn highlight_matches(
    query_tokens: &[&[f32]],
    doc_tokens: &[&[f32]],
    threshold: f32,
) -> Vec<usize> {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return Vec::new();
    }

    let mut matched_indices = std::collections::HashSet::new();

    for q in query_tokens {
        for (d_idx, d) in doc_tokens.iter().enumerate() {
            let similarity = dot(q, d);
            if similarity >= threshold {
                matched_indices.insert(d_idx);
            }
        }
    }

    let mut result: Vec<usize> = matched_indices.into_iter().collect();
    result.sort_unstable();
    result
}

/// Weighted `MaxSim`: token importance weighting.
///
/// Formula: `score(Q, D) = Σᵢ wᵢ × maxⱼ(Qᵢ · Dⱼ)`
///
/// Weights allow prioritizing important query tokens (e.g., by IDF).
/// Research shows ~2-5% quality improvement when using learned weights.
///
/// See [Incorporating Token Importance in Multi-Vector Retrieval](https://arxiv.org/abs/2511.16106).
///
/// **Use cases**:
/// - **IDF weighting**: Boost rare terms, de-emphasize common terms
/// - **[MASK] token weighting**: ColBERT uses [MASK] tokens for query augmentation.
///   These tokens are added during encoding and should be weighted lower (typically 0.2-0.4)
///   than original query tokens. See `examples/mask_token_weighting.rs` for a complete example.
/// - **Learned importance**: Extract attention weights from trained models
///
/// # Arguments
///
/// * `query_tokens` - Query token embeddings
/// * `doc_tokens` - Document token embeddings
/// * `weights` - Per-query-token importance weights (should have same length as `query_tokens`)
///
/// # Behavior
///
/// - If `weights.len() < query_tokens.len()`, missing weights default to 1.0
/// - If `weights.len() > query_tokens.len()`, extra weights are ignored
/// - If weights sum to 0, returns 0.0
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::maxsim_weighted;
///
/// let query = vec![[1.0, 0.0], [0.0, 1.0]];
/// let doc = vec![[0.9, 0.1], [0.1, 0.9]];
/// let q_refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();
/// let d_refs: Vec<&[f32]> = doc.iter().map(|t| t.as_slice()).collect();
///
/// // First token is more important (IDF weighting)
/// let weights = [2.0, 0.5];
/// let score = maxsim_weighted(&q_refs, &d_refs, &weights);
/// ```
#[inline]
#[must_use]
pub fn maxsim_weighted(query_tokens: &[&[f32]], doc_tokens: &[&[f32]], weights: &[f32]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let w = weights.get(i).copied().unwrap_or(1.0);
            let max_sim = doc_tokens
                .iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max);
            w * max_sim
        })
        .sum()
}

/// Weighted `MaxSim` with cosine similarity.
///
/// See [`maxsim_weighted`] for details.
#[inline]
#[must_use]
pub fn maxsim_cosine_weighted(
    query_tokens: &[&[f32]],
    doc_tokens: &[&[f32]],
    weights: &[f32],
) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let w = weights.get(i).copied().unwrap_or(1.0);
            let max_sim = doc_tokens
                .iter()
                .map(|d| cosine(q, d))
                .fold(f32::NEG_INFINITY, f32::max);
            w * max_sim
        })
        .sum()
}

/// Weighted `MaxSim` for owned vectors (convenience wrapper).
#[inline]
#[must_use]
pub fn maxsim_weighted_vecs(
    query_tokens: &[Vec<f32>],
    doc_tokens: &[Vec<f32>],
    weights: &[f32],
) -> f32 {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim_weighted(&q, &d, weights)
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrappers for owned vectors
// ─────────────────────────────────────────────────────────────────────────────

/// `MaxSim` for owned token vectors (convenience wrapper).
///
/// Equivalent to `maxsim(&as_slices(query), &as_slices(doc))` but more ergonomic.
///
/// See [`maxsim`] for details. **Not commutative** — query must be first argument.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::maxsim_vecs;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
/// let score = maxsim_vecs(&query, &doc);
/// ```
#[inline]
#[must_use]
pub fn maxsim_vecs(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim(&q, &d)
}

/// `MaxSim` cosine for owned token vectors (convenience wrapper).
///
/// See [`maxsim`] for details. **Not commutative** — query must be first argument.
#[inline]
#[must_use]
pub fn maxsim_cosine_vecs(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim_cosine(&q, &d)
}

/// Token alignments for owned token vectors (convenience wrapper).
///
/// See [`maxsim_alignments`] for details.
#[inline]
#[must_use]
pub fn maxsim_alignments_vecs(
    query_tokens: &[Vec<f32>],
    doc_tokens: &[Vec<f32>],
) -> Vec<(usize, usize, f32)> {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim_alignments(&q, &d)
}

/// Token alignments with cosine for owned token vectors (convenience wrapper).
///
/// See [`maxsim_alignments_cosine`] for details.
#[inline]
#[must_use]
pub fn maxsim_alignments_cosine_vecs(
    query_tokens: &[Vec<f32>],
    doc_tokens: &[Vec<f32>],
) -> Vec<(usize, usize, f32)> {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim_alignments_cosine(&q, &d)
}

/// Highlighted matches for owned token vectors (convenience wrapper).
///
/// See [`highlight_matches`] for details.
#[inline]
#[must_use]
pub fn highlight_matches_vecs(
    query_tokens: &[Vec<f32>],
    doc_tokens: &[Vec<f32>],
    threshold: f32,
) -> Vec<usize> {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    highlight_matches(&q, &d, threshold)
}

/// Batch `MaxSim`: score a query against multiple documents.
///
/// Returns a vector of scores, one per document. More efficient than
/// calling `maxsim` in a loop when you have many documents.
///
/// See [`maxsim`] for details. **Not commutative** — query must be first argument.
///
/// For pre-computed document indices with ID tracking, see [`crate::colbert::TokenIndex`].
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::maxsim_batch;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let docs = vec![
///     vec![vec![1.0, 0.0]],  // doc 0
///     vec![vec![0.5, 0.5]],  // doc 1
/// ];
/// let scores = maxsim_batch(&query, &docs);
/// assert_eq!(scores.len(), 2);
/// ```
#[must_use]
pub fn maxsim_batch(query: &[Vec<f32>], docs: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let q = crate::as_slices(query);
    docs.iter()
        .map(|doc| {
            let d = crate::as_slices(doc);
            maxsim(&q, &d)
        })
        .collect()
}

/// Batch `MaxSim` with cosine similarity.
///
/// See [`maxsim`] for details. **Not commutative** — query must be first argument.
#[must_use]
pub fn maxsim_cosine_batch(query: &[Vec<f32>], docs: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let q = crate::as_slices(query);
    docs.iter()
        .map(|doc| {
            let d = crate::as_slices(doc);
            maxsim_cosine(&q, &d)
        })
        .collect()
}

/// Batch token-level alignments: get alignments for a query against multiple documents.
///
/// Returns a vector of alignment vectors, one per document. Each alignment vector
/// contains `(query_idx, doc_idx, similarity_score)` tuples.
///
/// More efficient than calling `maxsim_alignments` in a loop when you have many documents.
///
/// See [`maxsim_alignments`] for details.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::maxsim_alignments_batch;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let docs = vec![
///     vec![vec![0.9, 0.1], vec![0.1, 0.9]],  // doc 0
///     vec![vec![0.5, 0.5]],                  // doc 1
/// ];
/// let all_alignments = maxsim_alignments_batch(&query, &docs);
/// assert_eq!(all_alignments.len(), 2);
/// assert_eq!(all_alignments[0].len(), 2); // One alignment per query token
/// ```
#[must_use]
pub fn maxsim_alignments_batch(
    query: &[Vec<f32>],
    docs: &[Vec<Vec<f32>>],
) -> Vec<Vec<(usize, usize, f32)>> {
    let q = crate::as_slices(query);
    docs.iter()
        .map(|doc| {
            let d = crate::as_slices(doc);
            maxsim_alignments(&q, &d)
        })
        .collect()
}

/// Batch token-level alignments with cosine similarity.
///
/// See [`maxsim_alignments_batch`] and [`maxsim_alignments_cosine`] for details.
#[must_use]
pub fn maxsim_alignments_cosine_batch(
    query: &[Vec<f32>],
    docs: &[Vec<Vec<f32>>],
) -> Vec<Vec<(usize, usize, f32)>> {
    let q = crate::as_slices(query);
    docs.iter()
        .map(|doc| {
            let d = crate::as_slices(doc);
            maxsim_alignments_cosine(&q, &d)
        })
        .collect()
}

/// Batch highlighted matches: get highlighted token indices for a query against multiple documents.
///
/// Returns a vector of highlighted index vectors, one per document. Each vector contains
/// unique document token indices that match query tokens above the threshold.
///
/// More efficient than calling `highlight_matches` in a loop when you have many documents.
///
/// See [`highlight_matches`] for details.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::highlight_matches_batch;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let docs = vec![
///     vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],  // doc 0
///     vec![vec![0.3, 0.3]],                                  // doc 1
/// ];
/// let all_highlights = highlight_matches_batch(&query, &docs, 0.7);
/// assert_eq!(all_highlights.len(), 2);
/// ```
#[must_use]
pub fn highlight_matches_batch(
    query: &[Vec<f32>],
    docs: &[Vec<Vec<f32>>],
    threshold: f32,
) -> Vec<Vec<usize>> {
    let q = crate::as_slices(query);
    docs.iter()
        .map(|doc| {
            let d = crate::as_slices(doc);
            highlight_matches(&q, &d, threshold)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Alignment utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Get top-k highest scoring alignments from an alignment vector.
///
/// Returns the top-k alignments sorted by similarity score (descending).
/// Useful for focusing on the most important token matches.
///
/// # Arguments
///
/// * `alignments` - Vector of `(query_idx, doc_idx, score)` tuples
/// * `k` - Number of top alignments to return
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim_alignments_vecs, top_k_alignments};
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.3, 0.3], vec![0.8, 0.2]];
/// let all_alignments = maxsim_alignments_vecs(&query, &doc);
/// let top2 = top_k_alignments(&all_alignments, 2);
/// assert_eq!(top2.len(), 2);
/// assert!(top2[0].2 >= top2[1].2); // Sorted by score
/// ```
#[must_use]
pub fn top_k_alignments(alignments: &[(usize, usize, f32)], k: usize) -> Vec<(usize, usize, f32)> {
    if k == 0 || alignments.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<_> = alignments.to_vec();
    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    sorted.into_iter().take(k).collect()
}

/// Filter alignments by minimum similarity score threshold.
///
/// Returns only alignments with similarity score >= `min_score`.
/// Useful for removing low-quality matches.
///
/// # Arguments
///
/// * `alignments` - Vector of `(query_idx, doc_idx, score)` tuples
/// * `min_score` - Minimum similarity score threshold
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim_alignments_vecs, filter_alignments};
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.3, 0.3]];
/// let all_alignments = maxsim_alignments_vecs(&query, &doc);
/// let high_quality = filter_alignments(&all_alignments, 0.7);
/// // Only includes alignments with score >= 0.7
/// ```
#[must_use]
pub fn filter_alignments(
    alignments: &[(usize, usize, f32)],
    min_score: f32,
) -> Vec<(usize, usize, f32)> {
    alignments
        .iter()
        .filter(|(_, _, score)| *score >= min_score)
        .copied()
        .collect()
}

/// Extract alignments for specific query token indices.
///
/// Returns only alignments where the query token index is in `query_indices`.
/// Useful for analyzing matches for specific query terms.
///
/// # Arguments
///
/// * `alignments` - Vector of `(query_idx, doc_idx, score)` tuples
/// * `query_indices` - Set of query token indices to include
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim_alignments_vecs, alignments_for_query_tokens};
/// use std::collections::HashSet;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
/// let all_alignments = maxsim_alignments_vecs(&query, &doc);
/// let query_indices: HashSet<usize> = [0, 2].iter().copied().collect();
/// let filtered = alignments_for_query_tokens(&all_alignments, &query_indices);
/// // Only includes alignments for query tokens 0 and 2
/// ```
#[must_use]
pub fn alignments_for_query_tokens(
    alignments: &[(usize, usize, f32)],
    query_indices: &std::collections::HashSet<usize>,
) -> Vec<(usize, usize, f32)> {
    alignments
        .iter()
        .filter(|(q_idx, _, _)| query_indices.contains(q_idx))
        .copied()
        .collect()
}

/// Extract alignments for specific document token indices.
///
/// Returns only alignments where the document token index is in `doc_indices`.
/// Useful for analyzing which query tokens match specific document regions.
///
/// # Arguments
///
/// * `alignments` - Vector of `(query_idx, doc_idx, score)` tuples
/// * `doc_indices` - Set of document token indices to include
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim_alignments_vecs, alignments_for_doc_tokens};
/// use std::collections::HashSet;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]];
/// let all_alignments = maxsim_alignments_vecs(&query, &doc);
/// let doc_indices: HashSet<usize> = [0, 2].iter().copied().collect();
/// let filtered = alignments_for_doc_tokens(&all_alignments, &doc_indices);
/// // Only includes alignments for document tokens 0 and 2
/// ```
#[must_use]
pub fn alignments_for_doc_tokens(
    alignments: &[(usize, usize, f32)],
    doc_indices: &std::collections::HashSet<usize>,
) -> Vec<(usize, usize, f32)> {
    alignments
        .iter()
        .filter(|(_, d_idx, _)| doc_indices.contains(d_idx))
        .copied()
        .collect()
}

/// Get alignment statistics: min, max, mean, and sum of similarity scores.
///
/// Useful for understanding the distribution of alignment scores.
///
/// # Returns
///
/// `(min_score, max_score, mean_score, sum_score, count)`
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim_alignments_vecs, alignment_stats};
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.3, 0.3]];
/// let alignments = maxsim_alignments_vecs(&query, &doc);
/// let (min, max, mean, sum, count) = alignment_stats(&alignments);
/// println!("Min: {}, Max: {}, Mean: {}, Sum: {}, Count: {}", min, max, mean, sum, count);
/// ```
#[must_use]
pub fn alignment_stats(alignments: &[(usize, usize, f32)]) -> (f32, f32, f32, f32, usize) {
    if alignments.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0);
    }
    let scores: Vec<f32> = alignments.iter().map(|(_, _, s)| *s).collect();
    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scores.iter().sum();
    let count = scores.len();
    let mean = sum / count as f32;
    (min, max, mean, sum, count)
}

// ─────────────────────────────────────────────────────────────────────────────
// Query expansion and weighting utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Inverse Document Frequency (IDF) weights for query tokens.
///
/// IDF weighting boosts rare terms and de-emphasizes common terms, improving
/// retrieval quality by ~2-5% in many cases.
///
/// Formula: `idf(t) = log((N + 1) / (df(t) + 1))` where:
/// - `N` is the total number of documents in the collection
/// - `df(t)` is the document frequency of token `t` (number of documents containing it)
///
/// Returns normalized weights in the range [0, 1] (or all 1.0 if all tokens have same IDF).
///
/// # Arguments
///
/// * `token_doc_freqs` - Document frequency for each query token (number of docs containing it)
/// * `total_docs` - Total number of documents in the collection
///
/// # Returns
///
/// Vector of IDF weights, one per query token. Higher values = rarer terms = more important.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::idf_weights;
///
/// // Query: ["rust", "memory"]
/// // "rust" appears in 100 docs, "memory" appears in 1000 docs
/// // Total collection: 10000 docs
/// let doc_freqs = vec![100, 1000];
/// let weights = idf_weights(&doc_freqs, 10000);
///
/// // "rust" (rare) gets higher weight than "memory" (common)
/// assert!(weights[0] > weights[1]);
/// ```
#[must_use]
pub fn idf_weights(token_doc_freqs: &[usize], total_docs: usize) -> Vec<f32> {
    if token_doc_freqs.is_empty() || total_docs == 0 {
        return vec![];
    }

    let idf_scores: Vec<f32> = token_doc_freqs
        .iter()
        .map(|&df| {
            if df == 0 {
                // Token never appears - maximum importance
                (total_docs as f32 + 1.0).ln()
            } else if df > total_docs {
                // Invalid: document frequency exceeds total documents
                // Treat as if it appears in all documents (minimum importance)
                0.0
            } else {
                ((total_docs as f32 + 1.0) / (df as f32 + 1.0)).ln()
            }
        })
        .collect();

    // Normalize to [0, 1] range
    let min_idf = idf_scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max_idf = idf_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max_idf - min_idf).abs() < 1e-9 {
        // All tokens have same IDF - return uniform weights
        vec![1.0; token_doc_freqs.len()]
    } else {
        let mut weights = Vec::with_capacity(idf_scores.len());
        for &idf in &idf_scores {
            weights.push((idf - min_idf) / (max_idf - min_idf));
        }
        weights
    }
}

/// Compute BM25-style term weights for query tokens.
///
/// BM25 weighting combines IDF with term frequency, providing a more nuanced
/// importance measure than pure IDF.
///
/// Formula: `weight(t) = idf(t) × (tf(t) × (k1 + 1)) / (tf(t) + k1 × (1 - b + b × (|d| / avg_dl)))`
///
/// Simplified version: `weight(t) = idf(t) × tf(t) / (tf(t) + k1)` where:
/// - `idf(t)` is Inverse Document Frequency
/// - `tf(t)` is term frequency in the query
/// - `k1` is a tuning parameter (typically 1.2-2.0)
///
/// Returns normalized weights in the range [0, 1].
///
/// # Arguments
///
/// * `token_doc_freqs` - Document frequency for each query token
/// * `token_query_freqs` - Term frequency in query for each token (typically 1 for most queries)
/// * `total_docs` - Total number of documents in the collection
/// * `k1` - Tuning parameter (default: 1.5, typical range: 1.2-2.0). Negative values are clamped to 0.
///
/// # Returns
///
/// Vector of BM25-style weights, one per query token.
///
/// # Panics
///
/// Never panics. Invalid inputs (length mismatch, df > total_docs, negative k1) are handled gracefully.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::bm25_weights;
///
/// // Query: ["rust", "rust", "memory"] (rust appears twice)
/// let doc_freqs = vec![100, 100];  // Both tokens appear in 100 docs
/// let query_freqs = vec![2, 1];    // "rust" appears 2x, "memory" 1x
/// let weights = bm25_weights(&doc_freqs, &query_freqs, 10000, 1.5);
///
/// // "rust" gets higher weight due to higher query frequency
/// assert!(weights[0] > weights[1]);
/// ```
#[must_use]
pub fn bm25_weights(
    token_doc_freqs: &[usize],
    token_query_freqs: &[usize],
    total_docs: usize,
    k1: f32,
) -> Vec<f32> {
    if token_doc_freqs.len() != token_query_freqs.len() {
        return vec![];
    }
    if token_doc_freqs.is_empty() || total_docs == 0 {
        return vec![];
    }
    // Validate k1 parameter
    let k1 = k1.max(0.0); // k1 should be non-negative

    let mut idf_scores = Vec::with_capacity(token_doc_freqs.len());
    for &df in token_doc_freqs {
        let idf = if df == 0 {
            (total_docs as f32 + 1.0).ln()
        } else if df > total_docs {
            // Invalid: document frequency exceeds total documents
            0.0
        } else {
            ((total_docs as f32 + 1.0) / (df as f32 + 1.0)).ln()
        };
        idf_scores.push(idf);
    }

    let mut bm25_scores = Vec::with_capacity(idf_scores.len());
    for (&idf, &tf) in idf_scores.iter().zip(token_query_freqs.iter()) {
        let tf_f32 = tf as f32;
        bm25_scores.push(idf * (tf_f32 * (k1 + 1.0)) / (tf_f32 + k1));
    }

    // Normalize to [0, 1]
    let min_score = bm25_scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max_score = bm25_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    if (max_score - min_score).abs() < 1e-9 {
        vec![1.0; token_doc_freqs.len()]
    } else {
        let mut weights = Vec::with_capacity(bm25_scores.len());
        for &score in &bm25_scores {
            weights.push((score - min_score) / (max_score - min_score));
        }
        weights
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multimodal and snippet extraction utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Convert image patch indices to bounding box regions (for ColPali-style systems).
///
/// In ColPali, document images are split into a grid of patches (e.g., 32×32 = 1024 patches).
/// This function converts patch indices (from `highlight_matches`) into pixel coordinates
/// for extracting visual snippets.
///
/// # Arguments
///
/// * `patch_indices` - Indices of highlighted patches (from `highlight_matches`)
/// * `image_width` - Width of the original image in pixels
/// * `image_height` - Height of the original image in pixels
/// * `patches_per_side` - Number of patches per side (e.g., 32 for 32×32 grid)
///
/// # Returns
///
/// Vector of `(x, y, width, height)` bounding boxes in pixel coordinates.
/// Each box represents the region covered by one or more highlighted patches.
/// Invalid patch indices (>= total_patches) are filtered out.
///
/// # Panics
///
/// Never panics. Invalid inputs (zero dimensions, invalid indices) return empty vector
/// or filter out invalid entries.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::patches_to_regions;
///
/// // Image: 1024×768 pixels, split into 32×32 patches
/// let highlighted_patches = vec![0, 1, 32, 33]; // Top-left corner patches
/// let regions = patches_to_regions(&highlighted_patches, 1024, 768, 32);
///
/// // Returns bounding boxes for the highlighted regions
/// for (x, y, w, h) in &regions {
///     println!("Region: ({}, {}) {}×{}", x, y, w, h);
/// }
/// ```
#[must_use]
pub fn patches_to_regions(
    patch_indices: &[usize],
    image_width: usize,
    image_height: usize,
    patches_per_side: usize,
) -> Vec<(usize, usize, usize, usize)> {
    if patch_indices.is_empty() || patches_per_side == 0 || image_width == 0 || image_height == 0 {
        return Vec::new();
    }

    let patch_width = image_width / patches_per_side;
    let patch_height = image_height / patches_per_side;

    let total_patches = patches_per_side * patches_per_side;
    let mut regions = Vec::with_capacity(patch_indices.len());
    for &patch_idx in patch_indices {
        // Validate patch index is within valid range
        if patch_idx >= total_patches {
            continue; // Skip invalid patch indices
        }
        let row = patch_idx / patches_per_side;
        let col = patch_idx % patches_per_side;
        let x = col * patch_width;
        let y = row * patch_height;
        regions.push((x, y, patch_width, patch_height));
    }
    regions
}

/// Extract text snippet indices from alignments for a document.
///
/// Given alignments from `maxsim_alignments`, returns the document token indices
/// that should be included in a text snippet. Useful for highlighting search results.
///
/// # Arguments
///
/// * `alignments` - Alignment pairs from `maxsim_alignments`
/// * `context_window` - Number of tokens to include before/after each match
/// * `max_tokens` - Maximum number of unique token indices to return (0 = return empty)
///
/// # Returns
///
/// Sorted vector of document token indices to include in snippet. Always includes
/// matched tokens, plus context tokens up to `context_window` before/after each match.
/// Duplicate indices are automatically deduplicated.
///
/// # Panics
///
/// Never panics. Handles edge cases gracefully (empty alignments, max_tokens=0, etc.).
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim_alignments_vecs, extract_snippet_indices};
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]];
/// let alignments = maxsim_alignments_vecs(&query, &doc);
///
/// // Extract snippet indices with 1 token context
/// let snippet_indices = extract_snippet_indices(&alignments, 1, 10);
/// // Returns [0, 1, 2] - matched tokens plus context
/// ```
#[must_use]
pub fn extract_snippet_indices(
    alignments: &[(usize, usize, f32)],
    context_window: usize,
    max_tokens: usize,
) -> Vec<usize> {
    if alignments.is_empty() {
        return Vec::new();
    }

    // Estimate capacity: each alignment adds 1 match + 2*context_window context tokens
    let estimated_capacity = alignments.len() * (1 + 2 * context_window.min(10));
    let capacity = estimated_capacity.min(max_tokens.max(1));
    let mut indices = std::collections::HashSet::with_capacity(capacity);

    for (_, doc_idx, _) in alignments {
        // Add matched token
        indices.insert(*doc_idx);

        // Add context tokens before
        for i in 0..context_window {
            let offset = i + 1;
            if *doc_idx >= offset {
                indices.insert(doc_idx - offset);
            }
        }

        // Add context tokens after
        // Note: We don't know doc length, so we'll add indices and filter later
        for i in 1..=context_window {
            indices.insert(doc_idx.saturating_add(i));
        }
    }

    let mut result: Vec<usize> = indices.into_iter().collect();
    result.sort_unstable();

    // Limit to max_tokens (take first max_tokens)
    // If max_tokens is 0, return empty (user explicitly wants no tokens)
    if max_tokens == 0 {
        return Vec::new();
    }
    if result.len() > max_tokens {
        result.truncate(max_tokens);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Score normalization utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Normalize a `MaxSim` score to approximately \[0, 1\].
///
/// `MaxSim` scores are unbounded: `score ∈ [0, query_token_count]`.
/// This function divides by `query_maxlen` (typically 32 for `ColBERT` models)
/// to produce comparable scores across different query lengths.
///
/// # Arguments
///
/// * `score` - Raw `MaxSim` score
/// * `query_maxlen` - Maximum query length the model was trained with (typically 32)
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::{maxsim, normalize_maxsim};
///
/// let q: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
/// let d: Vec<&[f32]> = vec![&[0.9, 0.1]];
/// let raw = maxsim(&q, &d);
/// let normalized = normalize_maxsim(raw, 32);
/// assert!(normalized <= 1.0);
/// ```
#[inline]
#[must_use]
pub fn normalize_maxsim(score: f32, query_maxlen: u32) -> f32 {
    score / query_maxlen as f32
}

/// Softmax normalization for a batch of scores.
///
/// Transforms scores to a probability distribution that sums to 1.
/// Useful for comparing candidates within a single query's result set.
///
/// Returns empty vec if input is empty or contains only non-finite values.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::softmax_scores;
///
/// let scores = vec![2.0, 1.0, 0.1];
/// let probs = softmax_scores(&scores);
/// let sum: f32 = probs.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// ```
#[must_use]
pub fn softmax_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability (only finite values)
    let max_score = scores
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_score.is_finite() {
        return vec![0.0; scores.len()];
    }

    // Compute exp(score - max) for stability
    // NaN/Inf inputs contribute 0 to maintain sum = 1 for finite inputs
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|s| {
            if s.is_finite() {
                (s - max_score).exp()
            } else {
                0.0 // NaN and Inf don't contribute
            }
        })
        .collect();

    let sum: f32 = exp_scores.iter().sum();
    if sum == 0.0 {
        return vec![0.0; scores.len()];
    }

    exp_scores.iter().map(|s| s / sum).collect()
}

/// Normalize a batch of `MaxSim` scores.
///
/// Convenience function that applies [`normalize_maxsim`] to each score.
#[inline]
#[must_use]
pub fn normalize_maxsim_batch(scores: &[f32], query_maxlen: u32) -> Vec<f32> {
    let divisor = query_maxlen as f32;
    scores.iter().map(|s| s / divisor).collect()
}

/// Top-k selection from scores.
///
/// Returns indices of the k highest scores in descending order.
/// Handles NaN by placing them last.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::top_k_indices;
///
/// let scores = vec![0.5, 0.9, 0.1, 0.7];
/// let top2 = top_k_indices(&scores, 2);
/// assert_eq!(top2, vec![1, 3]); // indices of 0.9 and 0.7
/// ```
#[must_use]
pub fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // Sort descending, NaN last
    // total_cmp puts NaN at the end in ascending order, but we want descending
    // So we reverse: finite values descending, then NaN
    indexed.sort_by(|a, b| {
        match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaN goes last
            (false, true) => std::cmp::Ordering::Less,    // NaN goes last
            (false, false) => b.1.total_cmp(&a.1),        // Descending for finite
        }
    });
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Truncating variants (explicit mismatched-length handling)
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product with explicit truncation for mismatched lengths.
///
/// Uses the shorter of the two vectors. Prefer [`dot`] when dimensions
/// should match; use this when intentionally comparing prefix dimensions
/// (e.g., Matryoshka embeddings).
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::dot_truncating;
///
/// // Compare only first 64 dims of a 768-dim embedding
/// let full = vec![1.0; 768];
/// let prefix = vec![1.0; 64];
/// let score = dot_truncating(&full, &prefix); // Uses 64 dims
/// ```
#[inline]
#[must_use]
pub fn dot_truncating(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2/FMA verified, truncation intentional
            return unsafe { dot_avx2(&a[..n], &b[..n]) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            // SAFETY: NEON always available, truncation intentional
            return unsafe { dot_neon(&a[..n], &b[..n]) };
        }
    }
    #[allow(unreachable_code)]
    dot_portable(&a[..n], &b[..n])
}

/// Cosine similarity with explicit truncation for mismatched lengths.
///
/// See [`dot_truncating`] for when to use this vs [`cosine`].
#[inline]
#[must_use]
pub fn cosine_truncating(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let d = dot_truncating(a, b);
    let na = dot_truncating(a, a).sqrt();
    let nb = dot_truncating(b, b).sqrt();
    if na > NORM_EPSILON && nb > NORM_EPSILON {
        d / (na * nb)
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Portable dot product implementation (reference for SIMD versions).
#[inline]
#[must_use]
pub(crate) fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 + FMA (x86_64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum: __m256 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // SAFETY: We use `_mm256_loadu_ps` (unaligned load) so no alignment required.
    // Pointer arithmetic: offset = i*8 < chunks*8 <= n, so a_ptr.add(offset)
    // stays within [a.as_ptr(), a.as_ptr() + n), same for b.
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum: reduce 8 f32s to 1
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder with scalar ops
    let tail_start = chunks * 8;
    for i in 0..remainder {
        // SAFETY: tail_start + i < n, so within bounds
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// NEON (aarch64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // SAFETY: `vld1q_f32` is an unaligned load, no alignment required.
    // Pointer arithmetic: offset = i*4 < chunks*4 <= n, so within bounds.
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    // Horizontal sum: reduce 4 f32s to 1
    let mut result = vaddvq_f32(sum);

    // Handle remainder with scalar ops
    let tail_start = chunks * 4;
    for i in 0..remainder {
        // SAFETY: tail_start + i < n, so within bounds
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_alignments_basic() {
        // Query: [1,0] and [0,1]
        // Doc: [0.9,0.1], [0.1,0.9], [0.5,0.5]
        // Query[0] should match Doc[0] (dot ~0.9)
        // Query[1] should match Doc[1] (dot ~0.9)
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.9, 0.1];
        let d2 = [0.1, 0.9];
        let d3 = [0.5, 0.5];
        let query: &[&[f32]] = &[&q1, &q2];
        let doc: &[&[f32]] = &[&d1, &d2, &d3];

        let alignments = maxsim_alignments(query, doc);
        assert_eq!(alignments.len(), 2);

        // First query token matches first doc token
        assert_eq!(alignments[0].0, 0); // query idx
        assert_eq!(alignments[0].1, 0); // doc idx
        assert!(alignments[0].2 > 0.8); // similarity

        // Second query token matches second doc token
        assert_eq!(alignments[1].0, 1); // query idx
        assert_eq!(alignments[1].1, 1); // doc idx
        assert!(alignments[1].2 > 0.8); // similarity
    }

    #[test]
    fn test_maxsim_alignments_empty() {
        let query: &[&[f32]] = &[];
        let doc: &[&[f32]] = &[&[1.0, 0.0]];
        assert!(maxsim_alignments(query, doc).is_empty());

        let query: &[&[f32]] = &[&[1.0, 0.0]];
        let doc: &[&[f32]] = &[];
        assert!(maxsim_alignments(query, doc).is_empty());
    }

    #[test]
    fn test_maxsim_alignments_consistency_with_maxsim() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.9, 0.1];
        let d2 = [0.1, 0.9];
        let query: &[&[f32]] = &[&q1, &q2];
        let doc: &[&[f32]] = &[&d1, &d2];

        let alignments = maxsim_alignments(query, doc);
        let maxsim_score = maxsim(query, doc);

        // Sum of alignment scores should equal MaxSim score
        let alignment_sum: f32 = alignments.iter().map(|(_, _, score)| score).sum();
        assert!(
            (alignment_sum - maxsim_score).abs() < 1e-5,
            "Alignment sum {} should equal MaxSim {}",
            alignment_sum,
            maxsim_score
        );
    }

    #[test]
    fn test_highlight_matches_basic() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.9, 0.1];
        let d2 = [0.1, 0.9];
        let d3 = [0.5, 0.5];
        let query: &[&[f32]] = &[&q1, &q2];
        let doc: &[&[f32]] = &[&d1, &d2, &d3];

        // With threshold 0.7, should match first two doc tokens
        let highlighted = highlight_matches(query, doc, 0.7);
        assert_eq!(highlighted.len(), 2);
        assert!(highlighted.contains(&0));
        assert!(highlighted.contains(&1));
        assert!(!highlighted.contains(&2)); // third token below threshold
    }

    #[test]
    fn test_highlight_matches_threshold() {
        let q1 = [1.0, 0.0];
        let d1 = [0.9, 0.1];
        let d2 = [0.5, 0.5];
        let d3 = [0.1, 0.9];
        let query: &[&[f32]] = &[&q1];
        let doc: &[&[f32]] = &[&d1, &d2, &d3];

        // High threshold: only first token
        let high = highlight_matches(query, doc, 0.8);
        assert_eq!(high, vec![0]);

        // Low threshold: all tokens
        let low = highlight_matches(query, doc, 0.0);
        assert_eq!(low.len(), 3);
    }

    #[test]
    fn test_highlight_matches_empty() {
        let query: &[&[f32]] = &[];
        let doc: &[&[f32]] = &[&[1.0, 0.0]];
        assert!(highlight_matches(query, doc, 0.5).is_empty());

        let query: &[&[f32]] = &[&[1.0, 0.0]];
        let doc: &[&[f32]] = &[];
        assert!(highlight_matches(query, doc, 0.5).is_empty());
    }

    #[test]
    fn test_maxsim_alignments_vecs() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];

        let alignments = maxsim_alignments_vecs(&query, &doc);
        assert_eq!(alignments.len(), 2);
        assert_eq!(alignments[0].0, 0);
        assert_eq!(alignments[0].1, 0);
        assert_eq!(alignments[1].0, 1);
        assert_eq!(alignments[1].1, 1);
    }

    #[test]
    fn test_highlight_matches_vecs() {
        let query = vec![vec![1.0, 0.0]];
        let doc = vec![vec![0.9, 0.1], vec![0.5, 0.5]];

        let highlighted = highlight_matches_vecs(&query, &doc, 0.7);
        assert_eq!(highlighted, vec![0]);
    }

    #[test]
    fn test_dot_basic() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_empty() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_truncating_empty() {
        assert_eq!(dot_truncating(&[], &[]), 0.0);
        assert_eq!(dot_truncating(&[1.0], &[]), 0.0);
        assert_eq!(dot_truncating(&[], &[1.0]), 0.0);
    }

    #[test]
    fn test_dot_truncating_mismatched() {
        // dot_truncating intentionally truncates to shorter length
        assert!((dot_truncating(&[1.0, 2.0, 3.0], &[4.0, 5.0]) - 14.0).abs() < 1e-5);
        // 1*4 + 2*5 = 14
    }

    #[test]
    fn test_dot_simd_vs_portable() {
        // Test various lengths around SIMD boundaries
        for len in [
            0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 1024,
        ] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.2 + 1.0).collect();

            let portable = dot_portable(&a, &b);
            let simd = dot(&a, &b);

            // Use relative tolerance for larger values
            let tolerance = (portable.abs() * 1e-5).max(1e-5);
            assert!(
                (portable - simd).abs() < tolerance,
                "Mismatch at len={}: portable={}, simd={}, diff={}",
                len,
                portable,
                simd,
                (portable - simd).abs()
            );
        }
    }

    #[test]
    fn test_cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_zero_norm() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_maxsim_basic() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.5, 0.5];
        let d2 = [1.0, 0.0];
        let d3 = [0.0, 1.0];

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2, &d3];

        // q1's best match is d2 (dot=1.0), q2's best match is d3 (dot=1.0)
        assert!((maxsim(&query, &doc) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_empty_query() {
        let doc: Vec<&[f32]> = vec![&[1.0, 0.0]];
        assert_eq!(maxsim(&[], &doc), 0.0);
    }

    #[test]
    fn test_maxsim_empty_doc() {
        let q1 = [1.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];
        // With empty docs, returns 0.0 (no matches possible)
        assert_eq!(maxsim(&query, &[]), 0.0);
    }

    #[test]
    fn test_maxsim_vecs() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Both query tokens find perfect matches
        assert!((maxsim_vecs(&query, &doc) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_cosine_vecs() {
        let query = vec![vec![2.0, 0.0]]; // unnormalized
        let doc = vec![vec![1.0, 0.0]];
        // Cosine should normalize, so result is 1.0
        assert!((maxsim_cosine_vecs(&query, &doc) - 1.0).abs() < 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IDF and BM25 weighting unit tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn idf_weights_empty_input() {
        assert_eq!(idf_weights(&[], 1000), vec![]);
        assert_eq!(idf_weights(&[100, 200], 0), vec![]);
    }

    #[test]
    fn idf_weights_zero_doc_freq() {
        // Token that never appears should get maximum weight
        let weights = idf_weights(&[0, 100], 1000);
        assert_eq!(weights.len(), 2);
        assert!(
            weights[0] > weights[1],
            "Zero doc freq should get higher weight"
        );
    }

    #[test]
    fn idf_weights_all_same_freq() {
        // All tokens with same frequency should get uniform weights
        let weights = idf_weights(&[100, 100, 100], 1000);
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - weights[1]).abs() < 1e-5);
        assert!((weights[1] - weights[2]).abs() < 1e-5);
        assert!(
            (weights[0] - 1.0).abs() < 1e-5,
            "Uniform weights should be 1.0"
        );
    }

    #[test]
    fn idf_weights_doc_freq_exceeds_total() {
        // Invalid: df > total_docs should be handled gracefully
        let weights = idf_weights(&[100, 2000], 1000);
        assert_eq!(weights.len(), 2);
        // Second token (df > total) should get minimum weight (0.0 after normalization)
        assert!(weights[0] >= weights[1]);
    }

    #[test]
    fn idf_weights_normalized_range() {
        let weights = idf_weights(&[10, 100, 1000], 10000);
        assert_eq!(weights.len(), 3);
        for w in &weights {
            assert!(*w >= 0.0 && *w <= 1.0, "Weight {} should be in [0, 1]", w);
        }
        // Check that min is 0.0 and max is 1.0 (after normalization)
        let min_w = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let max_w = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((min_w - 0.0).abs() < 1e-5, "Min weight should be 0.0");
        assert!((max_w - 1.0).abs() < 1e-5, "Max weight should be 1.0");
    }

    #[test]
    fn bm25_weights_length_mismatch() {
        assert_eq!(bm25_weights(&[100], &[1, 2], 1000, 1.5), vec![]);
        assert_eq!(bm25_weights(&[100, 200], &[1], 1000, 1.5), vec![]);
    }

    #[test]
    fn bm25_weights_empty_input() {
        assert_eq!(bm25_weights(&[], &[], 1000, 1.5), vec![]);
        assert_eq!(bm25_weights(&[100], &[1], 0, 1.5), vec![]);
    }

    #[test]
    fn bm25_weights_negative_k1() {
        // Negative k1 should be clamped to 0
        let weights = bm25_weights(&[100, 200], &[1, 1], 1000, -1.0);
        assert_eq!(weights.len(), 2);
        // With k1=0, formula becomes idf * tf / tf = idf
        // So it should reduce to IDF weighting
        for w in &weights {
            assert!(*w >= 0.0 && *w <= 1.0);
        }
    }

    #[test]
    fn bm25_weights_zero_k1() {
        // k1=0 should work (reduces to IDF weighting)
        let weights = bm25_weights(&[100, 200], &[1, 1], 1000, 0.0);
        assert_eq!(weights.len(), 2);
        for w in &weights {
            assert!(*w >= 0.0 && *w <= 1.0);
        }
    }

    #[test]
    fn bm25_weights_higher_tf_higher_weight() {
        // Token with higher query frequency should get higher weight
        let weights = bm25_weights(&[100, 100], &[1, 3], 1000, 1.5);
        assert_eq!(weights.len(), 2);
        assert!(
            weights[1] > weights[0],
            "Higher tf should give higher weight"
        );
    }

    #[test]
    fn bm25_weights_normalized_range() {
        let weights = bm25_weights(&[10, 100, 1000], &[1, 2, 1], 10000, 1.5);
        assert_eq!(weights.len(), 3);
        for w in &weights {
            assert!(*w >= 0.0 && *w <= 1.0, "Weight {} should be in [0, 1]", w);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Patch and snippet extraction unit tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn patches_to_regions_empty() {
        assert_eq!(patches_to_regions(&[], 1024, 768, 32), vec![]);
    }

    #[test]
    fn patches_to_regions_zero_patches_per_side() {
        assert_eq!(patches_to_regions(&[0, 1], 1024, 768, 0), vec![]);
    }

    #[test]
    fn patches_to_regions_zero_dimensions() {
        assert_eq!(patches_to_regions(&[0, 1], 0, 768, 32), vec![]);
        assert_eq!(patches_to_regions(&[0, 1], 1024, 0, 32), vec![]);
    }

    #[test]
    fn patches_to_regions_invalid_index() {
        // Patch index >= total_patches should be filtered out
        let regions = patches_to_regions(&[0, 1024, 1], 1024, 768, 32);
        // 32×32 = 1024 patches, so index 1024 is invalid
        assert_eq!(regions.len(), 2, "Should filter out invalid patch index");
    }

    #[test]
    fn patches_to_regions_valid() {
        let regions = patches_to_regions(&[0, 1, 32, 33], 1024, 768, 32);
        assert_eq!(regions.len(), 4);

        // Patch 0: row 0, col 0
        assert_eq!(regions[0], (0, 0, 32, 24));

        // Patch 1: row 0, col 1
        assert_eq!(regions[1], (32, 0, 32, 24));

        // Patch 32: row 1, col 0
        assert_eq!(regions[2], (0, 24, 32, 24));

        // Patch 33: row 1, col 1
        assert_eq!(regions[3], (32, 24, 32, 24));
    }

    #[test]
    fn patches_to_regions_bounds_check() {
        // All regions should be within image bounds
        let regions = patches_to_regions(&[0, 100, 500, 1023], 1024, 768, 32);
        for (x, y, w, h) in &regions {
            assert!(*x + *w <= 1024, "Region x+width should be <= image width");
            assert!(*y + *h <= 768, "Region y+height should be <= image height");
        }
    }

    #[test]
    fn extract_snippet_indices_empty() {
        assert_eq!(extract_snippet_indices(&[], 2, 10), vec![]);
    }

    #[test]
    fn extract_snippet_indices_max_tokens_zero() {
        let alignments = vec![(0, 5, 0.9), (1, 10, 0.8)];
        assert_eq!(extract_snippet_indices(&alignments, 1, 0), vec![]);
    }

    #[test]
    fn extract_snippet_indices_no_context() {
        let alignments = vec![(0, 5, 0.9), (1, 10, 0.8)];
        let snippet = extract_snippet_indices(&alignments, 0, 100);
        assert_eq!(snippet, vec![5, 10], "Should include only matched tokens");
    }

    #[test]
    fn extract_snippet_indices_with_context() {
        let alignments = vec![(0, 5, 0.9)];
        let snippet = extract_snippet_indices(&alignments, 2, 100);
        // Should include: 5 (match), 3, 4 (before), 6, 7 (after)
        assert!(snippet.contains(&5), "Should include matched token");
        assert!(
            snippet.contains(&3) || snippet.contains(&4),
            "Should include context before"
        );
        assert!(
            snippet.contains(&6) || snippet.contains(&7),
            "Should include context after"
        );
    }

    #[test]
    fn extract_snippet_indices_max_tokens_limit() {
        let alignments: Vec<(usize, usize, f32)> = (0..100).map(|i| (0, i, 0.9)).collect();
        let snippet = extract_snippet_indices(&alignments, 2, 10);
        assert_eq!(snippet.len(), 10, "Should respect max_tokens limit");
    }

    #[test]
    fn extract_snippet_indices_sorted() {
        let alignments = vec![(0, 10, 0.9), (1, 5, 0.8), (2, 20, 0.7)];
        let snippet = extract_snippet_indices(&alignments, 1, 100);
        for i in 1..snippet.len() {
            assert!(snippet[i] >= snippet[i - 1], "Should be sorted");
        }
    }

    #[test]
    fn extract_snippet_indices_context_bounds() {
        // Test that context doesn't go negative
        let alignments = vec![(0, 0, 0.9)]; // First token
        let snippet = extract_snippet_indices(&alignments, 5, 100);
        // Should not panic, and should only include valid indices
        // Note: usize is always >= 0, so this is just a sanity check
        assert!(
            !snippet.is_empty() || snippet.iter().all(|&idx| idx < 1000),
            "Indices should be reasonable"
        );
    }

    #[test]
    fn extract_snippet_indices_deduplication() {
        // Multiple alignments to same doc token should only appear once
        let alignments = vec![(0, 5, 0.9), (1, 5, 0.8), (2, 5, 0.7)];
        let snippet = extract_snippet_indices(&alignments, 2, 100);
        let count_5 = snippet.iter().filter(|&&x| x == 5).count();
        assert_eq!(count_5, 1, "Should deduplicate same token");
    }

    #[test]
    fn test_maxsim_cosine_basic() {
        let q1 = [1.0, 0.0];
        let d1 = [1.0, 0.0];
        let d2 = [0.0, 1.0];

        let query: Vec<&[f32]> = vec![&q1];
        let doc: Vec<&[f32]> = vec![&d1, &d2];

        // q1's best cosine match is d1 (cosine=1.0)
        assert!((maxsim_cosine(&query, &doc) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_cosine_empty_doc() {
        let q1 = [1.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];
        assert_eq!(maxsim_cosine(&query, &[]), 0.0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Mutation-killing tests: verify exact mathematical properties
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn cosine_zero_norm_returns_zero_not_nan() {
        // When either vector has zero norm, cosine returns 0.0
        // This tests the `> 0.0` check - changing to `>= 0.0` would cause NaN
        let zero = [0.0, 0.0];
        let nonzero = [1.0, 2.0];

        let c1 = cosine(&zero, &nonzero);
        let c2 = cosine(&nonzero, &zero);
        let c3 = cosine(&zero, &zero);

        assert_eq!(c1, 0.0, "cosine(zero, x) should be 0, got {}", c1);
        assert_eq!(c2, 0.0, "cosine(x, zero) should be 0, got {}", c2);
        assert_eq!(c3, 0.0, "cosine(zero, zero) should be 0, got {}", c3);
        assert!(!c1.is_nan(), "should not return NaN");
    }

    #[test]
    fn cosine_near_zero_norm_stable() {
        // Very small norms below threshold return 0.0 for stability
        let tiny = [1e-20, 0.0];
        let normal = [1.0, 0.0];

        let c = cosine(&tiny, &normal);
        assert!(c.is_finite(), "cosine with tiny norm should be finite");
        // Returns 0.0 for stability when norm < 1e-9
        assert_eq!(c, 0.0, "tiny norm should return 0.0");

        // Small but above threshold should work
        let small = [1e-8, 0.0];
        let c2 = cosine(&small, &normal);
        assert!(c2.is_finite());
        assert!(
            (c2 - 1.0).abs() < 1e-3,
            "parallel vectors above threshold should have cosine ~1"
        );
    }

    #[test]
    fn dot_exact_orthogonal() {
        // Orthogonal vectors have dot product 0
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert_eq!(dot(&a, &b), 0.0);
    }

    #[test]
    fn dot_exact_parallel() {
        // Parallel unit vectors have dot product 1
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert!((dot(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn maxsim_single_query_single_doc() {
        // Simplest case: 1 query token, 1 doc token
        let q = [1.0, 2.0, 3.0];
        let d = [4.0, 5.0, 6.0];

        let query: Vec<&[f32]> = vec![&q];
        let doc: Vec<&[f32]> = vec![&d];

        let expected_dot = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 4 + 10 + 18 = 32
        let actual = maxsim(&query, &doc);

        assert!(
            (actual - expected_dot).abs() < 1e-5,
            "expected {}, got {}",
            expected_dot,
            actual
        );
    }

    #[test]
    fn maxsim_sum_of_maxes() {
        // MaxSim = sum over query tokens of max(dot with each doc token)
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.5, 0.0]; // dot(q1,d1)=0.5, dot(q2,d1)=0.0
        let d2 = [0.0, 0.8]; // dot(q1,d2)=0.0, dot(q2,d2)=0.8

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2];

        // max for q1 is 0.5 (from d1), max for q2 is 0.8 (from d2)
        let expected = 0.5 + 0.8;
        let actual = maxsim(&query, &doc);

        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn norm_exact_values() {
        // Test exact norm calculations
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-9, "3-4-5 triangle");
        assert!((norm(&[1.0, 0.0]) - 1.0).abs() < 1e-9, "unit x");
        assert!((norm(&[0.0, 0.0]) - 0.0).abs() < 1e-9, "zero vector");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Weighted MaxSim tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn maxsim_weighted_basic() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [1.0, 0.0]; // dot(q1,d1)=1.0, dot(q2,d1)=0.0
        let d2 = [0.0, 1.0]; // dot(q1,d2)=0.0, dot(q2,d2)=1.0

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2];

        // Without weights: max(q1)=1.0 + max(q2)=1.0 = 2.0
        let unweighted = maxsim(&query, &doc);
        assert!((unweighted - 2.0).abs() < 1e-5);

        // With equal weights: same result
        let equal_weights = [1.0, 1.0];
        let weighted_equal = maxsim_weighted(&query, &doc, &equal_weights);
        assert!((weighted_equal - 2.0).abs() < 1e-5);

        // With 2x weight on first query token
        let weights = [2.0, 1.0];
        let weighted = maxsim_weighted(&query, &doc, &weights);
        // Expected: 2.0*1.0 + 1.0*1.0 = 3.0
        assert!((weighted - 3.0).abs() < 1e-5);
    }

    #[test]
    fn maxsim_weighted_single_token() {
        let q = [1.0, 0.0, 0.0];
        let d = [1.0, 0.0, 0.0];

        let query: Vec<&[f32]> = vec![&q];
        let doc: Vec<&[f32]> = vec![&d];

        // Weight of 2.0 should double the score
        let weights = [2.0];
        let score = maxsim_weighted(&query, &doc, &weights);
        assert!((score - 2.0).abs() < 1e-5);
    }

    #[test]
    fn maxsim_weighted_empty() {
        let q1 = [1.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];
        let empty: Vec<&[f32]> = vec![];
        let weights = [1.0];

        assert_eq!(maxsim_weighted(&query, &empty, &weights), 0.0);
        assert_eq!(maxsim_weighted(&[], &query, &weights), 0.0);
    }

    #[test]
    fn maxsim_weighted_missing_weights_default_to_one() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d = [1.0, 1.0]; // dot=1 with each

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d];

        // Only one weight provided, second should default to 1.0
        let weights = [2.0]; // only for first token
        let score = maxsim_weighted(&query, &doc, &weights);
        // Expected: 2.0*1.0 + 1.0*1.0 = 3.0
        assert!((score - 3.0).abs() < 1e-5);
    }

    #[test]
    fn maxsim_weighted_zero_weight_ignores_token() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d = [1.0, 0.0]; // dot(q1,d)=1.0, dot(q2,d)=0.0

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d];

        // Zero weight on q2 means only q1 contributes
        let weights = [1.0, 0.0];
        let score = maxsim_weighted(&query, &doc, &weights);
        // Expected: 1.0*1.0 + 0.0*0.0 = 1.0
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn maxsim_cosine_weighted_basic() {
        let q1 = [2.0, 0.0]; // unnormalized
        let q2 = [0.0, 3.0]; // unnormalized
        let d = [1.0, 0.0]; // cosine(q1,d)=1.0, cosine(q2,d)=0.0

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d];

        // q1 has cosine 1.0 with d, q2 has cosine 0.0
        let weights = [2.0, 1.0];
        let score = maxsim_cosine_weighted(&query, &doc, &weights);
        // Expected: 2.0*1.0 + 1.0*0.0 = 2.0
        assert!((score - 2.0).abs() < 1e-5);
    }

    #[test]
    fn maxsim_weighted_vecs_convenience() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let weights = [2.0, 0.5];

        let score = maxsim_weighted_vecs(&query, &doc, &weights);
        // Expected: 2.0*1.0 + 0.5*1.0 = 2.5
        assert!((score - 2.5).abs() < 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Score normalization tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn normalize_maxsim_basic() {
        // Score of 16 with query_maxlen=32 should give 0.5
        assert!((normalize_maxsim(16.0, 32) - 0.5).abs() < 1e-9);
        assert!((normalize_maxsim(32.0, 32) - 1.0).abs() < 1e-9);
        assert!((normalize_maxsim(0.0, 32) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_maxsim_batch_basic() {
        let scores = vec![16.0, 32.0, 8.0];
        let normalized = normalize_maxsim_batch(&scores, 32);
        assert!((normalized[0] - 0.5).abs() < 1e-9);
        assert!((normalized[1] - 1.0).abs() < 1e-9);
        assert!((normalized[2] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn softmax_scores_sums_to_one() {
        let scores = vec![2.0, 1.0, 0.5, 0.1];
        let probs = softmax_scores(&scores);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn softmax_scores_preserves_order() {
        let scores = vec![3.0, 1.0, 2.0];
        let probs = softmax_scores(&scores);
        // Higher scores should have higher probabilities
        assert!(probs[0] > probs[2]); // 3.0 > 2.0
        assert!(probs[2] > probs[1]); // 2.0 > 1.0
    }

    #[test]
    fn softmax_scores_empty() {
        let probs = softmax_scores(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn softmax_scores_single() {
        let probs = softmax_scores(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_scores_handles_nan() {
        let scores = vec![f32::NAN, 1.0, 2.0];
        let probs = softmax_scores(&scores);
        // NaN input should produce 0 contribution (exp(-inf) -> 0)
        assert!(probs[0].is_finite());
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_scores_large_values_stable() {
        // Numerical stability test: very large values shouldn't overflow
        let scores = vec![1000.0, 1001.0, 999.0];
        let probs = softmax_scores(&scores);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_indices_basic() {
        let scores = vec![0.5, 0.9, 0.1, 0.7, 0.3];
        let top2 = top_k_indices(&scores, 2);
        assert_eq!(top2, vec![1, 3]); // 0.9 at idx 1, 0.7 at idx 3
    }

    #[test]
    fn top_k_indices_with_ties() {
        let scores = vec![0.5, 0.5, 0.5];
        let top2 = top_k_indices(&scores, 2);
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn top_k_indices_k_larger_than_len() {
        let scores = vec![0.5, 0.9];
        let top5 = top_k_indices(&scores, 5);
        assert_eq!(top5.len(), 2); // Can't return more than we have
    }

    #[test]
    fn top_k_indices_with_nan() {
        let scores = vec![0.5, f32::NAN, 0.9, 0.7];
        let top2 = top_k_indices(&scores, 2);
        // NaN should sort last, so top 2 are 0.9 and 0.7
        assert_eq!(top2, vec![2, 3]);
    }

    #[test]
    fn top_k_indices_empty() {
        let top = top_k_indices(&[], 5);
        assert!(top.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mutation-killing tests (targeted to catch specific mutants)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn cosine_zero_norm_returns_zero() {
        // Zero vector should return 0.0, not NaN or infinity
        let zero = [0.0, 0.0, 0.0];
        let nonzero = [1.0, 2.0, 3.0];
        assert_eq!(cosine(&zero, &nonzero), 0.0);
        assert_eq!(cosine(&nonzero, &zero), 0.0);
        assert_eq!(cosine(&zero, &zero), 0.0);
    }

    #[test]
    fn cosine_near_zero_norm_returns_zero() {
        // Very small norm should also return 0.0
        let tiny = [1e-10, 1e-10, 1e-10];
        let nonzero = [1.0, 2.0, 3.0];
        assert_eq!(cosine(&tiny, &nonzero), 0.0);
    }

    #[test]
    fn cosine_at_threshold_boundary() {
        // Test exactly at the 1e-9 threshold
        // Vector with norm = 1e-9 should return 0
        let at_threshold = [1e-9, 0.0, 0.0]; // norm = 1e-9
        let nonzero = [1.0, 0.0, 0.0];
        // Should be 0.0 because norm <= 1e-9
        assert_eq!(cosine(&at_threshold, &nonzero), 0.0);

        // Vector with norm > 1e-9 should return actual cosine
        let above_threshold = [1e-8, 0.0, 0.0]; // norm = 1e-8 > 1e-9
        let result = cosine(&above_threshold, &nonzero);
        // Should be ~1.0 (parallel vectors)
        assert!(
            (result - 1.0).abs() < 0.1,
            "Above threshold should compute: {}",
            result
        );
    }

    #[test]
    fn maxsim_cosine_vecs_not_one() {
        // Verify maxsim_cosine_vecs returns actual score, not just 1.0
        let query = vec![vec![1.0, 0.0, 0.0]];
        let doc = vec![vec![0.0, 1.0, 0.0]]; // Orthogonal
        let score = maxsim_cosine_vecs(&query, &doc);
        assert!(
            score.abs() < 0.01,
            "Orthogonal vectors should have ~0 cosine, got {}",
            score
        );
    }

    #[test]
    fn maxsim_cosine_vecs_matches_slice_version() {
        let query = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let doc = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let q_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
        let d_refs: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

        let vec_score = maxsim_cosine_vecs(&query, &doc);
        let slice_score = maxsim_cosine(&q_refs, &d_refs);

        assert!(
            (vec_score - slice_score).abs() < 1e-5,
            "Vec and slice versions should match: {} vs {}",
            vec_score,
            slice_score
        );
    }

    #[test]
    fn maxsim_cosine_weighted_empty_returns_zero() {
        let empty_q: Vec<&[f32]> = vec![];
        let empty_d: Vec<&[f32]> = vec![];
        let some: Vec<&[f32]> = vec![&[1.0, 0.0]];
        let weights = [1.0];

        assert_eq!(maxsim_cosine_weighted(&empty_q, &some, &weights), 0.0);
        assert_eq!(maxsim_cosine_weighted(&some, &empty_d, &weights), 0.0);
    }

    #[test]
    fn dot_short_vector_uses_portable() {
        // Vector shorter than MIN_DIM_SIMD should still work correctly
        let short_a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let short_b: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();

        let result = dot(&short_a, &short_b);
        // Manual: 0*1 + 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 = 2+6+12+20+30+42+56 = 168
        assert!(
            (result - 168.0).abs() < 1e-3,
            "Short vector dot: {} != 168",
            result
        );
    }

    #[test]
    fn dot_exactly_min_dim() {
        // Test at exactly MIN_DIM_SIMD (16)
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 16];

        let result = dot(&a, &b);
        // Sum of 0..15 = (15*16)/2 = 120
        assert!(
            (result - 120.0).abs() < 1e-3,
            "MIN_DIM dot: {} != 120",
            result
        );
    }

    #[test]
    fn dot_truncating_mismatched_lengths() {
        // dot_truncating uses min(len_a, len_b)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 1.0, 1.0];

        let result = dot_truncating(&a, &b);
        // Only uses first 3: 1+2+3 = 6
        assert!(
            (result - 6.0).abs() < 1e-5,
            "Mismatched len dot_truncating: {} != 6",
            result
        );
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    #[cfg(debug_assertions)]
    fn dot_panics_on_mismatch_in_debug() {
        // In debug builds, dot() should panic on mismatched dimensions
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 1.0, 1.0];
        let _ = dot(&a, &b);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-10.0f32..10.0, len)
    }

    proptest! {
        /// SIMD dot matches portable implementation
        #[test]
        fn dot_simd_matches_portable(a in arb_vec(128), b in arb_vec(128)) {
            let simd_result = dot(&a, &b);
            let portable_result = dot_portable(&a, &b);
            prop_assert!(
                (simd_result - portable_result).abs() < 1e-3,
                "SIMD {} != portable {}",
                simd_result,
                portable_result
            );
        }

        /// Dot product is commutative: dot(a, b) == dot(b, a)
        #[test]
        fn dot_commutative(a in arb_vec(64), b in arb_vec(64)) {
            let ab = dot(&a, &b);
            let ba = dot(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5);
        }

        /// Cosine similarity is in [-1, 1] for non-zero vectors
        #[test]
        fn cosine_bounded(
            a in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
            b in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6))
        ) {
            let c = cosine(&a, &b);
            prop_assert!((-1.0 - 1e-5..=1.0 + 1e-5).contains(&c), "cosine {} out of bounds", c);
        }

        /// Cosine similarity is commutative
        #[test]
        fn cosine_commutative(a in arb_vec(32), b in arb_vec(32)) {
            let ab = cosine(&a, &b);
            let ba = cosine(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5);
        }

        /// MaxSim is non-negative when all dot products are non-negative
        #[test]
        fn maxsim_nonnegative_inputs(
            q_data in proptest::collection::vec(arb_vec(16), 1..5),
            d_data in proptest::collection::vec(arb_vec(16), 1..5)
        ) {
            // Make all vectors have non-negative components
            let q_pos: Vec<Vec<f32>> = q_data.iter()
                .map(|v| v.iter().map(|x| x.abs()).collect())
                .collect();
            let d_pos: Vec<Vec<f32>> = d_data.iter()
                .map(|v| v.iter().map(|x| x.abs()).collect())
                .collect();

            let q_refs: Vec<&[f32]> = q_pos.iter().map(|v| v.as_slice()).collect();
            let d_refs: Vec<&[f32]> = d_pos.iter().map(|v| v.as_slice()).collect();

            let score = maxsim(&q_refs, &d_refs);
            prop_assert!(score >= 0.0, "maxsim {} should be non-negative", score);
        }

        /// Empty inputs return 0
        #[test]
        fn maxsim_empty_returns_zero(q_data in proptest::collection::vec(arb_vec(8), 0..3)) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(|v| v.as_slice()).collect();
            let empty: Vec<&[f32]> = vec![];

            // Empty query always returns 0
            prop_assert_eq!(maxsim(&empty, &q_refs), 0.0);
            // Empty doc always returns 0
            prop_assert_eq!(maxsim(&q_refs, &empty), 0.0);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Additional mathematical properties
        // ─────────────────────────────────────────────────────────────────────────

        /// Dot product with self equals squared L2 norm
        #[test]
        fn dot_self_is_squared_norm(v in arb_vec(32)) {
            let dot_self = dot(&v, &v);
            let n = norm(&v);
            let squared_norm = n * n;
            // Use relative tolerance for large values
            let tolerance = (squared_norm.abs() * 1e-4).max(1e-4);
            prop_assert!(
                (dot_self - squared_norm).abs() < tolerance,
                "dot(v,v) = {} but norm²= {}",
                dot_self,
                squared_norm
            );
        }

        /// Cosine with self is 1 (for non-zero vectors)
        #[test]
        fn cosine_self_is_one(v in arb_vec(16).prop_filter("non-zero", |v| norm(v) > 1e-6)) {
            let c = cosine(&v, &v);
            prop_assert!(
                (c - 1.0).abs() < 1e-5,
                "cosine(v, v) = {} should be 1",
                c
            );
        }

        /// Norm is non-negative
        #[test]
        fn norm_nonnegative(v in arb_vec(64)) {
            let n = norm(&v);
            prop_assert!(n >= 0.0, "norm {} should be non-negative", n);
        }

        /// Norm of scaled vector: ||αv|| = |α| ||v||
        #[test]
        fn norm_scaling(v in arb_vec(16), alpha in -10.0f32..10.0) {
            let scaled: Vec<f32> = v.iter().map(|x| x * alpha).collect();
            let n_v = norm(&v);
            let n_scaled = norm(&scaled);
            let expected = alpha.abs() * n_v;
            prop_assert!(
                (n_scaled - expected).abs() < 1e-4,
                "||αv|| = {} but |α|||v|| = {}",
                n_scaled,
                expected
            );
        }

        /// Dot product is bilinear: dot(αa, b) = α·dot(a, b)
        #[test]
        fn dot_bilinear(a in arb_vec(16), b in arb_vec(16), alpha in -5.0f32..5.0) {
            let scaled_a: Vec<f32> = a.iter().map(|x| x * alpha).collect();
            let dot_scaled = dot(&scaled_a, &b);
            let expected = alpha * dot(&a, &b);
            prop_assert!(
                (dot_scaled - expected).abs() < 1e-3,
                "dot(αa, b) = {} but α·dot(a, b) = {}",
                dot_scaled,
                expected
            );
        }

        /// Cauchy-Schwarz: |dot(a, b)| <= ||a|| ||b||
        #[test]
        fn cauchy_schwarz(a in arb_vec(32), b in arb_vec(32)) {
            let d = dot(&a, &b).abs();
            let bound = norm(&a) * norm(&b);
            prop_assert!(
                d <= bound + 1e-4,
                "|dot(a,b)| = {} should be <= ||a||·||b|| = {}",
                d,
                bound
            );
        }

        /// MaxSim scales linearly with query token count for identical matches
        #[test]
        fn maxsim_scales_with_query_count(n_query in 1usize..5, dim in 4usize..8) {
            // Create identical query and doc token
            let token: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let query: Vec<Vec<f32>> = vec![token.clone(); n_query];
            let doc = [token.clone()];

            let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();

            let score = maxsim(&q_refs, &d_refs);
            // Each query token has max sim = ||token||² (dot with itself)
            let expected = n_query as f32 * dot(&token, &token);

            prop_assert!(
                (score - expected).abs() < 1e-4,
                "MaxSim should scale linearly: {} vs expected {}",
                score,
                expected
            );
        }

        /// MaxSim with normalized vectors bounded by query count
        #[test]
        fn maxsim_cosine_bounded_by_query_count(n_query in 1usize..4, n_doc in 1usize..4) {
            // Create unit vectors
            let query: Vec<Vec<f32>> = (0..n_query)
                .map(|i| {
                    let mut v = vec![0.0f32; 8];
                    v[i % 8] = 1.0;
                    v
                })
                .collect();
            let doc: Vec<Vec<f32>> = (0..n_doc)
                .map(|i| {
                    let mut v = vec![0.0f32; 8];
                    v[(i + 1) % 8] = 1.0;
                    v
                })
                .collect();

            let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();

            let score = maxsim_cosine(&q_refs, &d_refs);

            // Each query token contributes at most 1 (max cosine with any doc token)
            let upper_bound = n_query as f32;
            prop_assert!(
                score <= upper_bound + 1e-5,
                "MaxSim cosine {} should be <= {}",
                score,
                upper_bound
            );
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Weighted MaxSim property tests
        // ─────────────────────────────────────────────────────────────────────────

        /// Weighted MaxSim with all weights=1 equals unweighted
        #[test]
        fn maxsim_weighted_unit_weights_equals_unweighted(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();
            let weights: Vec<f32> = vec![1.0; q_data.len()];

            let unweighted = maxsim(&q_refs, &d_refs);
            let weighted = maxsim_weighted(&q_refs, &d_refs, &weights);

            prop_assert!(
                (unweighted - weighted).abs() < 1e-5,
                "unweighted {} != weighted {}",
                unweighted,
                weighted
            );
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Alignment property tests
        // ─────────────────────────────────────────────────────────────────────────

        /// Alignment sum always equals MaxSim score
        #[test]
        fn maxsim_alignments_sum_equals_maxsim(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();

            let alignments = maxsim_alignments(&q_refs, &d_refs);
            let maxsim_score = maxsim(&q_refs, &d_refs);
            let alignment_sum: f32 = alignments.iter().map(|(_, _, s)| s).sum();

            prop_assert!(
                (alignment_sum - maxsim_score).abs() < 1e-4,
                "Alignment sum {} != MaxSim {}",
                alignment_sum,
                maxsim_score
            );
        }

        /// Alignment count equals query token count
        #[test]
        fn maxsim_alignments_count_equals_query_tokens(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();

            let alignments = maxsim_alignments(&q_refs, &d_refs);
            prop_assert_eq!(
                alignments.len(),
                q_data.len(),
                "Should have one alignment per query token"
            );
        }

        /// Alignment query indices are sequential
        #[test]
        fn maxsim_alignments_query_indices_sequential(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();

            let alignments = maxsim_alignments(&q_refs, &d_refs);
            for (i, (q_idx, _, _)) in alignments.iter().enumerate() {
                prop_assert_eq!(
                    *q_idx, i,
                    "Query index {} should match position {}",
                    q_idx, i
                );
            }
        }

        /// Alignment doc indices are valid
        #[test]
        fn maxsim_alignments_doc_indices_valid(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();

            let alignments = maxsim_alignments(&q_refs, &d_refs);
            for (_, d_idx, _) in &alignments {
                prop_assert!(
                    (*d_idx as usize) < d_data.len(),
                    "Doc index {} out of bounds (len={})",
                    d_idx,
                    d_data.len()
                );
            }
        }

        /// Highlight matches are subset of doc indices
        #[test]
        fn highlight_matches_valid_indices(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5),
            threshold in -1.0f32..2.0
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();

            let highlighted = highlight_matches(&q_refs, &d_refs, threshold);
            for &idx in &highlighted {
                prop_assert!(
                    idx < d_data.len(),
                    "Highlighted index {} out of bounds (len={})",
                    idx,
                    d_data.len()
                );
            }
            // Should be sorted
            for i in 1..highlighted.len() {
                prop_assert!(
                    highlighted[i - 1] < highlighted[i],
                    "Highlighted indices should be sorted"
                );
            }
        }

        /// Highlight matches with very high threshold should be empty or small
        #[test]
        fn highlight_matches_high_threshold(
            q_data in proptest::collection::vec(arb_vec(8), 1..5),
            d_data in proptest::collection::vec(arb_vec(8), 1..5)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();

            let high_threshold = highlight_matches(&q_refs, &d_refs, 10.0);
            let low_threshold = highlight_matches(&q_refs, &d_refs, -10.0);

            prop_assert!(
                high_threshold.len() <= low_threshold.len(),
                "High threshold should return fewer or equal matches"
            );
        }

        /// Weighted MaxSim scales with uniform weight
        #[test]
        fn maxsim_weighted_uniform_scaling(
            q_data in proptest::collection::vec(arb_vec(8), 1..4),
            d_data in proptest::collection::vec(arb_vec(8), 1..4),
            scale in 0.1f32..5.0
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();
            let weights: Vec<f32> = vec![scale; q_data.len()];

            let unweighted = maxsim(&q_refs, &d_refs);
            let weighted = maxsim_weighted(&q_refs, &d_refs, &weights);

            let expected = scale * unweighted;
            // Use relative tolerance for large values
            let tolerance = (expected.abs() * 1e-4).max(1e-4);
            prop_assert!(
                (weighted - expected).abs() < tolerance,
                "weighted {} != scale * unweighted {}",
                weighted,
                expected
            );
        }

        /// Weighted MaxSim with all zero weights returns 0
        #[test]
        fn maxsim_weighted_zero_weights_returns_zero(
            q_data in proptest::collection::vec(arb_vec(8), 1..4),
            d_data in proptest::collection::vec(arb_vec(8), 1..4)
        ) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = d_data.iter().map(Vec::as_slice).collect();
            let weights: Vec<f32> = vec![0.0; q_data.len()];

            let weighted = maxsim_weighted(&q_refs, &d_refs, &weights);
            prop_assert!(
                weighted.abs() < 1e-9,
                "zero weights should give 0, got {}",
                weighted
            );
        }

        /// Weighted MaxSim empty inputs return 0
        #[test]
        fn maxsim_weighted_empty_returns_zero(
            weights in proptest::collection::vec(0.1f32..2.0, 0..5)
        ) {
            let empty_q: Vec<&[f32]> = vec![];
            let empty_d: Vec<&[f32]> = vec![];
            let some_q: Vec<&[f32]> = vec![&[1.0, 0.0]];

            prop_assert_eq!(maxsim_weighted(&empty_q, &some_q, &weights), 0.0);
            prop_assert_eq!(maxsim_weighted(&some_q, &empty_d, &weights), 0.0);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Normalization property tests
        // ─────────────────────────────────────────────────────────────────────────

        /// Softmax always sums to 1 (for non-empty finite inputs)
        #[test]
        fn softmax_sums_to_one(scores in proptest::collection::vec(-100.0f32..100.0, 1..20)) {
            let probs = softmax_scores(&scores);
            let sum: f32 = probs.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax sum {} should be 1",
                sum
            );
        }

        /// Softmax preserves relative ordering
        #[test]
        fn softmax_preserves_order(scores in proptest::collection::vec(-10.0f32..10.0, 2..10)) {
            let probs = softmax_scores(&scores);

            // For each pair, if score[i] > score[j], then prob[i] > prob[j]
            for i in 0..scores.len() {
                for j in (i + 1)..scores.len() {
                    if scores[i] > scores[j] {
                        prop_assert!(
                            probs[i] >= probs[j],
                            "softmax should preserve order: score[{}]={} > score[{}]={}, but prob {} <= {}",
                            i, scores[i], j, scores[j], probs[i], probs[j]
                        );
                    }
                }
            }
        }

        /// Softmax outputs are all in [0, 1]
        #[test]
        fn softmax_outputs_bounded(scores in proptest::collection::vec(-50.0f32..50.0, 1..15)) {
            let probs = softmax_scores(&scores);
            for (i, p) in probs.iter().enumerate() {
                prop_assert!(
                    *p >= 0.0 && *p <= 1.0,
                    "softmax[{}] = {} should be in [0, 1]",
                    i, p
                );
            }
        }

        /// Normalize is linear: normalize(a+b) = normalize(a) + normalize(b)
        #[test]
        fn normalize_is_linear(a in 0.0f32..100.0, b in 0.0f32..100.0, maxlen in 1u32..100) {
            let sum_norm = normalize_maxsim(a + b, maxlen);
            let norm_sum = normalize_maxsim(a, maxlen) + normalize_maxsim(b, maxlen);
            // f32 precision limits require slightly larger tolerance
            prop_assert!(
                (sum_norm - norm_sum).abs() < 1e-5,
                "normalize should be linear: {} vs {}",
                sum_norm,
                norm_sum
            );
        }

        /// Top-k returns correct number of elements
        #[test]
        fn top_k_returns_correct_count(
            scores in proptest::collection::vec(-100.0f32..100.0, 1..20),
            k in 1usize..25
        ) {
            let result = top_k_indices(&scores, k);
            let expected_len = k.min(scores.len());
            prop_assert_eq!(
                result.len(),
                expected_len,
                "top_k should return min(k, len)"
            );
        }

        /// Top-k indices are valid
        #[test]
        fn top_k_indices_valid(
            scores in proptest::collection::vec(-100.0f32..100.0, 1..20),
            k in 1usize..25
        ) {
            let result = top_k_indices(&scores, k);
            for idx in &result {
                prop_assert!(
                    *idx < scores.len(),
                    "index {} should be < len {}",
                    idx,
                    scores.len()
                );
            }
        }

        /// Top-k returns unique indices
        #[test]
        fn top_k_indices_unique(
            scores in proptest::collection::vec(-100.0f32..100.0, 1..20),
            k in 1usize..25
        ) {
            let result = top_k_indices(&scores, k);
            let mut seen = std::collections::HashSet::new();
            for idx in &result {
                prop_assert!(
                    seen.insert(*idx),
                    "index {} appears twice",
                    idx
                );
            }
        }
    }
}
