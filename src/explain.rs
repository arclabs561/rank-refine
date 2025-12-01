//! Explainability for MaxSim scoring.
//!
//! Provides detailed provenance showing which query tokens matched which document tokens,
//! enabling debugging, highlighting, and user-facing explanations.

use crate::simd;

/// Detailed explanation of a MaxSim score computation.
///
/// Shows which query tokens contributed most, which document tokens they matched,
/// and optionally the text content of those tokens for human-readable explanations.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxSimExplanation {
    /// Total MaxSim score.
    pub total_score: f32,
    /// Contribution from each query token.
    pub token_contributions: Vec<TokenMatch>,
    /// Optional query token texts (for human-readable explanations).
    pub query_token_texts: Option<Vec<String>>,
    /// Optional document token texts (for human-readable explanations).
    pub doc_token_texts: Option<Vec<String>>,
}

/// Token-level match information.
///
/// Shows which document token a query token matched and how much it contributed.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenMatch {
    /// Query token index.
    pub query_token_idx: usize,
    /// Best matching document token index.
    pub best_doc_token_idx: usize,
    /// Similarity score between query and matched document token.
    pub similarity: f32,
    /// Optional query token text (if provided).
    pub query_token_text: Option<String>,
    /// Optional matched document token text (if provided).
    pub doc_token_text: Option<String>,
    /// Contribution to total MaxSim score (same as similarity for dot product).
    pub contribution: f32,
}

/// Compute MaxSim with full explainability.
///
/// Returns detailed information about which tokens matched, enabling debugging
/// and user-facing explanations.
///
/// # Arguments
///
/// * `query_tokens` - Query token embeddings
/// * `doc_tokens` - Document token embeddings
/// * `query_texts` - Optional query token texts (for human-readable output)
/// * `doc_texts` - Optional document token texts (for human-readable output)
/// * `use_cosine` - Whether to use cosine similarity (true) or dot product (false)
///
/// # Example
///
/// ```rust
/// use rank_refine::explain::maxsim_explained;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
///
/// let explanation = maxsim_explained(
///     &query,
///     &doc,
///     Some(&["capital", "France"]),
///     Some(&["Paris", "capital", "France"]),
///     false, // use dot product
/// );
///
/// println!("Total score: {}", explanation.total_score);
/// for match_info in &explanation.token_contributions {
///     println!(
///         "Query '{}' matched doc '{}' with score {:.3}",
///         match_info.query_token_text.as_deref().unwrap_or("?"),
///         match_info.doc_token_text.as_deref().unwrap_or("?"),
///         match_info.similarity
///     );
/// }
/// ```
#[must_use]
pub fn maxsim_explained(
    query_tokens: &[Vec<f32>],
    doc_tokens: &[Vec<f32>],
    query_texts: Option<&[&str]>,
    doc_texts: Option<&[&str]>,
    use_cosine: bool,
) -> MaxSimExplanation {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return MaxSimExplanation {
            total_score: 0.0,
            token_contributions: Vec::new(),
            query_token_texts: query_texts.map(|t| t.iter().map(|s| s.to_string()).collect()),
            doc_token_texts: doc_texts.map(|t| t.iter().map(|s| s.to_string()).collect()),
        };
    }

    let query_refs: Vec<&[f32]> = query_tokens.iter().map(|v| v.as_slice()).collect();
    let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(|v| v.as_slice()).collect();

    let alignments = if use_cosine {
        simd::maxsim_alignments_cosine(&query_refs, &doc_refs)
    } else {
        simd::maxsim_alignments(&query_refs, &doc_refs)
    };

    let token_contributions: Vec<TokenMatch> = alignments
        .into_iter()
        .map(|(q_idx, d_idx, similarity)| TokenMatch {
            query_token_idx: q_idx,
            best_doc_token_idx: d_idx,
            similarity,
            query_token_text: query_texts.and_then(|texts| texts.get(q_idx).map(|s| s.to_string())),
            doc_token_text: doc_texts.and_then(|texts| texts.get(d_idx).map(|s| s.to_string())),
            contribution: similarity,
        })
        .collect();

    let total_score: f32 = token_contributions.iter().map(|m| m.contribution).sum();

    MaxSimExplanation {
        total_score,
        token_contributions,
        query_token_texts: query_texts.map(|t| t.iter().map(|s| s.to_string()).collect()),
        doc_token_texts: doc_texts.map(|t| t.iter().map(|s| s.to_string()).collect()),
    }
}

/// Batch reranking input structure.
///
/// Encapsulates query and candidate information for efficient batch processing.
#[derive(Debug, Clone)]
pub struct RerankerInput<'a, K> {
    /// Dense query embedding (single vector).
    pub query_dense: Option<&'a [f32]>,
    /// Query token embeddings (for late interaction).
    pub query_tokens: Option<&'a [Vec<f32>]>,
    /// Candidates to rerank.
    pub candidates: Vec<Candidate<'a, K>>,
}

/// A candidate document for reranking.
#[derive(Debug, Clone)]
pub struct Candidate<'a, K> {
    /// Document identifier.
    pub id: K,
    /// Original score (from fusion or initial retrieval).
    pub original_score: f32,
    /// Dense embedding (single vector).
    pub dense_embedding: Option<&'a [f32]>,
    /// Token embeddings (for late interaction).
    pub token_embeddings: Option<&'a [Vec<f32>]>,
    /// Optional document text (for cross-encoder).
    pub text: Option<&'a str>,
}

/// Reranking method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RerankMethod {
    /// Dense cosine similarity.
    DenseCosine,
    /// MaxSim (late interaction).
    MaxSim,
    /// MaxSim with cosine similarity.
    MaxSimCosine,
    /// Weighted MaxSim (weights provided separately, not stored in enum).
    MaxSimWeighted,
}

/// Ranked result from reranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedResult<K> {
    /// Document identifier.
    pub id: K,
    /// Final reranked score.
    pub score: f32,
    /// Original score (before reranking).
    pub original_score: f32,
    /// Rank position (0-indexed).
    pub rank: usize,
}

/// Rerank candidates in batch.
///
/// Efficiently scores multiple candidates using the specified method.
///
/// # Example
///
/// ```rust
/// use rank_refine::explain::{RerankerInput, Candidate, RerankMethod, rerank_batch};
///
/// let query_tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
///
/// let candidates = vec![
///     Candidate {
///         id: "doc1",
///         original_score: 0.8,
///         dense_embedding: None,
///         token_embeddings: Some(&[vec![0.9, 0.1], vec![0.1, 0.9]]),
///         text: None,
///     },
///     Candidate {
///         id: "doc2",
///         original_score: 0.7,
///         dense_embedding: None,
///         token_embeddings: Some(&[vec![0.5, 0.5]]),
///         text: None,
///     },
/// ];
///
/// let input = RerankerInput {
///     query_dense: None,
///     query_tokens: Some(&query_tokens),
///     candidates,
/// };
///
/// let results = rerank_batch(input, RerankMethod::MaxSim, 10);
/// ```
#[must_use]
pub fn rerank_batch<'a, K: Clone>(
    input: RerankerInput<'a, K>,
    method: RerankMethod,
    top_k: usize,
) -> Vec<RankedResult<K>> {
    let mut results: Vec<RankedResult<K>> = input
        .candidates
        .into_iter()
        .map(|candidate| {
            let score = match method {
                RerankMethod::DenseCosine => {
                    if let (Some(q), Some(d)) = (input.query_dense, candidate.dense_embedding) {
                        simd::cosine(q, d)
                    } else {
                        candidate.original_score
                    }
                }
                RerankMethod::MaxSim => {
                    if let (Some(q_tokens), Some(d_tokens)) =
                        (input.query_tokens, candidate.token_embeddings)
                    {
                        simd::maxsim_vecs(q_tokens, d_tokens)
                    } else {
                        candidate.original_score
                    }
                }
                RerankMethod::MaxSimCosine => {
                    if let (Some(q_tokens), Some(d_tokens)) =
                        (input.query_tokens, candidate.token_embeddings)
                    {
                        simd::maxsim_cosine_vecs(q_tokens, d_tokens)
                    } else {
                        candidate.original_score
                    }
                }
                RerankMethod::MaxSimWeighted => {
                    // Weighted MaxSim requires weights parameter - use maxsim_weighted directly
                    candidate.original_score
                }
            };

            RankedResult {
                id: candidate.id,
                score,
                original_score: candidate.original_score,
                rank: 0, // Will be set after sorting
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Set ranks
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    // Apply top_k
    results.truncate(top_k);
    results
}

/// Token weight utilities for MaxSim.
pub mod weights {
    use std::collections::HashMap;

    /// Compute IDF-based weights from corpus statistics.
    ///
    /// Formula: `idf(t) = log(N / df(t))` where N = total documents, df = document frequency.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs for the query
    /// * `idf_table` - Precomputed IDF values for each token ID
    /// * `default_idf` - IDF value for tokens not in table (default: log(N))
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_refine::explain::weights::idf_weights;
    /// use std::collections::HashMap;
    ///
    /// let idf_table = HashMap::from([
    ///     (100, 2.0),  // rare token, high IDF
    ///     (200, 0.5),  // common token, low IDF
    /// ]);
    ///
    /// let weights = idf_weights(&[100, 200], &idf_table, 3.0);
    /// assert_eq!(weights.len(), 2);
    /// assert!(weights[0] > weights[1]); // rare token gets higher weight
    /// ```
    #[must_use]
    pub fn idf_weights(
        token_ids: &[u32],
        idf_table: &HashMap<u32, f32>,
        default_idf: f32,
    ) -> Vec<f32> {
        token_ids
            .iter()
            .map(|&id| idf_table.get(&id).copied().unwrap_or(default_idf))
            .collect()
    }

    /// Compute attention-based weights from transformer attention scores.
    ///
    /// Normalizes attention scores to sum to 1.0.
    ///
    /// # Arguments
    ///
    /// * `attention_scores` - Raw attention scores from transformer last layer
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_refine::explain::weights::attention_weights;
    ///
    /// let attention = vec![0.1, 0.3, 0.6];
    /// let weights = attention_weights(&attention);
    /// let sum: f32 = weights.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn attention_weights(attention_scores: &[f32]) -> Vec<f32> {
        if attention_scores.is_empty() {
            return Vec::new();
        }

        let sum: f32 = attention_scores.iter().sum();
        if sum.abs() < 1e-9 {
            // All zeros, return uniform weights
            return vec![1.0 / attention_scores.len() as f32; attention_scores.len()];
        }

        attention_scores.iter().map(|&s| s / sum).collect()
    }

    /// Load learned weights from a simple format (one weight per line).
    ///
    /// # Format
    ///
    /// Each line contains a single f32 weight. Empty lines are ignored.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or contains invalid floats.
    pub fn load_learned_weights(path: &std::path::Path) -> std::io::Result<Vec<f32>> {
        let content = std::fs::read_to_string(path)?;
        let weights: Result<Vec<f32>, _> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f32>())
            .collect();

        weights.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxsim_explained_basic() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];

        let explanation = maxsim_explained(&query, &doc, None, None, false);

        assert_eq!(explanation.token_contributions.len(), 2);
        assert!((explanation.total_score - 1.8).abs() < 0.1);
    }

    #[test]
    fn maxsim_explained_with_texts() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];

        let explanation = maxsim_explained(
            &query,
            &doc,
            Some(&["capital", "France"]),
            Some(&["Paris", "capital", "France"]),
            false,
        );

        assert_eq!(explanation.token_contributions.len(), 2);
        assert_eq!(
            explanation.token_contributions[0].query_token_text,
            Some("capital".to_string())
        );
    }

    #[test]
    fn rerank_batch_maxsim() {
        let query_tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let doc1_tokens = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let doc2_tokens = vec![vec![0.5, 0.5]];

        let candidates = vec![
            Candidate {
                id: "doc1",
                original_score: 0.8,
                dense_embedding: None,
                token_embeddings: Some(&doc1_tokens),
                text: None,
            },
            Candidate {
                id: "doc2",
                original_score: 0.7,
                dense_embedding: None,
                token_embeddings: Some(&doc2_tokens),
                text: None,
            },
        ];

        let input = RerankerInput {
            query_dense: None,
            query_tokens: Some(&query_tokens),
            candidates,
        };

        let results = rerank_batch(input, RerankMethod::MaxSim, 10);
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn weights_idf() {
        use std::collections::HashMap;
        let idf_table = HashMap::from([(100, 2.0), (200, 0.5)]);
        let weights = weights::idf_weights(&[100, 200], &idf_table, 1.0);
        assert_eq!(weights.len(), 2);
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn weights_attention() {
        let attention = vec![0.1, 0.3, 0.6];
        let weights = weights::attention_weights(&attention);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
